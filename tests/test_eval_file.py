from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import eval_file as eval_module
import pytest
from typer.testing import CliRunner


def test_testset_to_entries_maps_ragas_rows():
    rows = [
        {
            "user_input": "What is Atlas?",
            "reference": "Atlas is the codename.",
            "reference_contexts": ["Atlas is the internal codename for the project."],
        }
    ]

    entries = eval_module._testset_to_entries(rows)

    assert entries == [
        {
            "question": "What is Atlas?",
            "ground_truth_answer": "Atlas is the codename.",
            "chunk_text_snippet": "Atlas is the internal codename for the project.",
        }
    ]


def test_generate_dataset_file_saves_generated_entries(monkeypatch, tmp_path):
    source_file = tmp_path / "source.txt"
    source_file.write_text("Atlas is the codename.", encoding="utf-8")
    output_path = tmp_path / "generated.json"

    config = Namespace(
        source_path=str(source_file),
        llm_model="qwen3:8b",
        embedding_model="nomic-embed-text",
        ollama_base_url="http://127.0.0.1:11434",
    )

    class FakeService:
        def configure_runtime(self):
            eval_module.Settings.llm = "fake-llm"
            eval_module.Settings.embed_model = "fake-embed"

        def load_source_documents(self):
            return [
                type(
                    "Doc",
                    (),
                    {
                        "text": "Atlas is the codename.",
                        "metadata": {"file_path": str(source_file)},
                    },
                )()
            ]

    class FakeGenerator:
        @classmethod
        def from_llama_index(cls, llm, embedding_model):
            assert llm == "fake-llm"
            assert embedding_model == "fake-embed"
            return cls()

        def generate_with_llamaindex_docs(self, docs, testset_size):
            assert len(docs) == 1
            assert testset_size == 2
            return [
                {
                    "question": "What is Atlas?",
                    "ground_truth_answer": "Atlas is the codename.",
                    "contexts": ["Atlas is the codename."],
                }
            ]

    monkeypatch.setattr(eval_module, "_build_rag_service", lambda project_root, args: (config, FakeService()))
    monkeypatch.setattr(eval_module, "_load_testset_generator", lambda: FakeGenerator)
    monkeypatch.setattr(eval_module, "Settings", SimpleNamespace(llm=None, embed_model=None))

    args = Namespace(output=str(output_path), testset_size=2, rebuild_index=False)
    result = eval_module.generate_dataset_file(tmp_path, args)

    assert result == output_path
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload[0]["question"] == "What is Atlas?"


def test_generate_and_evaluate_uses_generated_dataset(monkeypatch, tmp_path):
    generated_path = tmp_path / "dataset.json"
    generated_path.write_text("[]", encoding="utf-8")
    captured = {}

    monkeypatch.setattr(eval_module, "generate_dataset_file", lambda project_root, args: generated_path)

    def fake_run_evaluation(args):
        captured["dataset"] = args.dataset
        return 0

    monkeypatch.setattr(eval_module, "run_evaluation", fake_run_evaluation)

    args = Namespace(source_path=str(tmp_path / "source.txt"))
    exit_code = eval_module.run_generate_and_evaluate(args)

    assert exit_code == 0
    assert captured["dataset"] == str(generated_path)


def test_run_evaluation_requires_explicit_source_path():
    args = Namespace(dataset="eval_dataset.json", source_path=None)

    with pytest.raises(SystemExit, match="source_path is required"):
        eval_module.run_evaluation(args)


def test_run_generate_and_evaluate_requires_explicit_source_path():
    args = Namespace(source_path=None)

    with pytest.raises(SystemExit, match="source_path is required"):
        eval_module.run_generate_and_evaluate(args)


def test_run_evaluation_uses_llamaindex_wrappers_for_ragas(monkeypatch, tmp_path):
    dataset_path = tmp_path / "dataset.json"
    dataset_path.write_text(
        json.dumps(
            [
                {
                    "question": "What is Atlas?",
                    "ground_truth_answer": "Atlas is the codename.",
                    "chunk_text_snippet": "Atlas is the codename.",
                }
            ]
        ),
        encoding="utf-8",
    )

    config = Namespace(
        source_path=str(tmp_path / "source.txt"),
        llm_model="qwen3:8b",
        embedding_model="BAAI/bge-small-en-v1.5",
        ollama_base_url="http://127.0.0.1:11434",
        chunk_size=256,
        chunk_overlap=32,
        similarity_top_k=2,
    )
    captured = {}

    class FakeService:
        pass

    class FakeLLMWrapper:
        def __init__(self, llm):
            self.llm = llm

    class FakeEmbeddingsWrapper:
        def __init__(self, embeddings):
            self.embeddings = embeddings

    def fake_evaluate(**kwargs):
        captured.update(kwargs)
        return {"faithfulness": 1.0}

    monkeypatch.setattr(eval_module, "_build_rag_service", lambda project_root, args: (config, FakeService()))
    monkeypatch.setattr(
        eval_module,
        "_load_ragas",
        lambda: (fake_evaluate, [], None, None),
    )
    monkeypatch.setattr(
        eval_module,
        "_load_ragas_wrappers",
        lambda: (FakeLLMWrapper, FakeEmbeddingsWrapper),
    )
    monkeypatch.setattr(
        eval_module,
        "query_dataset",
        lambda service, dataset: [{"answer": "Atlas", "contexts": ["Atlas is the codename."], "sources": []}],
    )
    monkeypatch.setattr(
        eval_module,
        "Settings",
        SimpleNamespace(llm="fake-llm", embed_model="fake-embed"),
    )

    args = Namespace(
        dataset=str(dataset_path),
        output_dir=str(tmp_path / "eval_results"),
        source_path=str(tmp_path / "source.txt"),
        rebuild_index=False,
        set_baseline=False,
        compare_baseline=False,
    )

    exit_code = eval_module.run_evaluation(args)

    assert exit_code == 0
    assert isinstance(captured["llm"], FakeLLMWrapper)
    assert captured["llm"].llm == "fake-llm"
    assert isinstance(captured["embeddings"], FakeEmbeddingsWrapper)
    assert captured["embeddings"].embeddings == "fake-embed"


def test_typer_help_lists_commands():
    result = CliRunner().invoke(eval_module.app, ["--help"])

    assert result.exit_code == 0
    assert "generate-dataset" in result.stdout
    assert "generate-and-evaluate" in result.stdout
    assert "set-baseline" in result.stdout
