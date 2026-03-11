from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from filepilot.config import AppConfig
import filepilot.rag.service as service_module


def make_config(tmp_path, **overrides) -> AppConfig:
    payload = {
        "source_path": str(tmp_path / "source.txt"),
        "llm_model": "qwen3:latest",
        "embedding_model": "nomic-embed-text",
        "ollama_base_url": "http://127.0.0.1:11434",
    }
    payload.update(overrides)
    return AppConfig.from_payload(tmp_path, payload)


def test_configure_settings_uses_central_config(monkeypatch, tmp_path):
    config = make_config(tmp_path)
    service = service_module.RagService(config)
    fake_settings = SimpleNamespace(llm=None, embed_model=None, text_splitter=None)

    class FakeOllama:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class FakeEmbedding:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class FakeSplitter:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    monkeypatch.setattr(service_module, "Ollama", FakeOllama)
    monkeypatch.setattr(service_module, "HuggingFaceEmbedding", FakeEmbedding)
    monkeypatch.setattr(service_module, "SentenceSplitter", FakeSplitter)
    monkeypatch.setattr(service_module, "Settings", fake_settings)

    service._configure_settings()

    assert isinstance(fake_settings.llm, FakeOllama)
    assert fake_settings.llm.kwargs["model"] == config.llm_model
    assert isinstance(fake_settings.embed_model, FakeEmbedding)
    assert fake_settings.embed_model.kwargs["model_name"] == config.embedding_model
    assert isinstance(fake_settings.text_splitter, FakeSplitter)
    assert fake_settings.text_splitter.kwargs["chunk_size"] == config.chunk_size


def test_configure_settings_ignores_unconfigured_settings_getters(monkeypatch, tmp_path):
    config = make_config(tmp_path)
    service = service_module.RagService(config)

    class FakeSettings:
        def __getattr__(self, name):
            raise ImportError("default OpenAI resolver should not be required")

    class FakeOllama:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.model = kwargs["model"]
            self.base_url = kwargs["base_url"]

    class FakeEmbedding:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.model_name = kwargs["model_name"]

    class FakeSplitter:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.chunk_size = kwargs["chunk_size"]
            self.chunk_overlap = kwargs["chunk_overlap"]

    monkeypatch.setattr(service_module, "Settings", FakeSettings())
    monkeypatch.setattr(service_module, "Ollama", FakeOllama)
    monkeypatch.setattr(service_module, "HuggingFaceEmbedding", FakeEmbedding)
    monkeypatch.setattr(service_module, "SentenceSplitter", FakeSplitter)

    service._configure_settings()

    assert service_module.Settings.llm.model == config.llm_model
    assert service_module.Settings.embed_model.model_name == config.embedding_model
    assert service_module.Settings.text_splitter.chunk_size == config.chunk_size


def test_load_documents_uses_file_reader_for_single_file(monkeypatch, tmp_path):
    config = make_config(tmp_path)
    service = service_module.RagService(config)
    source_file = tmp_path / "note.txt"
    source_file.write_text("hello", encoding="utf-8")
    captured = {}

    class FakeReader:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def load_data(self):
            return ["document"]

    monkeypatch.setattr(service_module, "SimpleDirectoryReader", FakeReader)

    documents = service._load_documents(source_file)

    assert documents == ["document"]
    assert captured["input_files"] == [str(source_file)]
    assert captured["filename_as_id"] is True


def test_load_source_documents_uses_configured_path(monkeypatch, tmp_path):
    source_file = tmp_path / "source.txt"
    source_file.write_text("hello", encoding="utf-8")
    config = make_config(tmp_path, source_path=str(source_file))
    service = service_module.RagService(config)
    captured = {}

    def fake_load_documents(path):
        captured["path"] = path
        return ["document"]

    monkeypatch.setattr(service, "_load_documents", fake_load_documents)

    documents = service.load_source_documents()

    assert documents == ["document"]
    assert captured["path"] == source_file.resolve()


def test_load_documents_rejects_unsupported_file_type(tmp_path):
    config = make_config(tmp_path, source_path=str(tmp_path / "malware.exe"))
    service = service_module.RagService(config)
    unsupported_file = tmp_path / "malware.exe"
    unsupported_file.write_text("x", encoding="utf-8")

    with pytest.raises(ValueError, match="Unsupported file type"):
        service._load_documents(unsupported_file)


def test_build_index_persists_index_and_counts_unique_files(monkeypatch, tmp_path):
    source_file = tmp_path / "source.txt"
    source_file.write_text("hello", encoding="utf-8")
    config = make_config(tmp_path, source_path=str(source_file))
    service = service_module.RagService(config)

    stale_file = config.index_dir / "stale.txt"
    stale_file.write_text("old", encoding="utf-8")

    doc_a = SimpleNamespace(metadata={"file_path": str(source_file)})
    doc_b = SimpleNamespace(metadata={"file_path": str(source_file)})
    persisted = {}

    class FakeStorageContext:
        def persist(self, persist_dir):
            persisted["persist_dir"] = persist_dir

    class FakeIndex:
        def __init__(self):
            self.storage_context = FakeStorageContext()

    monkeypatch.setattr(service, "_configure_settings", lambda: None)
    monkeypatch.setattr(service, "_load_documents", lambda path: [doc_a, doc_b])
    monkeypatch.setattr(
        service_module.VectorStoreIndex,
        "from_documents",
        staticmethod(lambda documents, **kwargs: FakeIndex()),
    )

    result = service.build_index()

    assert result.source_path == str(source_file.resolve())
    assert result.document_count == 2
    assert result.file_count == 1
    assert persisted["persist_dir"] == str(config.index_dir)
    assert stale_file.exists() is False


def test_query_requires_existing_index(tmp_path):
    config = make_config(tmp_path)
    service = service_module.RagService(config)

    with pytest.raises(ValueError, match="No index found yet"):
        service.query("hello")


def test_ensure_index_loaded_reads_persisted_index(monkeypatch, tmp_path):
    config = make_config(tmp_path)
    service = service_module.RagService(config)
    (config.index_dir / "index_store.json").write_text("{}", encoding="utf-8")
    captured = {}

    monkeypatch.setattr(service, "_configure_settings", lambda: captured.setdefault("configured", True))
    monkeypatch.setattr(
        service_module.StorageContext,
        "from_defaults",
        staticmethod(lambda persist_dir: {"persist_dir": persist_dir}),
    )
    monkeypatch.setattr(
        service_module,
        "load_index_from_storage",
        lambda storage_context: {"loaded_from": storage_context["persist_dir"]},
    )

    loaded_index = service._ensure_index_loaded()

    assert loaded_index == {"loaded_from": str(config.index_dir)}
    assert service._index == loaded_index
    assert captured["configured"] is True


def test_query_returns_answer_and_sources(monkeypatch, tmp_path):
    config = make_config(tmp_path)
    service = service_module.RagService(config)
    source_file = tmp_path / "doc.txt"
    source_file.write_text("content", encoding="utf-8")
    captured = {}

    class FakeEngine:
        def query(self, question):
            captured["question"] = question
            return FakeResponse()

    class FakeIndex:
        def as_query_engine(self, similarity_top_k):
            captured["top_k"] = similarity_top_k
            return FakeEngine()

    class FakeResponse:
        def __str__(self):
            return " answer "

        source_nodes = [
            SimpleNamespace(
                node=SimpleNamespace(
                    metadata={"file_path": str(source_file)},
                    get_content=lambda metadata_mode="none": " preview text ",
                ),
                score=0.75,
            )
        ]

    monkeypatch.setattr(service, "_ensure_index_loaded", lambda: FakeIndex())

    result = service.query("  what is this?  ")

    assert captured["question"] == "what is this?"
    assert captured["top_k"] == config.similarity_top_k
    assert result.answer == "answer"
    assert result.sources[0].file_name == "doc.txt"
    assert result.sources[0].content == "preview text"
    assert result.sources[0].preview == "preview text"


def test_query_wraps_ollama_runtime_errors(monkeypatch, tmp_path):
    config = make_config(tmp_path)
    service = service_module.RagService(config)

    class FakeEngine:
        def query(self, question):
            raise Exception("runner process has terminated")

    class FakeIndex:
        def as_query_engine(self, similarity_top_k):
            return FakeEngine()

    monkeypatch.setattr(service, "_ensure_index_loaded", lambda: FakeIndex())

    with pytest.raises(RuntimeError, match="crashed while generating a response"):
        service.query("hello")


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        ("pull model first", "is not installed"),
        ("connection refused", "Could not reach Ollama"),
    ],
)
def test_format_ollama_error_handles_common_cases(tmp_path, message, expected):
    config = make_config(tmp_path)
    service = service_module.RagService(config)

    formatted = service._format_ollama_error(Exception(message), "demo-model", "chat")

    assert expected in formatted
