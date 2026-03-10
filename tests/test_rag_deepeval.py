from __future__ import annotations

import os
from pathlib import Path

import pytest

pytest.importorskip("deepeval")

from deepeval import assert_test
from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric
from deepeval.test_case import LLMTestCase

from filepilot.config import AppConfig
from filepilot.rag.service import RagService


pytestmark = pytest.mark.deepeval

RAG_EVAL_SOURCE = Path(__file__).parent / "fixtures" / "rag_eval_source.txt"


def make_config(tmp_path, **overrides) -> AppConfig:
    payload = {
        "source_path": str(RAG_EVAL_SOURCE),
        "llm_model": "qwen3:latest",
        "embedding_model": "nomic-embed-text",
        "ollama_base_url": "http://127.0.0.1:11434",
    }
    payload.update(overrides)
    return AppConfig.from_payload(tmp_path, payload)


def require_live_deepeval() -> None:
    if os.getenv("RUN_DEEPEVAL_RAG") != "1":
        pytest.skip(
            "Set RUN_DEEPEVAL_RAG=1 to run live DeepEval RAG evaluations."
        )


@pytest.mark.parametrize(
    ("question", "expected_output"),
    [
        ("What is the project codename?", "The project codename is Atlas."),
        ("Where is the deployment region?", "The deployment region is Phoenix."),
    ],
)
def test_rag_pipeline_with_deepeval(tmp_path, question, expected_output):
    require_live_deepeval()

    config = make_config(tmp_path)
    service = RagService(config)
    service.build_index()
    result = service.query(question)

    assert result.sources, "The RAG pipeline returned no retrieval context."

    test_case = LLMTestCase(
        input=question,
        actual_output=result.answer,
        expected_output=expected_output,
        retrieval_context=[source.preview for source in result.sources if source.preview],
    )
    metrics = [
        AnswerRelevancyMetric(threshold=0.5),
        FaithfulnessMetric(threshold=0.5),
    ]

    assert_test(test_case, metrics)
