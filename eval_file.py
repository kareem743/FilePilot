from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import typer
from llama_index.core import Settings

from filepilot.config import AppConfig, ConfigStore
from filepilot.rag.service import QueryResult, RagService


app = typer.Typer(
    help="Generate benchmark datasets and evaluate the local FilePilot RagService pipeline.",
)

DATASET_REQUIRED_FIELDS = {"question", "ground_truth_answer"}


def _load_ragas() -> tuple[Any, list[Any], Any | None, Any | None]:
    try:
        from ragas import evaluate
    except ImportError as exc:
        raise SystemExit(
            "ragas is not installed. Install it first with: pip install ragas datasets"
        ) from exc

    metrics: list[Any] = []

    try:
        from ragas.metrics import AnswerRelevancy, ContextPrecision, ContextRecall, Faithfulness

        metrics = [
            Faithfulness(),
            AnswerRelevancy(),
            ContextPrecision(),
            ContextRecall(),
        ]
    except ImportError:
        from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness

        metrics = [
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        ]

    try:
        from ragas.dataset_schema import EvaluationDataset, SingleTurnSample

        return evaluate, metrics, EvaluationDataset, SingleTurnSample
    except ImportError:
        return evaluate, metrics, None, None


def _load_ragas_wrappers() -> tuple[Any | None, Any | None]:
    try:
        from ragas.integrations.llama_index import (
            LlamaIndexEmbeddingsWrapper,
            LlamaIndexLLMWrapper,
        )

        return LlamaIndexLLMWrapper, LlamaIndexEmbeddingsWrapper
    except ImportError:
        return None, None


def _load_testset_generator() -> Any:
    try:
        from ragas.testset import TestsetGenerator
    except ImportError as exc:
        raise SystemExit(
            "RAGAS testset generation is not installed. Install it first with: "
            "pip install ragas datasets"
        ) from exc
    return TestsetGenerator


def load_dataset(path: Path) -> list[dict[str, Any]]:
    raw = path.read_text(encoding="utf-8").strip()
    if not raw:
        return []

    if raw.lstrip().startswith("["):
        data = json.loads(raw)
        if not isinstance(data, list):
            raise ValueError("Dataset JSON must be a list of objects.")
        entries = data
    else:
        entries = []
        for line in raw.splitlines():
            line = line.strip()
            if not line:
                continue
            entries.append(json.loads(line))

    for idx, entry in enumerate(entries, start=1):
        missing = DATASET_REQUIRED_FIELDS - set(entry)
        if missing:
            raise ValueError(f"Dataset entry {idx} missing fields: {sorted(missing)}")

    return entries


def save_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def resolve_config(project_root: Path, args: SimpleNamespace) -> AppConfig:
    config = ConfigStore(project_root).load()
    payload = config.to_payload()

    if args.source_path is not None:
        payload["source_path"] = args.source_path
    if args.llm_model is not None:
        payload["llm_model"] = args.llm_model
    if args.embedding_model is not None:
        payload["embedding_model"] = args.embedding_model
    if args.ollama_base_url is not None:
        payload["ollama_base_url"] = args.ollama_base_url
    if args.top_k is not None:
        payload["similarity_top_k"] = args.top_k
    if args.chunk_size is not None:
        payload["chunk_size"] = args.chunk_size
    if args.chunk_overlap is not None:
        payload["chunk_overlap"] = args.chunk_overlap

    return AppConfig.from_payload(project_root, payload)


def _build_rag_service(project_root: Path, args: SimpleNamespace) -> tuple[AppConfig, RagService]:
    config = resolve_config(project_root, args)
    service = RagService(config)
    if args.rebuild_index or not service.has_persisted_index():
        if not config.source_path:
            raise SystemExit("No source_path configured. Set it in config.json or pass --source-path.")
        service.build_index()
    return config, service


def _documents_to_llamaindex(service: RagService) -> list[Any]:
    documents = service.load_source_documents()
    if not documents:
        raise SystemExit("No supported documents were found for dataset generation.")
    return documents


def _testset_to_entries(testset: Any) -> list[dict[str, Any]]:
    if hasattr(testset, "to_pandas"):
        rows = testset.to_pandas().to_dict(orient="records")
    elif hasattr(testset, "to_list"):
        rows = testset.to_list()
    elif isinstance(testset, list):
        rows = testset
    else:
        raise SystemExit("Unsupported RAGAS testset result format.")

    entries: list[dict[str, Any]] = []
    for row in rows:
        question = str(
            row.get("user_input")
            or row.get("question")
            or row.get("query")
            or ""
        ).strip()
        ground_truth = str(
            row.get("reference")
            or row.get("ground_truth")
            or row.get("ground_truth_answer")
            or row.get("answer")
            or ""
        ).strip()
        contexts = row.get("reference_contexts") or row.get("contexts") or []

        entry: dict[str, Any] = {
            "question": question,
            "ground_truth_answer": ground_truth,
        }
        if isinstance(contexts, list) and contexts:
            cleaned_contexts = [str(context).strip() for context in contexts if str(context).strip()]
            if cleaned_contexts:
                entry["reference_contexts"] = cleaned_contexts
                entry["chunk_text_snippet"] = cleaned_contexts[0][:1200]
        if question and ground_truth:
            entries.append(entry)

    if not entries:
        raise SystemExit("RAGAS generated a testset, but no usable question/answer pairs were returned.")
    return entries


def generate_dataset_file(project_root: Path, args: SimpleNamespace) -> Path:
    config, service = _build_rag_service(project_root, args)
    if not config.source_path:
        raise SystemExit("No source_path configured. Set it in config.json or pass --source-path.")

    TestsetGenerator = _load_testset_generator()
    service.configure_runtime()
    llamaindex_docs = _documents_to_llamaindex(service)

    generator = TestsetGenerator.from_llama_index(
        llm=Settings.llm,
        embedding_model=Settings.embed_model,
    )

    if hasattr(generator, "generate_with_llamaindex_docs"):
        testset = generator.generate_with_llamaindex_docs(
            llamaindex_docs,
            testset_size=args.testset_size,
        )
    elif hasattr(generator, "generate"):
        testset = generator.generate(
            documents=llamaindex_docs,
            testset_size=args.testset_size,
        )
    else:
        raise SystemExit("This installed RAGAS version does not expose a supported testset generator API.")

    entries = _testset_to_entries(testset)
    output_path = Path(args.output)
    save_json(output_path, entries)
    print(f"Generated dataset: {output_path}")
    print(f"Questions: {len(entries)}")
    print("Note: dataset generation creates a benchmark from source documents. System evaluation happens in the evaluate command via RagService.query().")
    return output_path


def _read_reference_context(entry: dict[str, Any], project_root: Path) -> list[str]:
    stored_contexts = [
        str(context).strip()
        for context in (entry.get("reference_contexts") or [])
        if str(context).strip()
    ]
    if stored_contexts:
        return stored_contexts

    snippet = str(entry.get("chunk_text_snippet", "")).strip()
    if snippet:
        return [snippet]

    contexts: list[str] = []
    for raw_source in entry.get("ground_truth_sources", []) or []:
        candidate = Path(str(raw_source))
        if not candidate.is_absolute():
            candidate = project_root / candidate
        if not candidate.exists() or not candidate.is_file():
            continue
        try:
            text = candidate.read_text(encoding="utf-8", errors="ignore").strip()
        except OSError:
            continue
        if text:
            contexts.append(text[:1200])
    return contexts


def _build_samples(
    dataset: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    project_root: Path,
    evaluation_dataset_cls: Any | None,
    single_turn_sample_cls: Any | None,
) -> Any:
    if evaluation_dataset_cls is not None and single_turn_sample_cls is not None:
        samples = []
        for entry, prediction in zip(dataset, predictions):
            samples.append(
                single_turn_sample_cls(
                    user_input=entry["question"],
                    response=prediction["answer"],
                    retrieved_contexts=prediction["contexts"],
                    reference=entry["ground_truth_answer"],
                    reference_contexts=_read_reference_context(entry, project_root),
                )
            )
        return evaluation_dataset_cls(samples=samples)

    try:
        from datasets import Dataset
    except ImportError as exc:
        raise SystemExit(
            "datasets is not installed. Install it first with: pip install datasets"
        ) from exc

    rows = []
    for entry, prediction in zip(dataset, predictions):
        rows.append(
            {
                "question": entry["question"],
                "answer": prediction["answer"],
                "contexts": prediction["contexts"],
                "ground_truth": entry["ground_truth_answer"],
            }
        )
    return Dataset.from_list(rows)


def query_dataset(service: RagService, dataset: list[dict[str, Any]]) -> list[dict[str, Any]]:
    predictions: list[dict[str, Any]] = []

    for idx, entry in enumerate(dataset, start=1):
        result: QueryResult = service.query(entry["question"])
        predictions.append(
            {
                "id": idx,
                "question": entry["question"],
                "answer": result.answer,
                "contexts": [source.content for source in result.sources if source.content],
                "sources": [source.file_path for source in result.sources if source.file_path],
            }
        )
    return predictions


def normalize_ragas_result(result: Any) -> dict[str, Any]:
    if hasattr(result, "to_pandas"):
        frame = result.to_pandas()
        records = frame.to_dict(orient="records")
        averages: dict[str, float] = {}
        for column in frame.columns:
            numeric = frame[column]
            if getattr(numeric, "dtype", None) is not None and str(numeric.dtype) != "object":
                averages[column] = round(float(numeric.mean()), 4)
        return {"rows": records, "averages": averages}

    if hasattr(result, "to_dict"):
        data = result.to_dict()
        averages: dict[str, float] = {}
        for key, value in data.items():
            if isinstance(value, (int, float)):
                averages[key] = round(float(value), 4)
        return {"rows": [data], "averages": averages}

    return {"rows": [{"result": str(result)}], "averages": {}}


def compare_results(baseline: dict[str, Any], latest: dict[str, Any]) -> tuple[str, int]:
    lines = [
        "=== RAGAS Comparison Report ===",
        f"Baseline: {baseline.get('run_timestamp', 'unknown')}",
        f"Current:  {latest.get('run_timestamp', 'unknown')}",
        "",
        "Metric                    Baseline    Current    Delta",
    ]

    exit_code = 0
    baseline_scores = baseline.get("ragas_scores", {})
    latest_scores = latest.get("ragas_scores", {})
    metric_names = sorted(set(baseline_scores) | set(latest_scores))

    for name in metric_names:
        base_value = float(baseline_scores.get(name, 0.0))
        current_value = float(latest_scores.get(name, 0.0))
        delta = current_value - base_value
        lines.append(f"{name:<24} {base_value:>8.4f}    {current_value:>7.4f}    {delta:>+7.4f}")
        if delta < -0.10:
            exit_code = 1

    if exit_code:
        lines.append("")
        lines.append("Regression detected: at least one RAGAS metric dropped by more than 0.10.")

    return "\n".join(lines), exit_code


def run_evaluation(args: SimpleNamespace) -> int:
    project_root = Path(__file__).resolve().parent
    if not getattr(args, "source_path", None):
        raise SystemExit("source_path is required. Pass --source-path explicitly when running evaluation.")
    config, service = _build_rag_service(project_root, args)
    dataset = load_dataset(Path(args.dataset))

    if not dataset:
        print("Dataset is empty.")
        return 2

    predictions = query_dataset(service, dataset)
    evaluate, metrics, evaluation_dataset_cls, single_turn_sample_cls = _load_ragas()
    ragas_dataset = _build_samples(
        dataset,
        predictions,
        project_root,
        evaluation_dataset_cls,
        single_turn_sample_cls,
    )
    llm_wrapper_cls, embeddings_wrapper_cls = _load_ragas_wrappers()
    evaluate_kwargs: dict[str, Any] = {
        "dataset": ragas_dataset,
        "metrics": metrics,
    }
    if llm_wrapper_cls is not None and Settings.llm is not None:
        evaluate_kwargs["llm"] = llm_wrapper_cls(Settings.llm)
    if embeddings_wrapper_cls is not None and Settings.embed_model is not None:
        evaluate_kwargs["embeddings"] = embeddings_wrapper_cls(Settings.embed_model)

    ragas_result = evaluate(**evaluate_kwargs)
    normalized = normalize_ragas_result(ragas_result)

    output_dir = Path(args.output_dir)
    results = {
        "run_timestamp": datetime.now().isoformat(timespec="seconds"),
        "config": {
            "source_path": config.source_path,
            "llm_model": config.llm_model,
            "embedding_model": config.embedding_model,
            "ollama_base_url": config.ollama_base_url,
            "chunk_size": config.chunk_size,
            "chunk_overlap": config.chunk_overlap,
            "similarity_top_k": config.similarity_top_k,
        },
        "ragas_scores": normalized["averages"],
        "ragas_rows": normalized["rows"],
        "predictions": predictions,
    }

    latest_path = output_dir / "latest.json"
    save_json(latest_path, results)

    print("=== RAGAS Evaluation Report ===")
    print(f"Run: {results['run_timestamp']}")
    print(f"Questions: {len(dataset)}")
    for metric_name, score in results["ragas_scores"].items():
        print(f"{metric_name}: {score:.4f}")
    print(f"Saved: {latest_path}")

    if args.set_baseline:
        baseline_path = output_dir / "baseline.json"
        save_json(baseline_path, results)
        print(f"Baseline saved: {baseline_path}")

    if args.compare_baseline:
        baseline_path = output_dir / "baseline.json"
        if not baseline_path.exists():
            print("Baseline not found. Run again with --set-baseline first.")
            return 2
        baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
        report, exit_code = compare_results(baseline, results)
        report_path = output_dir / "comparison_report.txt"
        report_path.write_text(report, encoding="utf-8")
        print("")
        print(report)
        print(f"Saved: {report_path}")
        return exit_code

    return 0


def run_set_baseline(args: SimpleNamespace) -> int:
    results_path = Path(args.results)
    if not results_path.exists():
        print(f"Results file not found: {results_path}")
        return 2

    payload = json.loads(results_path.read_text(encoding="utf-8"))
    baseline_path = Path(args.output_dir) / "baseline.json"
    save_json(baseline_path, payload)
    print(f"Baseline saved: {baseline_path}")
    return 0


def run_generate_dataset(args: SimpleNamespace) -> int:
    project_root = Path(__file__).resolve().parent
    generate_dataset_file(project_root, args)
    return 0


def run_generate_and_evaluate(args: SimpleNamespace) -> int:
    project_root = Path(__file__).resolve().parent
    if not getattr(args, "source_path", None):
        raise SystemExit("source_path is required. Pass --source-path explicitly when generating and evaluating.")
    dataset_path = generate_dataset_file(project_root, args)
    args.dataset = str(dataset_path)
    return run_evaluation(args)


def _make_args(**kwargs: Any) -> SimpleNamespace:
    return SimpleNamespace(**kwargs)


@app.command("evaluate", help="Run evaluation on an existing dataset file using RagService.query().")
def evaluate_cmd(
    dataset: Path = typer.Option(..., "--dataset", help="Path to dataset JSON or JSONL"),
    output_dir: Path = typer.Option("eval_results", "--output-dir", help="Directory for latest.json and reports"),
    source_path: str = typer.Option(..., "--source-path", help="Source file or folder to index for this evaluation run"),
    llm_model: str | None = typer.Option(None, "--llm-model", help="Override config.json llm_model"),
    embedding_model: str | None = typer.Option(None, "--embedding-model", help="Override config.json embedding_model"),
    ollama_base_url: str | None = typer.Option(None, "--ollama-base-url", help="Override config.json ollama_base_url"),
    top_k: int | None = typer.Option(None, "--top-k", help="Override retrieval top-k"),
    chunk_size: int | None = typer.Option(None, "--chunk-size", help="Override chunk size"),
    chunk_overlap: int | None = typer.Option(None, "--chunk-overlap", help="Override chunk overlap"),
    rebuild_index: bool = typer.Option(False, "--rebuild-index", help="Rebuild the persisted index before evaluation"),
    set_baseline: bool = typer.Option(False, "--set-baseline", help="Save this run as eval_results/baseline.json"),
    compare_baseline: bool = typer.Option(False, "--compare-baseline", help="Compare this run to eval_results/baseline.json"),
) -> None:
    code = run_evaluation(
        _make_args(
            dataset=str(dataset),
            output_dir=str(output_dir),
            source_path=source_path,
            llm_model=llm_model,
            embedding_model=embedding_model,
            ollama_base_url=ollama_base_url,
            top_k=top_k,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            rebuild_index=rebuild_index,
            set_baseline=set_baseline,
            compare_baseline=compare_baseline,
        )
    )
    if code != 0:
        raise typer.Exit(code)


@app.command("set-baseline", help="Copy an existing results file to baseline.json.")
def set_baseline_cmd(
    results: Path = typer.Option(..., "--results", help="Path to an existing results JSON file"),
    output_dir: Path = typer.Option("eval_results", "--output-dir", help="Directory to write baseline.json"),
) -> None:
    code = run_set_baseline(_make_args(results=str(results), output_dir=str(output_dir)))
    if code != 0:
        raise typer.Exit(code)


@app.command("generate-dataset", help="Generate a benchmark dataset JSON from source documents. This does not run RagService.query().")
def generate_dataset_cmd(
    output: Path = typer.Option("eval_dataset.json", "--output", help="Path to write the generated dataset JSON"),
    testset_size: int = typer.Option(5, "--testset-size", help="Number of question/answer pairs to generate"),
    source_path: str | None = typer.Option(None, "--source-path", help="Override config.json source_path"),
    llm_model: str | None = typer.Option(None, "--llm-model", help="Override config.json llm_model"),
    embedding_model: str | None = typer.Option(None, "--embedding-model", help="Override config.json embedding_model"),
    ollama_base_url: str | None = typer.Option(None, "--ollama-base-url", help="Override config.json ollama_base_url"),
    top_k: int | None = typer.Option(None, "--top-k", help="Override retrieval top-k"),
    chunk_size: int | None = typer.Option(None, "--chunk-size", help="Override chunk size"),
    chunk_overlap: int | None = typer.Option(None, "--chunk-overlap", help="Override chunk overlap"),
    rebuild_index: bool = typer.Option(False, "--rebuild-index", help="Rebuild the persisted index before generation"),
) -> None:
    code = run_generate_dataset(
        _make_args(
            output=str(output),
            testset_size=testset_size,
            source_path=source_path,
            llm_model=llm_model,
            embedding_model=embedding_model,
            ollama_base_url=ollama_base_url,
            top_k=top_k,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            rebuild_index=rebuild_index,
        )
    )
    if code != 0:
        raise typer.Exit(code)


@app.command("generate-and-evaluate", help="Generate a benchmark dataset, then evaluate RagService on it.")
def generate_and_evaluate_cmd(
    output: Path = typer.Option("eval_dataset.json", "--output", help="Path to write the generated dataset JSON"),
    testset_size: int = typer.Option(5, "--testset-size", help="Number of question/answer pairs to generate"),
    output_dir: Path = typer.Option("eval_results", "--output-dir", help="Directory for latest.json and reports"),
    source_path: str = typer.Option(..., "--source-path", help="Source file or folder to use for dataset generation and evaluation"),
    llm_model: str | None = typer.Option(None, "--llm-model", help="Override config.json llm_model"),
    embedding_model: str | None = typer.Option(None, "--embedding-model", help="Override config.json embedding_model"),
    ollama_base_url: str | None = typer.Option(None, "--ollama-base-url", help="Override config.json ollama_base_url"),
    top_k: int | None = typer.Option(None, "--top-k", help="Override retrieval top-k"),
    chunk_size: int | None = typer.Option(None, "--chunk-size", help="Override chunk size"),
    chunk_overlap: int | None = typer.Option(None, "--chunk-overlap", help="Override chunk overlap"),
    rebuild_index: bool = typer.Option(False, "--rebuild-index", help="Rebuild the persisted index before generation or evaluation"),
    set_baseline: bool = typer.Option(False, "--set-baseline", help="Save this run as eval_results/baseline.json"),
    compare_baseline: bool = typer.Option(False, "--compare-baseline", help="Compare this run to eval_results/baseline.json"),
) -> None:
    code = run_generate_and_evaluate(
        _make_args(
            output=str(output),
            testset_size=testset_size,
            output_dir=str(output_dir),
            source_path=source_path,
            llm_model=llm_model,
            embedding_model=embedding_model,
            ollama_base_url=ollama_base_url,
            top_k=top_k,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            rebuild_index=rebuild_index,
            set_baseline=set_baseline,
            compare_baseline=compare_baseline,
        )
    )
    if code != 0:
        raise typer.Exit(code)


if __name__ == "__main__":
    app()
