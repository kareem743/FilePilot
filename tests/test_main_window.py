from __future__ import annotations

from types import SimpleNamespace

from PyQt6.QtCore import Qt

from filepilot.config import AppConfig
from filepilot.rag.service import BuildResult, QueryResult, QuerySource
import filepilot.ui.main_window as window_module


def make_config(tmp_path, **overrides) -> AppConfig:
    payload = {
        "source_path": "",
        "llm_model": "qwen3:latest",
        "embedding_model": "nomic-embed-text",
        "window_width": 420,
        "window_height": 520,
        "window_margin": 24,
        "always_on_top": True,
    }
    payload.update(overrides)
    return AppConfig.from_payload(tmp_path, payload)


class DummyConfigStore:
    def __init__(self):
        self.saved_payloads = []

    def save(self, config):
        self.saved_payloads.append(config.to_payload())


class IdleService:
    def __init__(self, config):
        self.config = config

    def has_persisted_index(self):
        return False


def test_task_runner_emits_result_and_finished(qapp):
    runner = window_module.TaskRunner(lambda: "ok")
    results = []
    finished = []

    runner.signals.result.connect(results.append)
    runner.signals.finished.connect(lambda: finished.append(True))
    runner.run()

    assert results == ["ok"]
    assert finished == [True]


def test_task_runner_emits_error_and_finished(qapp):
    runner = window_module.TaskRunner(lambda: (_ for _ in ()).throw(ValueError("bad")))
    errors = []
    finished = []

    runner.signals.error.connect(errors.append)
    runner.signals.finished.connect(lambda: finished.append(True))
    runner.run()

    assert errors and "ValueError: bad" in errors[0]
    assert finished == [True]


def test_main_window_uses_central_config_values(monkeypatch, tmp_path, qapp):
    config = make_config(
        tmp_path,
        source_path="docs/manual.pdf",
        llm_model="qwen3:8b",
        embedding_model="nomic-embed-text",
        chunk_size=512,
        similarity_top_k=5,
    )
    store = DummyConfigStore()

    class PersistedService(IdleService):
        def has_persisted_index(self):
            return True

    monkeypatch.setattr(window_module, "RagService", PersistedService)

    window = window_module.MainWindow(config, store)

    assert window.path_input.text() == "docs/manual.pdf"
    assert window.llm_model_input.text() == "qwen3:8b"
    assert window.embedding_model_input.text() == "nomic-embed-text"
    assert "Setting" in [window.tabs.tabText(index) for index in range(window.tabs.count())]
    assert window.chunk_size_input.text() == "512"
    assert window.similarity_top_k_input.text() == "5"
    assert "Persisted index found" in window.status_label.text()
    assert window.windowFlags() & Qt.WindowType.WindowStaysOnTopHint
    window.close()


def test_persist_config_updates_config_store(monkeypatch, tmp_path, qapp):
    config = make_config(tmp_path)
    store = DummyConfigStore()
    monkeypatch.setattr(window_module, "RagService", IdleService)
    window = window_module.MainWindow(config, store)

    window.path_input.setText("docs/new.txt")
    window.llm_model_input.setText("qwen3:14b")
    window.embedding_model_input.setText("custom-embed")
    window._persist_config()

    assert config.source_path == "docs/new.txt"
    assert config.llm_model == "qwen3:14b"
    assert config.embedding_model == "custom-embed"
    assert store.saved_payloads[-1]["llm_model"] == "qwen3:14b"
    assert window.settings_llm_model_input.text() == "qwen3:14b"
    window.close()


def test_save_settings_updates_config_store(monkeypatch, tmp_path, qapp):
    config = make_config(tmp_path)
    store = DummyConfigStore()
    messages = []

    monkeypatch.setattr(window_module, "RagService", IdleService)
    monkeypatch.setattr(
        window_module.QMessageBox,
        "critical",
        lambda parent, title, message: messages.append(message),
    )
    window = window_module.MainWindow(config, store)

    window.settings_llm_model_input.setText("qwen3:14b")
    window.settings_embedding_model_input.setText("custom-embed")
    window.ollama_base_url_input.setText("http://localhost:11434")
    window.chunk_size_input.setText("256")
    window.chunk_overlap_input.setText("32")
    window.similarity_top_k_input.setText("4")
    window.supported_extensions_input.setText(".txt, .md, .py")
    window._save_settings()

    assert messages == []
    assert config.llm_model == "qwen3:14b"
    assert config.embedding_model == "custom-embed"
    assert config.ollama_base_url == "http://localhost:11434"
    assert config.chunk_size == 256
    assert config.chunk_overlap == 32
    assert config.similarity_top_k == 4
    assert config.supported_extensions == (".txt", ".md", ".py")
    assert window.llm_model_input.text() == "qwen3:14b"
    assert store.saved_payloads[-1]["chunk_size"] == 256
    window.close()


def test_start_index_build_validates_missing_path(monkeypatch, tmp_path, qapp):
    config = make_config(tmp_path, source_path="")
    store = DummyConfigStore()
    messages = []

    monkeypatch.setattr(window_module, "RagService", IdleService)
    monkeypatch.setattr(
        window_module.QMessageBox,
        "critical",
        lambda parent, title, message: messages.append(message),
    )
    window = window_module.MainWindow(config, store)

    window.path_input.setText("")
    window._start_index_build()

    assert messages == ["Choose a file or folder first."]
    assert window._ui_state.indexing is False
    window.close()


def test_start_index_build_runs_service_and_updates_ui(monkeypatch, tmp_path, qapp):
    source_file = tmp_path / "source.txt"
    source_file.write_text("hello", encoding="utf-8")
    config = make_config(tmp_path, source_path=str(source_file))
    store = DummyConfigStore()

    class BuildService(IdleService):
        def build_index(self):
            return BuildResult(
                source_path=str(source_file),
                document_count=3,
                file_count=1,
                llm_model=self.config.llm_model,
                embedding_model=self.config.embedding_model,
            )

    monkeypatch.setattr(window_module, "RagService", BuildService)
    window = window_module.MainWindow(config, store)
    window._thread_pool = SimpleNamespace(start=lambda runner: runner.run())

    window._start_index_build()

    assert "Index ready." in window.load_summary.toPlainText()
    assert window.status_label.text() == "Index built successfully."
    assert window.tabs.currentWidget() is window.ask_tab
    assert store.saved_payloads[-1]["source_path"] == str(source_file)
    window.close()


def test_start_query_validates_empty_question(monkeypatch, tmp_path, qapp):
    config = make_config(tmp_path)
    store = DummyConfigStore()
    messages = []

    monkeypatch.setattr(window_module, "RagService", IdleService)
    monkeypatch.setattr(
        window_module.QMessageBox,
        "critical",
        lambda parent, title, message: messages.append(message),
    )
    window = window_module.MainWindow(config, store)

    window.question_input.setPlainText("")
    window._start_query()

    assert messages == ["Enter a question first."]
    assert window._ui_state.asking is False
    window.close()


def test_start_query_runs_service_and_populates_outputs(monkeypatch, tmp_path, qapp):
    config = make_config(tmp_path)
    store = DummyConfigStore()

    class QueryService(IdleService):
        def query(self, question):
            return QueryResult(
                answer=f"Answer for {question}",
                sources=[
                    QuerySource(
                        file_name="doc.txt",
                        file_path="docs/doc.txt",
                        score=0.9,
                        content="source preview",
                        preview="source preview",
                    )
                ],
            )

    monkeypatch.setattr(window_module, "RagService", QueryService)
    monkeypatch.setattr(
        window_module.QMessageBox,
        "critical",
        lambda parent, title, message: None,
    )
    window = window_module.MainWindow(config, store)
    window._thread_pool = SimpleNamespace(start=lambda runner: runner.run())

    window.question_input.setPlainText("What is in the file?")
    window._start_query()

    assert "Answer for What is in the file?" in window.answer_output.toPlainText()
    assert "File: doc.txt" in window.sources_output.toPlainText()
    assert window.status_label.text() == "Answer ready."
    window.close()
