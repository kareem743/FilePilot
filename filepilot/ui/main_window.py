from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import traceback
from typing import Any, Callable

from PyQt5.QtCore import QObject, QRunnable, Qt, QThreadPool, pyqtSignal, pyqtSlot
from PyQt5.QtGui import QGuiApplication
from PyQt5.QtWidgets import (
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from filepilot.config import AppConfig, ConfigStore
from filepilot.rag.service import BuildResult, QueryResult, RagService


class WorkerSignals(QObject):
    result = pyqtSignal(object)
    error = pyqtSignal(str)
    finished = pyqtSignal()


class TaskRunner(QRunnable):
    def __init__(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> None:
        super().__init__()
        self._fn = fn
        self._args = args
        self._kwargs = kwargs
        self.signals = WorkerSignals()

    @pyqtSlot()
    def run(self) -> None:
        try:
            result = self._fn(*self._args, **self._kwargs)
        except Exception as exc:
            details = "".join(traceback.format_exception_only(type(exc), exc)).strip()
            self.signals.error.emit(details)
        else:
            self.signals.result.emit(result)
        finally:
            self.signals.finished.emit()


@dataclass
class UiState:
    indexing: bool = False
    asking: bool = False


class MainWindow(QMainWindow):
    def __init__(self, config: AppConfig, config_store: ConfigStore) -> None:
        super().__init__()
        self._config = config
        self._config_store = config_store
        self._service = RagService(config)
        self._thread_pool = QThreadPool.globalInstance()
        self._ui_state = UiState()
        self._positioned = False

        self._build_ui()
        self._restore_state()
        self._refresh_actions()

    def _build_ui(self) -> None:
        self.setWindowTitle("FilePilot")
        self.setMinimumSize(self._config.window_width, self._config.window_height)
        self.resize(self._config.window_width, self._config.window_height)

        flags = self.windowFlags() | Qt.Tool
        if self._config.always_on_top:
            flags |= Qt.WindowStaysOnTopHint
        self.setWindowFlags(flags)

        central = QWidget(self)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        self.tabs = QTabWidget(self)
        self.load_tab = QWidget(self)
        self.ask_tab = QWidget(self)

        self.tabs.addTab(self.load_tab, "Load")
        self.tabs.addTab(self.ask_tab, "Ask")

        self._build_load_tab()
        self._build_ask_tab()

        self.status_label = QLabel("Ready.", self)
        self.status_label.setObjectName("StatusLabel")
        self.status_label.setWordWrap(True)

        layout.addWidget(self.tabs)
        layout.addWidget(self.status_label)
        self.setCentralWidget(central)
        self._apply_theme()

    def _apply_theme(self) -> None:
        self.setStyleSheet(
            """
            QWidget {
                background: #12161d;
                color: #e4ebf5;
            }
            QMainWindow {
                background: #12161d;
            }
            QTabWidget::pane {
                border: 1px solid #2a3240;
                background: #171c24;
                border-radius: 10px;
                margin-top: 6px;
            }
            QTabBar::tab {
                background: #202734;
                color: #94a1b5;
                padding: 8px 16px;
                margin-right: 6px;
                border-top-left-radius: 8px;
                border-top-right-radius: 8px;
                font-weight: 600;
            }
            QTabBar::tab:selected {
                background: #2a3444;
                color: #ffffff;
            }
            QTabBar::tab:hover:!selected {
                background: #252f3d;
            }
            QLineEdit,
            QPlainTextEdit {
                background: #0f1319;
                color: #eef3fb;
                border: 1px solid #313a48;
                border-radius: 8px;
                padding: 8px 10px;
                selection-background-color: #3b82f6;
            }
            QLineEdit:focus,
            QPlainTextEdit:focus {
                border: 1px solid #4f8cff;
            }
            QPushButton {
                border-radius: 8px;
                padding: 9px 14px;
                font-weight: 600;
            }
            QPushButton#PrimaryButton {
                background: #2f6fed;
                color: #ffffff;
                border: 1px solid #3e7fff;
            }
            QPushButton#PrimaryButton:hover {
                background: #3a79f3;
            }
            QPushButton#SecondaryButton {
                background: #1d2430;
                border: 1px solid #2e3746;
                color: #d7dfeb;
            }
            QPushButton#SecondaryButton:hover {
                background: #252d3a;
            }
            QLabel#StatusLabel {
                background: #1a2029;
                border: 1px solid #2d3645;
                border-radius: 8px;
                padding: 10px;
                color: #c9d4e5;
            }
            QPushButton:disabled {
                background: #1a2029;
                border: 1px solid #252d3a;
                color: #6f7c91;
            }
            """
        )

    def _build_load_tab(self) -> None:
        layout = QVBoxLayout(self.load_tab)
        layout.setContentsMargins(6, 6, 6, 6)

        form = QFormLayout()
        form.setSpacing(8)

        self.path_input = QLineEdit(self)
        self.path_input.setText(self._config.source_path)
        self.path_input.setPlaceholderText("Select a file or folder to index")
        form.addRow("Path", self.path_input)

        self.llm_model_input = QLineEdit(self)
        self.llm_model_input.setText(self._config.llm_model)
        form.addRow("LLM", self.llm_model_input)

        self.embedding_model_input = QLineEdit(self)
        self.embedding_model_input.setText(self._config.embedding_model)
        form.addRow("Embedding", self.embedding_model_input)

        layout.addLayout(form)

        path_buttons = QHBoxLayout()
        self.file_button = QPushButton("Browse File", self)
        self.file_button.setObjectName("SecondaryButton")
        self.folder_button = QPushButton("Browse Folder", self)
        self.folder_button.setObjectName("SecondaryButton")
        path_buttons.addWidget(self.file_button)
        path_buttons.addWidget(self.folder_button)
        layout.addLayout(path_buttons)

        self.index_button = QPushButton("Build Index", self)
        self.index_button.setObjectName("PrimaryButton")
        layout.addWidget(self.index_button)

        self.load_summary = QPlainTextEdit(self)
        self.load_summary.setReadOnly(True)
        self.load_summary.setPlaceholderText("Index details will appear here.")
        layout.addWidget(self.load_summary)

        layout.addStretch(1)

        self.file_button.clicked.connect(self._choose_file)
        self.folder_button.clicked.connect(self._choose_folder)
        self.index_button.clicked.connect(self._start_index_build)
        self.path_input.editingFinished.connect(self._persist_config)
        self.llm_model_input.editingFinished.connect(self._persist_config)
        self.embedding_model_input.editingFinished.connect(self._persist_config)

    def _build_ask_tab(self) -> None:
        layout = QVBoxLayout(self.ask_tab)
        layout.setContentsMargins(6, 6, 6, 6)

        self.question_input = QPlainTextEdit(self)
        self.question_input.setPlaceholderText("Ask a question about the indexed content")
        self.question_input.setFixedHeight(90)
        layout.addWidget(self.question_input)

        self.ask_button = QPushButton("Ask", self)
        self.ask_button.setObjectName("PrimaryButton")
        layout.addWidget(self.ask_button)

        self.answer_output = QPlainTextEdit(self)
        self.answer_output.setReadOnly(True)
        self.answer_output.setPlaceholderText("Answer will appear here.")
        layout.addWidget(self.answer_output)

        self.sources_output = QPlainTextEdit(self)
        self.sources_output.setReadOnly(True)
        self.sources_output.setPlaceholderText("Retrieved source snippets will appear here.")
        layout.addWidget(self.sources_output)

        self.ask_button.clicked.connect(self._start_query)

    def showEvent(self, event) -> None:  # noqa: N802
        super().showEvent(event)
        if self._positioned:
            return

        screen = QGuiApplication.primaryScreen()
        if screen is None:
            return

        geometry = screen.availableGeometry()
        margin = self._config.window_margin
        x_pos = geometry.right() - self.width() - margin
        y_pos = geometry.bottom() - self.height() - margin
        self.move(max(geometry.left(), x_pos), max(geometry.top(), y_pos))
        self._positioned = True

    def _restore_state(self) -> None:
        if self._config.source_path and self._service.has_persisted_index():
            self.load_summary.setPlainText(
                f"Saved index found.\nSource: {self._config.source_path}\nLLM: {self._current_llm_model()}\nEmbedding: {self._current_embedding_model()}"
            )
            self.status_label.setText("Persisted index found. You can ask questions now.")
        elif self._service.has_persisted_index():
            self.status_label.setText("Persisted index found. Set the source path if you want to rebuild it.")

    def _choose_file(self) -> None:
        selected, _ = QFileDialog.getOpenFileName(
            self,
            "Choose a file to index",
            self.path_input.text() or str(self._config.project_root),
        )
        if selected:
            self.path_input.setText(selected)
            self._persist_config()

    def _choose_folder(self) -> None:
        selected = QFileDialog.getExistingDirectory(
            self,
            "Choose a folder to index",
            self.path_input.text() or str(self._config.project_root),
        )
        if selected:
            self.path_input.setText(selected)
            self._persist_config()

    def _start_index_build(self) -> None:
        source_path = self.path_input.text().strip()
        if not source_path:
            self._show_error("Choose a file or folder first.")
            return

        path = Path(source_path).expanduser()
        if not path.exists():
            self._show_error(f"Path does not exist: {path}")
            return

        self._ui_state.indexing = True
        self._refresh_actions()
        self.status_label.setText("Building index...")
        self.load_summary.setPlainText("Indexing in progress...")
        self._persist_config()

        worker = TaskRunner(self._service.build_index)
        worker.signals.result.connect(self._on_index_built)
        worker.signals.error.connect(self._on_worker_error)
        worker.signals.finished.connect(self._on_index_finished)
        self._thread_pool.start(worker)

    def _start_query(self) -> None:
        question = self.question_input.toPlainText().strip()
        if not question:
            self._show_error("Enter a question first.")
            return

        self._ui_state.asking = True
        self._refresh_actions()
        self.status_label.setText("Querying index...")
        self.answer_output.setPlainText("Thinking...")
        self.sources_output.clear()
        self._persist_config()

        worker = TaskRunner(self._service.query, question)
        worker.signals.result.connect(self._on_query_finished)
        worker.signals.error.connect(self._on_worker_error)
        worker.signals.finished.connect(self._on_query_cleanup)
        self._thread_pool.start(worker)

    def _on_index_built(self, result: BuildResult) -> None:
        self.path_input.setText(result.source_path)
        self.llm_model_input.setText(result.llm_model)
        self.embedding_model_input.setText(result.embedding_model)
        self._persist_config()
        self.load_summary.setPlainText(
            "\n".join(
                [
                    "Index ready.",
                    f"Source: {result.source_path}",
                    f"Documents loaded: {result.document_count}",
                    f"Files indexed: {result.file_count}",
                    f"LLM: {result.llm_model}",
                    f"Embedding: {result.embedding_model}",
                ]
            )
        )
        self.status_label.setText("Index built successfully.")
        self.tabs.setCurrentWidget(self.ask_tab)

    def _on_query_finished(self, result: QueryResult) -> None:
        self.answer_output.setPlainText(result.answer or "No answer returned.")

        if not result.sources:
            self.sources_output.setPlainText("No source snippets returned.")
        else:
            blocks = []
            for source in result.sources:
                score = f"{source.score:.3f}" if source.score is not None else "n/a"
                blocks.append(
                    "\n".join(
                        [
                            f"File: {source.file_name}",
                            f"Path: {source.file_path or 'Unknown'}",
                            f"Score: {score}",
                            f"Snippet: {source.preview or 'No preview available.'}",
                        ]
                    )
                )
            self.sources_output.setPlainText("\n\n".join(blocks))

        self.status_label.setText("Answer ready.")

    def _on_worker_error(self, message: str) -> None:
        self.status_label.setText("Operation failed.")
        self._show_error(message)
        if self._ui_state.indexing:
            self.load_summary.setPlainText(message)
        if self._ui_state.asking:
            self.answer_output.setPlainText("")
            self.sources_output.setPlainText("")

    def _on_index_finished(self) -> None:
        self._ui_state.indexing = False
        self._refresh_actions()

    def _on_query_cleanup(self) -> None:
        self._ui_state.asking = False
        self._refresh_actions()

    def _refresh_actions(self) -> None:
        busy = self._ui_state.indexing or self._ui_state.asking
        self.index_button.setEnabled(not busy)
        self.ask_button.setEnabled(not busy)
        self.file_button.setEnabled(not busy)
        self.folder_button.setEnabled(not busy)

    def _current_llm_model(self) -> str:
        return self.llm_model_input.text().strip() or self._config.llm_model

    def _current_embedding_model(self) -> str:
        return self.embedding_model_input.text().strip() or self._config.embedding_model

    def _persist_config(self) -> None:
        self._config.source_path = self.path_input.text().strip()
        self._config.llm_model = self._current_llm_model()
        self._config.embedding_model = self._current_embedding_model()
        self.llm_model_input.setText(self._config.llm_model)
        self.embedding_model_input.setText(self._config.embedding_model)
        self._config_store.save(self._config)

    def closeEvent(self, event) -> None:  # noqa: N802
        self._persist_config()
        super().closeEvent(event)

    def _show_error(self, message: str) -> None:
        QMessageBox.critical(self, "FilePilot", message)
