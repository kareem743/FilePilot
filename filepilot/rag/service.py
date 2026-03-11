from __future__ import annotations

from dataclasses import dataclass
import multiprocessing
from pathlib import Path
import shutil

from llama_index.core import Settings, SimpleDirectoryReader, StorageContext, VectorStoreIndex
from llama_index.core import load_index_from_storage
from llama_index.core.node_parser import SentenceSplitter
from llama_index.llms.ollama import Ollama

try:
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
except ImportError:
    HuggingFaceEmbedding = None

from filepilot.config import AppConfig


@dataclass
class BuildResult:
    source_path: str
    document_count: int
    file_count: int
    llm_model: str
    embedding_model: str


@dataclass
class QuerySource:
    file_name: str
    file_path: str
    score: float | None
    preview: str


@dataclass
class QueryResult:
    answer: str
    sources: list[QuerySource]


class RagService:
    def __init__(self, config: AppConfig) -> None:
        self._config = config
        self._index: VectorStoreIndex | None = None

    def configure_runtime(self) -> None:
        self._configure_settings()

    def load_source_documents(self) -> list:
        path = Path(self._config.source_path).expanduser().resolve()
        if not path.exists():
            raise ValueError(f"Path does not exist: {path}")
        return self._load_documents(path)

    def build_index(self) -> BuildResult:
        path = Path(self._config.source_path).expanduser().resolve()
        self.configure_runtime()
        documents = self.load_source_documents()
        if not documents:
            raise ValueError(
                "No supported documents were found. Try a .txt, .md, .pdf, .csv, .json, or .py file."
            )

        self._reset_index_dir()
        try:
            index = VectorStoreIndex.from_documents(documents)
        except Exception as exc:
            raise RuntimeError(
                self._format_ollama_error(
                    exc,
                    model_name=self._config.embedding_model,
                    operation="embedding",
                )
            ) from exc
        index.storage_context.persist(persist_dir=str(self._config.index_dir))
        self._index = index

        file_paths = {
            str(document.metadata.get("file_path", ""))
            for document in documents
            if document.metadata.get("file_path")
        }

        return BuildResult(
            source_path=str(path),
            document_count=len(documents),
            file_count=len(file_paths) or len(documents),
            llm_model=self._config.llm_model,
            embedding_model=self._config.embedding_model,
        )

    def query(self, question: str) -> QueryResult:
        cleaned_question = question.strip()
        if not cleaned_question:
            raise ValueError("Enter a question first.")

        index = self._ensure_index_loaded()
        query_engine = index.as_query_engine(
            similarity_top_k=self._config.similarity_top_k,
        )
        try:
            response = query_engine.query(cleaned_question)
        except Exception as exc:
            raise RuntimeError(
                self._format_ollama_error(
                    exc,
                    model_name=self._config.llm_model,
                    operation="chat",
                )
            ) from exc

        return QueryResult(
            answer=str(response).strip(),
            sources=self._extract_sources(response),
        )

    def has_persisted_index(self) -> bool:
        return self._config.index_dir.exists() and any(self._config.index_dir.iterdir())

    def _ensure_index_loaded(self) -> VectorStoreIndex:
        if self._index is not None:
            self._configure_settings()
            return self._index

        if not self.has_persisted_index():
            raise ValueError("No index found yet. Build an index from the Load tab first.")

        self._configure_settings()
        storage_context = StorageContext.from_defaults(
            persist_dir=str(self._config.index_dir)
        )
        self._index = load_index_from_storage(storage_context)
        return self._index

    def _configure_settings(self) -> None:
        try:
            current_llm = getattr(Settings, "llm", None)
        except Exception:
            current_llm = None
        try:
            current_embed = getattr(Settings, "embed_model", None)
        except Exception:
            current_embed = None
        try:
            current_splitter = getattr(Settings, "text_splitter", None)
        except Exception:
            current_splitter = None

        if (
            isinstance(current_llm, Ollama)
            and current_llm.model == self._config.llm_model
            and getattr(current_llm, "base_url", None) == self._config.ollama_base_url
            and current_embed is not None
            and getattr(current_embed, "model_name", None) == self._config.embedding_model
            and isinstance(current_splitter, SentenceSplitter)
            and getattr(current_splitter, "chunk_size", None) == self._config.chunk_size
            and getattr(current_splitter, "chunk_overlap", None) == self._config.chunk_overlap
        ):
            return

        Settings.llm = Ollama(
            model=self._config.llm_model,
            base_url=self._config.ollama_base_url,
            request_timeout=120.0,
            temperature=0.1,
        )
        if HuggingFaceEmbedding is None:
            raise RuntimeError(
                "Hugging Face embeddings are not installed. Install them with: "
                "pip install llama-index-embeddings-huggingface"
            )
        Settings.embed_model = HuggingFaceEmbedding(
            model_name=self._config.embedding_model,
        )
        Settings.text_splitter = SentenceSplitter(
            chunk_size=self._config.chunk_size,
            chunk_overlap=self._config.chunk_overlap,
        )

    def _load_documents(self, path: Path) -> list:
        if path.is_file():
            if path.suffix.lower() not in self._config.supported_extensions:
                raise ValueError(
                    f"Unsupported file type: {path.suffix}. Supported: {', '.join(self._config.supported_extensions)}"
                )
            reader = SimpleDirectoryReader(input_files=[str(path)], filename_as_id=True)
            return reader.load_data()

        reader = SimpleDirectoryReader(
            input_dir=str(path),
            recursive=True,
            filename_as_id=True,
            required_exts=list(self._config.supported_extensions),
        )
        return reader.load_data(num_workers=max(1, multiprocessing.cpu_count() - 1))

    def _reset_index_dir(self) -> None:
        if self._config.index_dir.exists():
            shutil.rmtree(self._config.index_dir)
        self._config.index_dir.mkdir(parents=True, exist_ok=True)
        self._index = None

    def _extract_sources(self, response) -> list[QuerySource]:
        sources: list[QuerySource] = []
        for source_node in getattr(response, "source_nodes", [])[: self._config.similarity_top_k]:
            node = source_node.node
            metadata = getattr(node, "metadata", {}) or {}
            file_path = str(metadata.get("file_path", ""))
            if hasattr(node, "get_content"):
                content = node.get_content(metadata_mode="none")
            else:
                content = str(getattr(node, "text", ""))
            preview = " ".join(content.strip().split())
            sources.append(
                QuerySource(
                    file_name=Path(file_path).name if file_path else "Document",
                    file_path=file_path,
                    score=getattr(source_node, "score", None),
                    preview=preview[:580],
                )
            )
        return sources

    def _format_ollama_error(
        self,
        exc: Exception,
        model_name: str,
        operation: str,
    ) -> str:
        message = str(exc).strip() or exc.__class__.__name__
        lowered = message.lower()

        if operation == "embedding":
            return f"Embedding model '{model_name}' failed: {message}"

        if "runner process has terminated" in lowered:
            return (
                f"Ollama model '{model_name}' crashed while generating a response. "
                "Try another installed chat model in the LLM field."
            )
        if "not found" in lowered or "pull model" in lowered:
            return f"Ollama model '{model_name}' is not installed. Pull it with: ollama pull {model_name}"
        if "connection refused" in lowered or "failed to establish a new connection" in lowered:
            return (
                f"Could not reach Ollama at {self._config.ollama_base_url}. "
                "Start Ollama and try again."
            )

        action = "Embedding" if operation == "embedding" else "Query"
        return f"{action} failed: {message}"
