from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path


DEFAULT_CONFIG = {
    "source_path": "",
    "llm_model": "qwen3:8b",
    "embedding_model": "BAAI/bge-small-en-v1.5",
    "ollama_base_url": "http://127.0.0.1:11434",
    "chunk_size": 700,
    "chunk_overlap": 80,
    "similarity_top_k": 3,
    "supported_extensions": [".txt", ".md", ".pdf", ".csv", ".json", ".py"],
    "window_width": 420,
    "window_height": 520,
    "window_margin": 24,
    "always_on_top": True,
}


@dataclass
class AppConfig:
    project_root: Path
    data_dir: Path
    index_dir: Path
    config_file: Path
    source_path: str
    llm_model: str
    embedding_model: str
    ollama_base_url: str
    chunk_size: int
    chunk_overlap: int
    similarity_top_k: int
    supported_extensions: tuple[str, ...]
    window_width: int
    window_height: int
    window_margin: int
    always_on_top: bool

    @classmethod
    def from_payload(cls, project_root: Path, payload: dict) -> "AppConfig":
        data_dir = project_root / "data"
        index_dir = data_dir / "index_store"
        config_file = project_root / "config.json"

        data_dir.mkdir(parents=True, exist_ok=True)
        index_dir.mkdir(parents=True, exist_ok=True)

        merged = dict(DEFAULT_CONFIG)
        merged.update(payload)

        return cls(
            project_root=project_root,
            data_dir=data_dir,
            index_dir=index_dir,
            config_file=config_file,
            source_path=str(merged["source_path"]),
            llm_model=str(merged["llm_model"]),
            embedding_model=str(merged["embedding_model"]),
            ollama_base_url=str(merged["ollama_base_url"]),
            chunk_size=int(merged["chunk_size"]),
            chunk_overlap=int(merged["chunk_overlap"]),
            similarity_top_k=int(merged["similarity_top_k"]),
            supported_extensions=tuple(merged["supported_extensions"]),
            window_width=int(merged["window_width"]),
            window_height=int(merged["window_height"]),
            window_margin=int(merged["window_margin"]),
            always_on_top=bool(merged["always_on_top"]),
        )

    def to_payload(self) -> dict:
        return {
            "source_path": self.source_path,
            "llm_model": self.llm_model,
            "embedding_model": self.embedding_model,
            "ollama_base_url": self.ollama_base_url,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "similarity_top_k": self.similarity_top_k,
            "supported_extensions": list(self.supported_extensions),
            "window_width": self.window_width,
            "window_height": self.window_height,
            "window_margin": self.window_margin,
            "always_on_top": self.always_on_top,
        }


class ConfigStore:
    def __init__(self, project_root: Path) -> None:
        self._project_root = project_root
        self._config_file = project_root / "config.json"
        self._legacy_state_file = project_root / "data" / "app_state.json"

    def load(self) -> AppConfig:
        payload: dict
        if self._config_file.exists():
            payload = self._read_json(self._config_file)
        else:
            payload = self._load_legacy_state()

        config = AppConfig.from_payload(self._project_root, payload)
        self.save(config)
        return config

    def save(self, config: AppConfig) -> None:
        config.config_file.write_text(
            json.dumps(config.to_payload(), indent=2),
            encoding="utf-8",
        )

    def _load_legacy_state(self) -> dict:
        payload = dict(DEFAULT_CONFIG)
        payload.update(self._read_json(self._legacy_state_file))
        return payload

    @staticmethod
    def _read_json(path: Path) -> dict:
        if not path.exists():
            return {}
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
