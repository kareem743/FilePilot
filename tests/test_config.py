from __future__ import annotations

import json

from filepilot.config import AppConfig, ConfigStore, DEFAULT_CONFIG


def test_app_config_from_payload_applies_defaults(tmp_path):
    config = AppConfig.from_payload(tmp_path, {"llm_model": "custom-llm"})

    assert config.project_root == tmp_path
    assert config.config_file == tmp_path / "config.json"
    assert config.data_dir.exists()
    assert config.index_dir.exists()
    assert config.llm_model == "custom-llm"
    assert config.embedding_model == DEFAULT_CONFIG["embedding_model"]
    assert config.supported_extensions == tuple(DEFAULT_CONFIG["supported_extensions"])


def test_config_store_load_prefers_config_file(tmp_path):
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "llm_model": "qwen3:latest",
                "embedding_model": "nomic-embed-text",
                "source_path": "from-config.txt",
            }
        ),
        encoding="utf-8",
    )
    legacy_dir = tmp_path / "data"
    legacy_dir.mkdir()
    (legacy_dir / "app_state.json").write_text(
        json.dumps({"llm_model": "legacy-model"}),
        encoding="utf-8",
    )

    config = ConfigStore(tmp_path).load()

    assert config.llm_model == "qwen3:latest"
    assert config.source_path == "from-config.txt"


def test_config_store_load_migrates_legacy_state_when_config_is_missing(tmp_path):
    legacy_dir = tmp_path / "data"
    legacy_dir.mkdir()
    (legacy_dir / "app_state.json").write_text(
        json.dumps(
            {
                "source_path": "legacy.pdf",
                "llm_model": "legacy-llm",
                "embedding_model": "legacy-embed",
            }
        ),
        encoding="utf-8",
    )

    store = ConfigStore(tmp_path)
    config = store.load()

    assert config.source_path == "legacy.pdf"
    assert config.llm_model == "legacy-llm"
    assert config.embedding_model == "legacy-embed"

    saved_payload = json.loads((tmp_path / "config.json").read_text(encoding="utf-8"))
    assert saved_payload["llm_model"] == "legacy-llm"


def test_config_store_invalid_json_falls_back_to_defaults(tmp_path):
    (tmp_path / "config.json").write_text("{invalid", encoding="utf-8")

    config = ConfigStore(tmp_path).load()

    assert config.llm_model == DEFAULT_CONFIG["llm_model"]
    assert config.embedding_model == DEFAULT_CONFIG["embedding_model"]


def test_config_store_save_writes_current_config(tmp_path):
    store = ConfigStore(tmp_path)
    config = AppConfig.from_payload(tmp_path, {"llm_model": "qwen3:14b"})

    store.save(config)

    saved_payload = json.loads((tmp_path / "config.json").read_text(encoding="utf-8"))
    assert saved_payload["llm_model"] == "qwen3:14b"
