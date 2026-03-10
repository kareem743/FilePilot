from __future__ import annotations

import app as app_module


def test_main_builds_window_and_runs_event_loop(monkeypatch):
    captured = {}

    class FakeApplication:
        def __init__(self, args):
            captured["args"] = args

        def setApplicationName(self, name):
            captured["app_name"] = name

        def exec_(self):
            captured["exec_called"] = True
            return 7

    class FakeConfigStore:
        def __init__(self, project_root):
            captured["project_root"] = project_root

        def load(self):
            captured["load_called"] = True
            return "CONFIG"

    class FakeMainWindow:
        def __init__(self, config, config_store):
            captured["window_config"] = config
            captured["window_store"] = config_store

        def show(self):
            captured["window_shown"] = True

    monkeypatch.setattr(app_module, "QApplication", FakeApplication)
    monkeypatch.setattr(app_module, "ConfigStore", FakeConfigStore)
    monkeypatch.setattr(app_module, "MainWindow", FakeMainWindow)

    result = app_module.main()

    assert result == 7
    assert captured["app_name"] == "FilePilot"
    assert captured["load_called"] is True
    assert captured["window_config"] == "CONFIG"
    assert captured["window_shown"] is True
    assert captured["exec_called"] is True
