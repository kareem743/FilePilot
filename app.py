from pathlib import Path
import sys
import torch
from PyQt6.QtWidgets import QApplication

from filepilot.config import ConfigStore
from filepilot.ui.main_window import MainWindow


def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName("FilePilot")

    config_store = ConfigStore(Path(__file__).resolve().parent)
    config = config_store.load()
    window = MainWindow(config=config, config_store=config_store)
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
