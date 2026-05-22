from __future__ import annotations

import sys
from collections.abc import Sequence

from PySide6.QtWidgets import QApplication

from inspection_app.icons import apply_application_icon, apply_window_icon
from inspection_app.main_window import MainWindow
from inspection_app.state import AppState
from inspection_app.theme import apply_app_theme
from inspection_app.ui_locale import configure_app_locale


def main(argv: Sequence[str] | None = None) -> int:
    configure_app_locale()
    app = QApplication(list(argv) if argv is not None else sys.argv)
    configure_app_locale(app)
    apply_app_theme(app)
    app.setApplicationName("Anomaly Inspection")
    apply_application_icon(app)

    state = AppState(status_message="Desktop app shell initialized.")
    window = MainWindow(state)
    apply_window_icon(window)
    window.show()

    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
