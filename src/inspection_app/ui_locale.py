from __future__ import annotations

from PySide6.QtCore import QLocale
from PySide6.QtWidgets import QApplication, QWidget

APP_LOCALE = QLocale(QLocale.Language.English, QLocale.Country.UnitedStates)


def configure_app_locale(app: QApplication | None = None) -> None:
    QLocale.setDefault(APP_LOCALE)


def apply_app_locale(widget: QWidget) -> None:
    widget.setLocale(APP_LOCALE)
    for child in widget.findChildren(QWidget):
        child.setLocale(APP_LOCALE)
