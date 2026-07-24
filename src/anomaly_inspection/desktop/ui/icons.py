from __future__ import annotations

from pathlib import Path
from typing import Literal

try:
    import qtawesome as qta
except Exception:  # pragma: no cover - dependency/import environment fallback
    qta = None

from PySide6.QtCore import QSize
from PySide6.QtGui import QIcon, QPixmap
from PySide6.QtWidgets import QApplication

from anomaly_inspection.resources import image_asset_path
from anomaly_inspection.desktop.ui.theme import active_theme, result_colors, status_colors


IconName = Literal[
    "idle",
    "waiting",
    "ready",
    "blocked",
    "live",
    "captured",
    "processing",
    "complete",
    "error",
    "saved",
    "empty",
    "image",
    "camera",
    "zone",
    "logs",
    "run",
    "start_camera",
    "capture",
    "save",
    "browse",
    "open",
    "load",
    "filter",
    "fit",
]


ICON_NAMES: dict[str, str] = {
    "idle": "fa6s.circle",
    "waiting": "fa6s.clock",
    "ready": "fa6s.circle-check",
    "blocked": "fa6s.triangle-exclamation",
    "live": "fa6s.circle-dot",
    "captured": "fa6s.camera",
    "processing": "fa6s.spinner",
    "complete": "fa6s.circle-check",
    "error": "fa6s.circle-xmark",
    "saved": "fa6s.floppy-disk",
    "empty": "fa6s.circle-info",
    "image": "fa6s.image",
    "camera": "fa6s.video",
    "zone": "fa6s.draw-polygon",
    "logs": "fa6s.table-list",
    "run": "fa6s.play",
    "start_camera": "fa6s.video",
    "capture": "fa6s.camera",
    "save": "fa6s.floppy-disk",
    "browse": "fa6s.folder-open",
    "open": "fa6s.arrow-up-right-from-square",
    "load": "fa6s.download",
    "filter": "fa6s.filter",
    "fit": "fa6s.expand",
}

FALLBACK_GLYPHS: dict[str, str] = {
    "idle": "o",
    "waiting": "...",
    "ready": "OK",
    "blocked": "!",
    "live": "*",
    "captured": "[]",
    "processing": "...",
    "complete": "OK",
    "error": "X",
    "saved": "OK",
    "empty": "i",
    "image": "[]",
    "camera": "cam",
    "zone": "poly",
    "logs": "log",
    "run": ">",
    "start_camera": "cam",
    "capture": "cap",
    "save": "save",
    "browse": "...",
    "open": "open",
    "load": "load",
    "filter": "filter",
    "fit": "fit",
}

RESULT_ICON_NAMES: dict[str, str] = {
    "OK": "fa6s.circle-check",
    "NG": "fa6s.circle-xmark",
    "NO_PART": "fa6s.circle-minus",
    "ERROR": "fa6s.triangle-exclamation",
    "neutral": "fa6s.circle",
}

RESULT_FALLBACK_GLYPHS: dict[str, str] = {
    "OK": "OK",
    "NG": "NG",
    "NO_PART": "--",
    "ERROR": "!",
    "neutral": "o",
}


def qtawesome_available() -> bool:
    return qta is not None


def app_logo_path() -> Path:
    return image_asset_path("logo.png")


def app_window_icon() -> QIcon:
    path = app_logo_path()
    return QIcon(str(path)) if path.exists() else QIcon()


def apply_window_icon(window) -> None:
    icon = app_window_icon()
    if not icon.isNull():
        window.setWindowIcon(icon)


def apply_application_icon(app: QApplication) -> None:
    icon = app_window_icon()
    if not icon.isNull():
        app.setWindowIcon(icon)


def state_icon(icon_name: IconName, *, color: str | None = None) -> QIcon:
    return _qtawesome_icon(ICON_NAMES.get(icon_name, ICON_NAMES["empty"]), color or _state_color(icon_name))


def result_icon(result_name: str, *, color: str | None = None) -> QIcon:
    normalized = result_name if result_name in RESULT_ICON_NAMES else "neutral"
    return _qtawesome_icon(RESULT_ICON_NAMES[normalized], color or result_colors(normalized).foreground)


def icon_pixmap(icon: QIcon, size: int) -> QPixmap:
    if icon.isNull():
        return QPixmap()
    return icon.pixmap(QSize(size, size))


def state_fallback_text(icon_name: str) -> str:
    return FALLBACK_GLYPHS.get(icon_name, "")


def result_fallback_text(result_name: str) -> str:
    return RESULT_FALLBACK_GLYPHS.get(result_name, RESULT_FALLBACK_GLYPHS["neutral"])


def _qtawesome_icon(name: str, color: str) -> QIcon:
    if qta is None:
        return QIcon()
    try:
        return qta.icon(name, color=color)
    except Exception:
        return QIcon()


def _state_color(icon_name: str) -> str:
    theme = active_theme()
    if icon_name in {"ready", "captured", "complete", "saved"}:
        return theme.palette.success
    if icon_name in {"blocked", "waiting"}:
        return theme.palette.warning
    if icon_name == "error":
        return theme.palette.danger
    return status_colors("info").foreground
