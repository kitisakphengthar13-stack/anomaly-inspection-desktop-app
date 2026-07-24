"""Paths to assets that are shipped with the application package."""

from __future__ import annotations

from pathlib import Path


_RESOURCE_ROOT = Path(__file__).resolve().parent / "resources"


def ui_asset_path(filename: str) -> Path:
    return _RESOURCE_ROOT / "ui" / filename


def image_asset_path(filename: str) -> Path:
    return _RESOURCE_ROOT / "images" / filename
