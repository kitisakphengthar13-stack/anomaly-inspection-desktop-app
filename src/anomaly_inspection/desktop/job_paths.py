from __future__ import annotations

from pathlib import Path

from anomaly_inspection.core.config import DEFAULT_PROJECT_NAME, DEFAULT_PROJECT_SLUG, make_project_slug, normalize_project_name

DEFAULT_INSPECTION_JOB_NAME = DEFAULT_PROJECT_NAME
DEFAULT_JOB_SLUG = DEFAULT_PROJECT_SLUG
LEGACY_REFERENCE_IMAGE_PATH = Path("data/reference/empty_reference.png")
LEGACY_ZONES_PATH = Path("configs/zones.json")


def normalize_inspection_job_name(name: str | None) -> str:
    return normalize_project_name(name)


def make_job_slug(name: str | None) -> str:
    return make_project_slug(name)


def default_camera_output_dir(job_slug: str) -> Path:
    return Path("outputs") / (job_slug or DEFAULT_JOB_SLUG) / "camera"


def default_image_output_dir(job_slug: str) -> Path:
    return Path("outputs") / (job_slug or DEFAULT_JOB_SLUG) / "image"


def default_logs_root_dir(job_slug: str) -> Path:
    return Path("outputs") / (job_slug or DEFAULT_JOB_SLUG)


def default_job_asset_root(job_slug: str) -> Path:
    return Path("data") / "jobs" / (job_slug or DEFAULT_JOB_SLUG)


def default_reference_image_path(job_slug: str) -> Path:
    return default_job_asset_root(job_slug) / "reference" / "empty_reference.png"


def default_zones_path(job_slug: str) -> Path:
    return default_job_asset_root(job_slug) / "zones" / "zones.json"
