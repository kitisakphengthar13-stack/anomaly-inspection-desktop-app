from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from inspection_app.job_paths import DEFAULT_INSPECTION_JOB_NAME, make_job_slug
from inspection_app.theme import DEFAULT_THEME_NAME


@dataclass
class AppState:
    """Shared desktop-app state for future GUI pages."""

    config_path: Path | None = None
    output_dir: Path | None = None
    image_output_dir: Path | None = None
    camera_output_dir: Path | None = None
    logs_root_dir: Path | None = None
    reference_image_path: Path | None = None
    zones_path: Path | None = None
    reference_path_auto: bool = True
    zones_path_auto: bool = True
    inspection_job_name: str = DEFAULT_INSPECTION_JOB_NAME
    inspection_job_slug: str = "default_job"
    status_message: str = "Ready"
    config_loaded: bool = False
    last_validation_message: str | None = None
    desktop_theme_name: str = DEFAULT_THEME_NAME

    def __post_init__(self) -> None:
        self.inspection_job_name = self.inspection_job_name.strip() or DEFAULT_INSPECTION_JOB_NAME
        self.inspection_job_slug = make_job_slug(self.inspection_job_name)
        if self.reference_image_path is not None:
            self.reference_path_auto = False
        if self.zones_path is not None:
            self.zones_path_auto = False

    def set_inspection_job_name(self, name: str) -> None:
        self.inspection_job_name = name.strip() or DEFAULT_INSPECTION_JOB_NAME
        self.inspection_job_slug = make_job_slug(self.inspection_job_name)
