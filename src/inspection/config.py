from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import yaml


DEFAULT_PROJECT_NAME = "Default Job"
DEFAULT_PROJECT_SLUG = "default_job"
WINDOWS_RESERVED_PROJECT_SLUGS = {
    "con",
    "prn",
    "aux",
    "nul",
    *(f"com{index}" for index in range(1, 10)),
    *(f"lpt{index}" for index in range(1, 10)),
}


@dataclass(frozen=True)
class ModelConfig:
    path: Path
    anomaly_threshold: float
    device: str = "auto"
    format: str = "auto"

    @property
    def checkpoint_path(self) -> Path:
        return self.path


@dataclass(frozen=True)
class PresenceConfig:
    reference_image_path: Path
    zones_path: Path
    pixel_diff_threshold: int = 30
    min_foreground_ratio: float = 0.08
    min_blob_area: int = 500
    blur_kernel_size: int = 5
    morphology_kernel_size: int = 5
    use_largest_blob_filter: bool = True


@dataclass(frozen=True)
class OutputConfig:
    save_annotated: bool = True
    save_heatmap: bool = True
    save_presence_mask: bool = True
    organize_by_result: bool = True
    show_images: bool = False
    save_csv_log: bool = True


@dataclass(frozen=True)
class ProjectConfig:
    name: str = DEFAULT_PROJECT_NAME

    @property
    def slug(self) -> str:
        return make_project_slug(self.name)


@dataclass(frozen=True)
class InspectionConfig:
    project: ProjectConfig
    model: ModelConfig
    presence: PresenceConfig
    output: OutputConfig
    config_path: Path


def _resolve_path(raw: str, base_dir: Path) -> Path:
    path = Path(raw)
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def _require_section(data: Dict[str, Any], section: str) -> Dict[str, Any]:
    value = data.get(section)
    if not isinstance(value, dict):
        raise ValueError(f"Missing or invalid '{section}' section in config.")
    return value


def _validate_odd_kernel(name: str, value: int) -> int:
    if value < 0:
        raise ValueError(f"{name} must be >= 0.")
    if value not in (0, 1) and value % 2 == 0:
        raise ValueError(f"{name} must be 0, 1, or an odd integer.")
    return value


def load_config(config_path: str | Path) -> InspectionConfig:
    config_path = Path(config_path).resolve()
    with config_path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file) or {}

    if not isinstance(data, dict):
        raise ValueError("Config file must contain a YAML mapping.")

    base_dir = config_path.parent.parent if config_path.parent.name == "configs" else config_path.parent
    project_data = data.get("project", {}) or {}
    if not isinstance(project_data, dict):
        raise ValueError("Missing or invalid 'project' section in config.")
    model_data = _require_section(data, "model")
    presence_data = _require_section(data, "presence")
    output_data = data.get("output", {}) or {}

    threshold = float(model_data.get("anomaly_threshold", 0.5))
    if threshold < 0:
        raise ValueError("model.anomaly_threshold must be >= 0.")

    pixel_diff_threshold = int(presence_data.get("pixel_diff_threshold", 30))
    if not 0 <= pixel_diff_threshold <= 255:
        raise ValueError("presence.pixel_diff_threshold must be between 0 and 255.")

    min_foreground_ratio = float(presence_data.get("min_foreground_ratio", 0.08))
    if not 0 <= min_foreground_ratio <= 1:
        raise ValueError("presence.min_foreground_ratio must be between 0 and 1.")

    min_blob_area = int(presence_data.get("min_blob_area", 500))
    if min_blob_area < 0:
        raise ValueError("presence.min_blob_area must be >= 0.")

    model_path_value = model_data.get("path", model_data.get("checkpoint_path"))
    if not model_path_value:
        raise ValueError("model.path is required. Legacy model.checkpoint_path is also accepted.")
    model_format = str(model_data.get("format", "auto")).lower()
    if model_format not in {"auto", "ckpt", "torch_export", "pt", "torch"}:
        raise ValueError("model.format must be one of: auto, ckpt, torch_export.")

    model = ModelConfig(
        path=_resolve_path(str(model_path_value), base_dir),
        anomaly_threshold=threshold,
        device=str(model_data.get("device", "auto")),
        format="torch_export" if model_format in {"pt", "torch"} else model_format,
    )
    presence = PresenceConfig(
        reference_image_path=_resolve_path(str(presence_data["reference_image_path"]), base_dir),
        zones_path=_resolve_path(str(presence_data["zones_path"]), base_dir),
        pixel_diff_threshold=pixel_diff_threshold,
        min_foreground_ratio=min_foreground_ratio,
        min_blob_area=min_blob_area,
        blur_kernel_size=_validate_odd_kernel("presence.blur_kernel_size", int(presence_data.get("blur_kernel_size", 5))),
        morphology_kernel_size=_validate_odd_kernel("presence.morphology_kernel_size", int(presence_data.get("morphology_kernel_size", 5))),
        use_largest_blob_filter=bool(presence_data.get("use_largest_blob_filter", True)),
    )
    output = OutputConfig(
        save_annotated=bool(output_data.get("save_annotated", True)),
        save_heatmap=bool(output_data.get("save_heatmap", True)),
        save_presence_mask=bool(output_data.get("save_presence_mask", True)),
        organize_by_result=bool(output_data.get("organize_by_result", True)),
        show_images=bool(output_data.get("show_images", False)),
        save_csv_log=bool(output_data.get("save_csv_log", True)),
    )
    project_name = str(project_data.get("name", "")).strip() or DEFAULT_PROJECT_NAME
    project = ProjectConfig(name=project_name)
    return InspectionConfig(project=project, model=model, presence=presence, output=output, config_path=config_path)


def normalize_project_name(name: str | None) -> str:
    value = (name or "").strip()
    return value or DEFAULT_PROJECT_NAME


def make_project_slug(name: str | None) -> str:
    import re

    normalized = normalize_project_name(name).lower()
    slug = re.sub(r"[^a-z0-9]+", "_", normalized)
    slug = re.sub(r"_+", "_", slug).strip("_") or DEFAULT_PROJECT_SLUG
    if slug in WINDOWS_RESERVED_PROJECT_SLUGS:
        return f"job_{slug}"
    return slug
