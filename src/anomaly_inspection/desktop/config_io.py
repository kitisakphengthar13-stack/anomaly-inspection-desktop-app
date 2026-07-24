from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import yaml

from anomaly_inspection.core.config import InspectionConfig, load_config


def default_config_data(repo_root: Path | None = None) -> dict[str, Any]:
    root = repo_root or Path.cwd()
    sample_path = root / "configs" / "inspection.sample.yaml"
    if sample_path.exists():
        with sample_path.open("r", encoding="utf-8") as file:
            data = yaml.safe_load(file) or {}
        if isinstance(data, dict):
            return data

    return {
        "project": {
            "name": "default_job",
        },
        "model": {
            "path": "path/to/anomaly_model.pt",
            "format": "auto",
            "anomalib_model": "",
            "anomaly_threshold": 0.5,
            "device": "auto",
        },
        "presence": {
            "reference_image_path": "data/reference/empty_reference.png",
            "zones_path": "configs/zones.json",
            "pixel_diff_threshold": 30,
            "min_foreground_ratio": 0.08,
            "min_blob_area": 500,
            "blur_kernel_size": 5,
            "morphology_kernel_size": 5,
            "use_largest_blob_filter": True,
        },
        "output": {
            "save_annotated": True,
            "save_heatmap": True,
            "save_presence_mask": True,
            "organize_by_result": True,
            "show_images": True,
            "save_csv_log": True,
        },
    }


def config_to_yaml_data(config: InspectionConfig) -> dict[str, Any]:
    model_data = {
        "path": str(config.model.path),
        "format": config.model.format,
        "anomaly_threshold": config.model.anomaly_threshold,
        "device": config.model.device,
    }
    if config.model.anomalib_model:
        model_data["anomalib_model"] = config.model.anomalib_model
    return {
        "project": {
            "name": config.project.name,
        },
        "model": model_data,
        "presence": {
            "reference_image_path": str(config.presence.reference_image_path),
            "zones_path": str(config.presence.zones_path),
            "pixel_diff_threshold": config.presence.pixel_diff_threshold,
            "min_foreground_ratio": config.presence.min_foreground_ratio,
            "min_blob_area": config.presence.min_blob_area,
            "blur_kernel_size": config.presence.blur_kernel_size,
            "morphology_kernel_size": config.presence.morphology_kernel_size,
            "use_largest_blob_filter": config.presence.use_largest_blob_filter,
        },
        "output": {
            "save_annotated": config.output.save_annotated,
            "save_heatmap": config.output.save_heatmap,
            "save_presence_mask": config.output.save_presence_mask,
            "organize_by_result": config.output.organize_by_result,
            "show_images": config.output.show_images,
            "save_csv_log": config.output.save_csv_log,
        },
    }


def save_config_data(path: str | Path, data: dict[str, Any]) -> None:
    config_path = Path(path)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with config_path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(data, file, sort_keys=False)


def validate_config_data(
    data: dict[str, Any],
    base_path: str | Path | None = None,
    *,
    require_existing_files: bool = False,
) -> InspectionConfig:
    _validate_required_path_fields(data)
    base_dir = Path(base_path).resolve().parent if base_path else Path.cwd()
    base_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", dir=base_dir, encoding="utf-8", delete=False) as file:
        temp_path = Path(file.name)
        yaml.safe_dump(data, file, sort_keys=False)

    try:
        config = load_config(temp_path)
    finally:
        temp_path.unlink(missing_ok=True)
    if require_existing_files:
        validate_required_files_exist(config)
    return config


def validate_required_files_exist(config: InspectionConfig) -> None:
    missing = [
        ("Model file", config.model.path),
        ("Reference image", config.presence.reference_image_path),
        ("Zones JSON", config.presence.zones_path),
    ]
    messages = [f"{label} does not exist: {path}" for label, path in missing if not path.exists()]
    if messages:
        raise ValueError("\n".join(messages))


def _validate_required_path_fields(data: dict[str, Any]) -> None:
    model = data.get("model", {}) if isinstance(data, dict) else {}
    presence = data.get("presence", {}) if isinstance(data, dict) else {}
    required = (
        ("model.path", model.get("path", model.get("checkpoint_path"))),
        ("presence.reference_image_path", presence.get("reference_image_path")),
        ("presence.zones_path", presence.get("zones_path")),
    )
    for name, value in required:
        if not str(value or "").strip():
            raise ValueError(f"{name} is required.")
