import yaml
import pytest
from pathlib import Path

from inspection.config import load_config


def base_config(tmp_path, model_section):
    payload = {
        "model": model_section,
        "presence": {
            "reference_image_path": "ref.png",
            "zones_path": "zones.json",
        },
    }
    path = tmp_path / "inspection.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    return path


def test_config_accepts_new_model_path_key(tmp_path):
    path = base_config(tmp_path, {"path": "model.pt", "format": "auto", "anomaly_threshold": 0.5})

    config = load_config(path)

    assert config.project.name == "Default Job"
    assert config.model.path == tmp_path / "model.pt"
    assert config.model.format == "auto"


def test_config_accepts_openvino_model_format(tmp_path):
    path = base_config(tmp_path, {"path": "model.xml", "format": "openvino", "anomaly_threshold": 0.5, "device": "cpu"})

    config = load_config(path)

    assert config.model.path == tmp_path / "model.xml"
    assert config.model.format == "openvino"
    assert config.model.device == "cpu"


def test_config_accepts_optional_anomalib_model(tmp_path):
    path = base_config(
        tmp_path,
        {
            "path": "model.ckpt",
            "format": "ckpt",
            "anomalib_model": "reverse_distillation",
            "anomaly_threshold": 0.5,
        },
    )

    config = load_config(path)

    assert config.model.anomalib_model == "reverse_distillation"


def test_config_allows_missing_anomalib_model(tmp_path):
    path = base_config(tmp_path, {"path": "model.ckpt", "format": "ckpt", "anomaly_threshold": 0.5})

    config = load_config(path)

    assert config.model.anomalib_model is None
    assert config.model.checkpoint_inference_mode == "engine"


def test_config_accepts_explicit_engine_checkpoint_inference_mode(tmp_path):
    path = base_config(
        tmp_path,
        {
            "path": "model.ckpt",
            "format": "ckpt",
            "anomaly_threshold": 0.5,
            "checkpoint_inference_mode": "engine",
        },
    )

    config = load_config(path)

    assert config.model.checkpoint_inference_mode == "engine"


def test_config_accepts_deprecated_direct_checkpoint_inference_mode_with_warning(tmp_path):
    path = base_config(
        tmp_path,
        {
            "path": "model.ckpt",
            "format": "ckpt",
            "anomaly_threshold": 0.5,
            "checkpoint_inference_mode": "direct",
        },
    )

    with pytest.warns(DeprecationWarning, match="checkpoint_inference_mode is deprecated"):
        config = load_config(path)

    assert config.model.checkpoint_inference_mode == "direct"


def test_config_blank_checkpoint_inference_mode_defaults_to_engine(tmp_path):
    path = base_config(
        tmp_path,
        {
            "path": "model.ckpt",
            "format": "ckpt",
            "anomaly_threshold": 0.5,
            "checkpoint_inference_mode": "",
        },
    )

    config = load_config(path)

    assert config.model.checkpoint_inference_mode == "engine"


def test_config_rejects_invalid_checkpoint_inference_mode(tmp_path):
    path = base_config(
        tmp_path,
        {
            "path": "model.ckpt",
            "format": "ckpt",
            "anomaly_threshold": 0.5,
            "checkpoint_inference_mode": "magic",
        },
    )

    try:
        load_config(path)
    except ValueError as exc:
        assert "model.checkpoint_inference_mode must be one of: direct, engine" in str(exc)
    else:
        raise AssertionError("Expected invalid checkpoint inference mode to fail.")


def test_config_accepts_legacy_checkpoint_path_key(tmp_path):
    path = base_config(tmp_path, {"checkpoint_path": "model.ckpt", "anomaly_threshold": 1.0})

    config = load_config(path)

    assert config.model.path == tmp_path / "model.ckpt"
    assert config.model.checkpoint_path == tmp_path / "model.ckpt"


def test_config_parses_output_options(tmp_path):
    path = base_config(tmp_path, {"path": "model.pt"})
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    payload["output"] = {"show_images": True, "save_csv_log": False}
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    config = load_config(path)

    assert config.output.show_images is True
    assert config.output.save_csv_log is False


def test_config_accepts_project_name(tmp_path):
    path = base_config(tmp_path, {"path": "model.pt"})
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    payload["project"] = {"name": "Transistor"}
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    config = load_config(path)

    assert config.project.name == "Transistor"


def test_config_defaults_blank_project_name(tmp_path):
    path = base_config(tmp_path, {"path": "model.pt"})
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    payload["project"] = {"name": "   "}
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    config = load_config(path)

    assert config.project.name == "Default Job"


def test_sample_config_with_project_section_loads():
    config = load_config("configs/inspection.sample.yaml")
    raw = yaml.safe_load(Path("configs/inspection.sample.yaml").read_text(encoding="utf-8"))

    assert config.project.name == "default_job"
    assert "checkpoint_inference_mode" not in raw["model"]
