import csv
import importlib.util
import json
from pathlib import Path

import cv2
import numpy as np


SCRIPT_PATH = Path(__file__).resolve().parents[2] / "tools" / "validate_anomalib_artifact.py"
SPEC = importlib.util.spec_from_file_location("validate_anomalib_artifact", SCRIPT_PATH)
validate_tool = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(validate_tool)


class FakePrediction:
    pred_score = np.array([0.72], dtype=np.float32)
    pred_label = np.array([True])
    anomaly_map = np.ones((1, 16, 16), dtype=np.float32)
    pred_mask = np.ones((1, 16, 16), dtype=bool)


class FakeBackend:
    def __init__(self, backend_name):
        self.backend_name = backend_name

    def predict(self, image):
        assert image.shape == (8, 8, 3)
        return FakePrediction()


class FakeAnomalyInferencer:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        model_format = kwargs["model_format"]
        if model_format == "ckpt":
            self.backend_name = "engine_checkpoint"
        elif model_format == "torch_export":
            self.backend_name = "exported_torch"
        elif model_format == "openvino":
            self.backend_name = "openvino"
        else:
            self.backend_name = "unknown"
        self.backend = FakeBackend(self.backend_name)

    def load(self):
        pass


def write_image(path):
    cv2.imwrite(str(path), np.zeros((8, 8, 3), dtype=np.uint8))


def validate_with_fake_inferencer(tmp_path, monkeypatch, model_format):
    artifact = tmp_path / f"model.{ {'ckpt': 'ckpt', 'torch_export': 'pt', 'openvino': 'xml'}[model_format] }"
    image = tmp_path / "image.png"
    artifact.write_bytes(b"artifact")
    write_image(image)
    monkeypatch.setattr(validate_tool, "AnomalyInferencer", FakeAnomalyInferencer)

    return validate_tool.validate_artifact(
        model_name="patchcore",
        artifact_path=artifact,
        model_format=model_format,
        image_path=image,
        device="cpu",
    )


def test_missing_artifact_path_reports_no_local_artifact(tmp_path):
    image = tmp_path / "image.png"
    write_image(image)

    report = validate_tool.validate_artifact(
        model_name="patchcore",
        artifact_path=tmp_path / "missing.ckpt",
        model_format="ckpt",
        image_path=image,
        device="cpu",
    )

    assert report["status"] == "no_local_artifact"
    assert report["load_success"] is False
    assert "does not exist" in report["errors"]


def test_missing_image_path_reports_verified_fail(tmp_path):
    artifact = tmp_path / "model.ckpt"
    artifact.write_bytes(b"artifact")

    report = validate_tool.validate_artifact(
        model_name="patchcore",
        artifact_path=artifact,
        model_format="ckpt",
        image_path=tmp_path / "missing.png",
        device="cpu",
    )

    assert report["status"] == "verified_fail"
    assert report["predict_success"] is False
    assert "Image path does not exist" in report["errors"]


def test_ckpt_selects_engine_checkpoint_backend(tmp_path, monkeypatch):
    report = validate_with_fake_inferencer(tmp_path, monkeypatch, "ckpt")

    assert report["status"] == "verified_pass"
    assert report["selected_backend"] == "engine_checkpoint"
    assert report["score"] == np.float32(0.72)
    assert report["label"] is True
    assert report["heatmap_shape"] == "1x16x16"
    assert report["pred_mask_shape"] == "1x16x16"


def test_pt_selects_exported_torch_backend(tmp_path, monkeypatch):
    report = validate_with_fake_inferencer(tmp_path, monkeypatch, "torch_export")

    assert report["status"] == "verified_pass"
    assert report["selected_backend"] == "exported_torch"


def test_openvino_selects_current_openvino_backend(tmp_path, monkeypatch):
    report = validate_with_fake_inferencer(tmp_path, monkeypatch, "openvino")

    assert report["status"] == "verified_pass"
    assert report["selected_backend"] == "openvino"


def test_json_output_is_written_with_expected_fields(tmp_path, monkeypatch):
    report = validate_with_fake_inferencer(tmp_path, monkeypatch, "ckpt")
    output_path = tmp_path / "report.json"

    validate_tool.write_json(report, output_path)

    raw = json.loads(output_path.read_text(encoding="utf-8"))
    assert raw["model_name"] == "patchcore"
    assert raw["selected_backend"] == "engine_checkpoint"
    assert raw["status"] == "verified_pass"


def test_csv_output_is_written_with_expected_fields(tmp_path, monkeypatch):
    report = validate_with_fake_inferencer(tmp_path, monkeypatch, "openvino")
    output_path = tmp_path / "report.csv"

    validate_tool.write_csv(report, output_path)

    with output_path.open("r", newline="", encoding="utf-8") as file:
        row = next(csv.DictReader(file))
    assert row["model_name"] == "patchcore"
    assert row["selected_backend"] == "openvino"
    assert row["status"] == "verified_pass"
    assert "probe_output_object_type" in row


def test_video_model_name_is_out_of_scope(tmp_path):
    artifact = tmp_path / "model.ckpt"
    image = tmp_path / "image.png"
    artifact.write_bytes(b"artifact")
    write_image(image)

    report = validate_tool.validate_artifact(
        model_name="ai_vad",
        artifact_path=artifact,
        model_format="ckpt",
        image_path=image,
        device="cpu",
    )

    assert report["status"] == "unsupported_not_image_runtime"
    assert "Video models are out of scope" in report["errors"]
