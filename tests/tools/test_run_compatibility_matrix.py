import csv
import importlib.util
import json
from pathlib import Path

import pytest
import yaml


SCRIPT_PATH = Path(__file__).resolve().parents[2] / "tools" / "run_compatibility_matrix.py"
SPEC = importlib.util.spec_from_file_location("run_compatibility_matrix", SCRIPT_PATH)
matrix_tool = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(matrix_tool)


def manifest_rows(*rows):
    return {"artifacts": list(rows)}


def row(
    row_id="patchcore_ckpt",
    model_name="patchcore",
    model_format="ckpt",
    artifact_path="model.ckpt",
    image_path="image.png",
    **extra,
):
    data = {
        "id": row_id,
        "model_name": model_name,
        "format": model_format,
        "artifact_path": artifact_path,
        "image_path": image_path,
        "device": "cpu",
        "anomaly_threshold": 0.5,
    }
    data.update(extra)
    return data


def fake_report(**overrides):
    report = {
        "model_name": "patchcore",
        "format": "ckpt",
        "artifact_path": "model.ckpt",
        "image_path": "image.png",
        "selected_backend": "engine_checkpoint",
        "load_success": True,
        "predict_success": True,
        "output_object_type": "ImageBatch",
        "output_keys_or_attributes": "pred_score,pred_label,anomaly_map,pred_mask",
        "score": 0.72,
        "label": True,
        "heatmap_shape": "1x256x256",
        "pred_mask_shape": "1x256x256",
        "warnings": "",
        "errors": "",
        "load_ms": 10.0,
        "predict_ms": 2.0,
        "status": "verified_pass",
    }
    report.update(overrides)
    return report


def test_loads_valid_yaml_manifest(tmp_path):
    manifest_path = tmp_path / "matrix.yaml"
    manifest_path.write_text(yaml.safe_dump(manifest_rows(row())), encoding="utf-8")

    manifest = matrix_tool.load_manifest(manifest_path)

    assert manifest["artifacts"][0]["id"] == "patchcore_ckpt"


def test_loads_valid_json_manifest(tmp_path):
    manifest_path = tmp_path / "matrix.json"
    manifest_path.write_text(json.dumps(manifest_rows(row())), encoding="utf-8")

    manifest = matrix_tool.load_manifest(manifest_path)

    assert manifest["artifacts"][0]["format"] == "ckpt"


def test_missing_manifest_file_fails_clearly(tmp_path):
    with pytest.raises(FileNotFoundError, match="Manifest file does not exist"):
        matrix_tool.load_manifest(tmp_path / "missing.yaml")


def test_malformed_manifest_fails_clearly(tmp_path):
    manifest_path = tmp_path / "bad.yaml"
    manifest_path.write_text("artifacts: [", encoding="utf-8")

    with pytest.raises(ValueError, match="Could not parse manifest"):
        matrix_tool.load_manifest(manifest_path)


def test_multiple_rows_produce_multiple_output_rows(monkeypatch):
    calls = []

    def fake_validate_artifact(**kwargs):
        calls.append(kwargs)
        return fake_report(
            model_name=kwargs["model_name"],
            format=kwargs["model_format"],
            artifact_path=str(kwargs["artifact_path"]),
            image_path=str(kwargs["image_path"]),
            selected_backend="openvino" if kwargs["model_format"] == "openvino" else "engine_checkpoint",
        )

    monkeypatch.setattr(matrix_tool, "validate_artifact", fake_validate_artifact)
    manifest = manifest_rows(
        row("patchcore_ckpt", "patchcore", "ckpt"),
        row("patchcore_openvino", "patchcore", "openvino", artifact_path="model.xml"),
    )

    results = matrix_tool.run_matrix(manifest)

    assert len(results) == 2
    assert results[0]["backend"] == "engine_checkpoint"
    assert results[1]["backend"] == "openvino"
    assert len(calls) == 2


def test_missing_artifact_path_produces_no_local_artifact():
    results = matrix_tool.run_matrix(manifest_rows(row(artifact_path="")))

    assert results[0]["status"] == "no_local_artifact"
    assert "Artifact path is missing" in results[0]["errors"]


def test_missing_image_path_produces_verified_fail():
    results = matrix_tool.run_matrix(manifest_rows(row(image_path="")))

    assert results[0]["status"] == "verified_fail"
    assert "Image path is missing" in results[0]["errors"]


def test_video_model_produces_unsupported_not_image_runtime(monkeypatch):
    def fail_if_called(**kwargs):
        raise AssertionError("Video model should not be validated.")

    monkeypatch.setattr(matrix_tool, "validate_artifact", fail_if_called)

    results = matrix_tool.run_matrix(manifest_rows(row(model_name="ai_vad")))

    assert results[0]["status"] == "unsupported_not_image_runtime"
    assert "Video models are out of scope" in results[0]["errors"]


def test_continue_on_error_continues_after_validator_failure(monkeypatch):
    calls = []

    def fake_validate_artifact(**kwargs):
        calls.append(kwargs["model_name"])
        if kwargs["model_name"] == "broken":
            raise RuntimeError("boom")
        return fake_report(model_name=kwargs["model_name"])

    monkeypatch.setattr(matrix_tool, "validate_artifact", fake_validate_artifact)
    manifest = manifest_rows(row("bad", "broken"), row("good", "patchcore"))

    results = matrix_tool.run_matrix(manifest, continue_on_error=True)

    assert [result["status"] for result in results] == ["verified_fail", "verified_pass"]
    assert "boom" in results[0]["errors"]
    assert calls == ["broken", "patchcore"]


def test_without_continue_on_error_raises_validator_failure(monkeypatch):
    monkeypatch.setattr(matrix_tool, "validate_artifact", lambda **kwargs: (_ for _ in ()).throw(RuntimeError("boom")))

    with pytest.raises(RuntimeError, match="boom"):
        matrix_tool.run_matrix(manifest_rows(row()))


def test_rows_are_passed_to_single_artifact_validator(monkeypatch):
    calls = []

    def fake_validate_artifact(**kwargs):
        calls.append(kwargs)
        return fake_report(
            model_name=kwargs["model_name"],
            format=kwargs["model_format"],
            selected_backend={
                "ckpt": "engine_checkpoint",
                "torch_export": "exported_torch",
                "openvino": "openvino",
            }[kwargs["model_format"]],
        )

    monkeypatch.setattr(matrix_tool, "validate_artifact", fake_validate_artifact)
    manifest = manifest_rows(
        row("ckpt", "patchcore", "ckpt"),
        row("pt", "reverse_distillation", "torch_export", artifact_path="model.pt"),
        row("xml", "patchcore", "openvino", artifact_path="model.xml"),
    )

    results = matrix_tool.run_matrix(manifest, probe_openvino_inferencer=True, device_override="cpu")

    assert [call["model_format"] for call in calls] == ["ckpt", "torch_export", "openvino"]
    assert all(call["device"] == "cpu" for call in calls)
    assert calls[-1]["probe_openvino_inferencer"] is True
    assert [result["backend"] for result in results] == ["engine_checkpoint", "exported_torch", "openvino"]


def test_json_output_is_written(tmp_path):
    output_path = tmp_path / "matrix.json"
    rows = [matrix_tool.matrix_row("row1", row(), fake_report())]

    matrix_tool.write_json_report(rows, output_path)

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["summary"]["total_artifacts"] == 1
    assert payload["artifacts"][0]["id"] == "row1"


def test_csv_output_is_written(tmp_path):
    output_path = tmp_path / "matrix.csv"
    rows = [matrix_tool.matrix_row("row1", row(), fake_report())]

    matrix_tool.write_csv_report(rows, output_path)

    with output_path.open("r", newline="", encoding="utf-8") as file:
        csv_row = next(csv.DictReader(file))
    assert csv_row["id"] == "row1"
    assert csv_row["backend"] == "engine_checkpoint"
    assert csv_row["status"] == "verified_pass"


def test_stdout_summary_includes_grouped_counts(capsys):
    rows = [
        matrix_tool.matrix_row("row1", row(model_name="patchcore"), fake_report(model_name="patchcore")),
        matrix_tool.matrix_row(
            "row2",
            row(model_name="reverse_distillation", model_format="torch_export"),
            fake_report(model_name="reverse_distillation", format="torch_export", selected_backend="exported_torch"),
        ),
    ]

    matrix_tool.print_summary(rows)

    output = capsys.readouterr().out
    assert "total artifacts: 2" in output
    assert "verified_pass: 2" in output
    assert "patchcore: 1" in output
    assert "torch_export: 1" in output


def test_image_special_models_are_not_classified_as_video(monkeypatch):
    calls = []

    def fake_validate_artifact(**kwargs):
        calls.append(kwargs["model_name"])
        return fake_report(model_name=kwargs["model_name"], status="verified_fail")

    monkeypatch.setattr(matrix_tool, "validate_artifact", fake_validate_artifact)
    special_models = ["vlm_ad", "win_clip", "anomaly_v_f_m", "c_f_m", "draem", "efficient_ad", "glass"]
    manifest = manifest_rows(*(row(model_name=model_name) for model_name in special_models))

    results = matrix_tool.run_matrix(manifest)

    assert calls == special_models
    assert all(result["status"] == "verified_fail" for result in results)


def test_main_writes_outputs_and_returns_success(tmp_path, monkeypatch, capsys):
    manifest_path = tmp_path / "matrix.yaml"
    json_path = tmp_path / "matrix.json"
    csv_path = tmp_path / "matrix.csv"
    manifest_path.write_text(yaml.safe_dump(manifest_rows(row())), encoding="utf-8")
    monkeypatch.setattr(matrix_tool, "validate_artifact", lambda **kwargs: fake_report())

    exit_code = matrix_tool.main(
        [
            "--manifest",
            str(manifest_path),
            "--output-json",
            str(json_path),
            "--output-csv",
            str(csv_path),
        ],
    )

    assert exit_code == 0
    assert json_path.exists()
    assert csv_path.exists()
    assert "Compatibility Matrix Summary" in capsys.readouterr().out


def test_main_fail_on_verified_fail_returns_nonzero(tmp_path, monkeypatch):
    manifest_path = tmp_path / "matrix.yaml"
    manifest_path.write_text(yaml.safe_dump(manifest_rows(row())), encoding="utf-8")
    monkeypatch.setattr(matrix_tool, "validate_artifact", lambda **kwargs: fake_report(status="verified_fail"))

    exit_code = matrix_tool.main(["--manifest", str(manifest_path), "--fail-on-verified-fail"])

    assert exit_code == 1
