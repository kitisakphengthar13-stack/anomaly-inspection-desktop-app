from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import warnings
from pathlib import Path
from typing import Any

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from inspection.anomaly_inferencer import (  # noqa: E402
    AnomalyInferencer,
    extract_heatmap,
    extract_pred_label,
    extract_score,
)


VIDEO_MODEL_NAMES = {"ai_vad", "fuvas"}
KNOWN_OUTPUT_FIELDS = (
    "pred_score",
    "pred_label",
    "anomaly_map",
    "pred_mask",
    "heat_map",
    "heatmap",
    "prediction",
    "image_path",
)
CSV_FIELDS = (
    "model_name",
    "format",
    "artifact_path",
    "image_path",
    "selected_backend",
    "load_success",
    "predict_success",
    "output_object_type",
    "output_keys_or_attributes",
    "score",
    "label",
    "heatmap_shape",
    "pred_mask_shape",
    "warnings",
    "errors",
    "load_ms",
    "predict_ms",
    "status",
    "probe_output_object_type",
    "probe_output_keys_or_attributes",
    "probe_score",
    "probe_label",
    "probe_heatmap_shape",
    "probe_pred_mask_shape",
    "probe_warnings",
    "probe_errors",
    "probe_load_ms",
    "probe_predict_ms",
)


def validate_artifact(
    *,
    model_name: str,
    artifact_path: str | Path,
    model_format: str,
    image_path: str | Path,
    device: str = "auto",
    anomaly_threshold: float = 0.5,
    probe_openvino_inferencer: bool = False,
) -> dict[str, Any]:
    artifact = Path(artifact_path)
    image = Path(image_path)
    report = base_report(model_name, model_format, artifact, image)

    if canonical_model_name(model_name) in VIDEO_MODEL_NAMES:
        report["status"] = "unsupported_not_image_runtime"
        report["errors"] = "Video models are out of scope for this single-image validation tool."
        return report

    if not artifact.exists():
        report["status"] = "no_local_artifact"
        report["errors"] = f"Artifact path does not exist: {artifact}"
        return report

    if not image.exists():
        report["status"] = "verified_fail"
        report["errors"] = f"Image path does not exist: {image}"
        return report

    image_bgr = cv2.imread(str(image), cv2.IMREAD_COLOR)
    if image_bgr is None:
        report["status"] = "verified_fail"
        report["errors"] = f"Image path could not be read by OpenCV: {image}"
        return report

    captured_warnings: list[str] = []
    try:
        with warnings.catch_warnings(record=True) as warning_records:
            warnings.simplefilter("always")
            inferencer = AnomalyInferencer(
                model_path=artifact,
                anomaly_threshold=anomaly_threshold,
                device=device,
                model_format=model_format,
                anomalib_model=model_name,
            )

            load_start = time.perf_counter()
            inferencer.load()
            report["load_ms"] = elapsed_ms(load_start)
            report["load_success"] = True
            report["selected_backend"] = inferencer.backend_name

            predict_start = time.perf_counter()
            prediction = inferencer.backend.predict(image_bgr) if inferencer.backend else None
            report["predict_ms"] = elapsed_ms(predict_start)
            report["predict_success"] = True
            apply_prediction_details(report, prediction)
            captured_warnings.extend(format_warnings(warning_records))
    except Exception as exc:
        report["errors"] = str(exc)
        report["status"] = "verified_fail"
        report["warnings"] = join_messages(captured_warnings)
        return report

    report["warnings"] = join_messages(captured_warnings)
    if report["score"] == "":
        report["status"] = "verified_fail"
        report["errors"] = "Prediction did not expose a usable anomaly score."
    else:
        report["status"] = "verified_pass"

    if probe_openvino_inferencer and normalize_format(model_format) == "openvino":
        report["openvino_inferencer_probe"] = probe_anomalib_openvino(
            artifact_path=artifact,
            image_path=image,
            device=device,
        )
        flatten_probe(report, report["openvino_inferencer_probe"])

    return report


def base_report(model_name: str, model_format: str, artifact_path: Path, image_path: Path) -> dict[str, Any]:
    return {
        "model_name": model_name,
        "format": model_format,
        "artifact_path": str(artifact_path),
        "image_path": str(image_path),
        "selected_backend": "",
        "load_success": False,
        "predict_success": False,
        "output_object_type": "",
        "output_keys_or_attributes": "",
        "score": "",
        "label": "",
        "heatmap_shape": "",
        "pred_mask_shape": "",
        "warnings": "",
        "errors": "",
        "load_ms": "",
        "predict_ms": "",
        "status": "unknown_needs_investigation",
    }


def apply_prediction_details(report: dict[str, Any], prediction: Any) -> None:
    report["output_object_type"] = type(prediction).__name__ if prediction is not None else ""
    report["output_keys_or_attributes"] = ",".join(output_fields(prediction))
    score = extract_score(prediction)
    label = extract_pred_label(prediction)
    heatmap = extract_heatmap(prediction)
    pred_mask = named_value(prediction, "pred_mask")
    report["score"] = "" if score is None else float(score)
    report["label"] = "" if label is None else bool(label)
    report["heatmap_shape"] = shape_string(heatmap)
    report["pred_mask_shape"] = shape_string(pred_mask)


def probe_anomalib_openvino(*, artifact_path: Path, image_path: Path, device: str) -> dict[str, Any]:
    probe = {
        "output_object_type": "",
        "output_keys_or_attributes": "",
        "score": "",
        "label": "",
        "heatmap_shape": "",
        "pred_mask_shape": "",
        "warnings": "",
        "errors": "",
        "load_ms": "",
        "predict_ms": "",
    }
    captured_warnings: list[str] = []
    try:
        with warnings.catch_warnings(record=True) as warning_records:
            warnings.simplefilter("always")
            from anomalib.deploy import OpenVINOInferencer

            load_start = time.perf_counter()
            inferencer = OpenVINOInferencer(path=artifact_path, device=openvino_probe_device(device))
            probe["load_ms"] = elapsed_ms(load_start)
            predict_start = time.perf_counter()
            prediction = inferencer.predict(image_path)
            probe["predict_ms"] = elapsed_ms(predict_start)
            probe["output_object_type"] = type(prediction).__name__
            probe["output_keys_or_attributes"] = ",".join(output_fields(prediction))
            score = extract_score(prediction)
            label = extract_pred_label(prediction)
            heatmap = extract_heatmap(prediction)
            pred_mask = named_value(prediction, "pred_mask")
            probe["score"] = "" if score is None else float(score)
            probe["label"] = "" if label is None else bool(label)
            probe["heatmap_shape"] = shape_string(heatmap)
            probe["pred_mask_shape"] = shape_string(pred_mask)
            captured_warnings.extend(format_warnings(warning_records))
    except Exception as exc:
        probe["errors"] = str(exc)
    probe["warnings"] = join_messages(captured_warnings)
    return probe


def flatten_probe(report: dict[str, Any], probe: dict[str, Any]) -> None:
    for key, value in probe.items():
        report[f"probe_{key}"] = value


def output_fields(prediction: Any) -> list[str]:
    if prediction is None:
        return []
    if isinstance(prediction, dict):
        return [str(key) for key in prediction.keys()]
    fields = [name for name in KNOWN_OUTPUT_FIELDS if hasattr(prediction, name)]
    if fields:
        return fields
    return [
        name
        for name in dir(prediction)
        if not name.startswith("_") and not callable(getattr(prediction, name, None))
    ][:30]


def named_value(prediction: Any, name: str) -> Any:
    if prediction is None:
        return None
    if isinstance(prediction, dict):
        return prediction.get(name)
    return getattr(prediction, name, None)


def shape_string(value: Any) -> str:
    if value is None:
        return ""
    try:
        return "x".join(str(dim) for dim in np.asarray(value).shape)
    except Exception:
        return ""


def elapsed_ms(start: float) -> float:
    return round((time.perf_counter() - start) * 1000.0, 3)


def format_warnings(warning_records: list[warnings.WarningMessage]) -> list[str]:
    return [f"{record.category.__name__}: {record.message}" for record in warning_records]


def join_messages(messages: list[str]) -> str:
    return " | ".join(str(message) for message in messages if str(message))


def canonical_model_name(model_name: str) -> str:
    return model_name.strip().lower().replace("-", "_")


def normalize_format(model_format: str) -> str:
    normalized = model_format.strip().lower()
    if normalized in {"pt", "torch"}:
        return "torch_export"
    if normalized in {"xml"}:
        return "openvino"
    return normalized


def openvino_probe_device(device: str) -> str:
    normalized = device.strip().lower()
    if normalized in {"", "auto"}:
        return "AUTO"
    if normalized == "cpu":
        return "CPU"
    if normalized == "cuda":
        return "GPU"
    return device.strip().upper()


def write_json(report: dict[str, Any], output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")


def write_csv(report: dict[str, Any], output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {field: report.get(field, "") for field in CSV_FIELDS}
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerow(row)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate one Anomalib image-model artifact through the app runtime.")
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--artifact-path", required=True)
    parser.add_argument("--format", required=True, choices=("ckpt", "torch_export", "openvino"))
    parser.add_argument("--image-path", required=True)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--anomaly-threshold", type=float, default=0.5)
    parser.add_argument("--output-json")
    parser.add_argument("--output-csv")
    parser.add_argument("--probe-openvino-inferencer", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = validate_artifact(
        model_name=args.model_name,
        artifact_path=args.artifact_path,
        model_format=args.format,
        image_path=args.image_path,
        device=args.device,
        anomaly_threshold=args.anomaly_threshold,
        probe_openvino_inferencer=args.probe_openvino_inferencer,
    )
    if args.output_json:
        write_json(report, args.output_json)
    if args.output_csv:
        write_csv(report, args.output_csv)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "verified_pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
