"""Run image-only Anomalib artifact compatibility checks from a manifest.

This runner does not train, export, or discover models. It validates only
artifacts that already exist locally. A `no_local_artifact` result means the
artifact was not verified in this environment; it does not mean the model is
unsupported. A `verified_pass` result means one artifact loaded and predicted
successfully for one image. Production support should only be claimed after
enough representative artifacts and images pass.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import yaml

TOOLS_ROOT = Path(__file__).resolve().parent
if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))

from validate_anomalib_artifact import (  # noqa: E402
    VIDEO_MODEL_NAMES,
    canonical_model_name,
    validate_artifact,
)


STATUS_VALUES = {
    "verified_pass",
    "verified_fail",
    "no_local_artifact",
    "needs_special_handling",
    "unsupported_not_image_runtime",
    "unsupported_special_runtime",
    "unknown_needs_investigation",
}
SUPPORT_LEVELS = {"public_supported", "experimental", "developer_only", "not_target"}
FORMAT_VALUES = {"ckpt", "torch_export", "openvino"}
ROW_FIELDS = (
    "id",
    "model_name",
    "format",
    "artifact_path",
    "image_path",
    "backend",
    "load_success",
    "predict_success",
    "raw_output_type",
    "output_keys_or_attrs",
    "score",
    "label",
    "heatmap_shape",
    "pred_mask_shape",
    "status",
    "warnings",
    "errors",
    "load_ms",
    "predict_ms",
    "notes",
    "support_level",
    "expected_status",
    "tags",
)


def load_manifest(path: str | Path) -> dict[str, Any]:
    manifest_path = Path(path)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest file does not exist: {manifest_path}")
    try:
        text = manifest_path.read_text(encoding="utf-8")
        if manifest_path.suffix.lower() == ".json":
            data = json.loads(text)
        else:
            data = yaml.safe_load(text)
    except Exception as exc:
        raise ValueError(f"Could not parse manifest {manifest_path}: {exc}") from exc
    validate_manifest(data, manifest_path)
    return data


def validate_manifest(data: Any, manifest_path: Path | None = None) -> None:
    source = f" in {manifest_path}" if manifest_path else ""
    if not isinstance(data, dict):
        raise ValueError(f"Compatibility manifest{source} must be a mapping.")
    artifacts = data.get("artifacts")
    if not isinstance(artifacts, list):
        raise ValueError(f"Compatibility manifest{source} must contain an artifacts list.")
    for index, row in enumerate(artifacts):
        if not isinstance(row, dict):
            raise ValueError(f"Artifact row {index}{source} must be a mapping.")
        row_id = row.get("id", f"row_{index}")
        for field in ("model_name", "format"):
            if not str(row.get(field, "")).strip():
                raise ValueError(f"Artifact row {row_id!r}{source} is missing required field: {field}.")
        model_format = str(row["format"]).strip().lower()
        if model_format not in FORMAT_VALUES:
            raise ValueError(f"Artifact row {row_id!r}{source} has unsupported format {model_format!r}.")
        support_level = str(row.get("support_level", "")).strip()
        if support_level and support_level not in SUPPORT_LEVELS:
            raise ValueError(f"Artifact row {row_id!r}{source} has unsupported support_level {support_level!r}.")
        expected_status = str(row.get("expected_status", "")).strip()
        if expected_status and expected_status not in STATUS_VALUES:
            raise ValueError(f"Artifact row {row_id!r}{source} has unsupported expected_status {expected_status!r}.")


def run_matrix(
    manifest: dict[str, Any],
    *,
    continue_on_error: bool = False,
    probe_openvino_inferencer: bool = False,
    device_override: str | None = None,
    limit: int | None = None,
    filter_model: str | None = None,
    filter_format: str | None = None,
) -> list[dict[str, Any]]:
    validate_manifest(manifest)
    rows = filtered_artifacts(
        manifest["artifacts"],
        limit=limit,
        filter_model=filter_model,
        filter_format=filter_format,
    )
    results: list[dict[str, Any]] = []
    for index, artifact in enumerate(rows):
        try:
            result = validate_manifest_row(
                artifact,
                row_index=index,
                probe_openvino_inferencer=probe_openvino_inferencer,
                device_override=device_override,
            )
        except Exception as exc:
            if not continue_on_error:
                raise
            result = error_row(artifact, index, exc)
        results.append(result)
    return results


def filtered_artifacts(
    artifacts: list[dict[str, Any]],
    *,
    limit: int | None,
    filter_model: str | None,
    filter_format: str | None,
) -> list[dict[str, Any]]:
    rows = artifacts
    if filter_model:
        target_model = canonical_model_name(filter_model)
        rows = [row for row in rows if canonical_model_name(str(row.get("model_name", ""))) == target_model]
    if filter_format:
        target_format = filter_format.strip().lower()
        rows = [row for row in rows if str(row.get("format", "")).strip().lower() == target_format]
    if limit is not None:
        rows = rows[: max(limit, 0)]
    return rows


def validate_manifest_row(
    artifact: dict[str, Any],
    *,
    row_index: int,
    probe_openvino_inferencer: bool,
    device_override: str | None,
) -> dict[str, Any]:
    row_id = str(artifact.get("id") or f"row_{row_index}")
    model_name = str(artifact.get("model_name", "")).strip()
    model_format = str(artifact.get("format", "")).strip().lower()
    artifact_path = str(artifact.get("artifact_path", "") or "").strip()
    image_path = str(artifact.get("image_path", "") or "").strip()
    device = device_override or str(artifact.get("device", "auto") or "auto")
    threshold = float(artifact.get("anomaly_threshold", 0.5))

    if canonical_model_name(model_name) in VIDEO_MODEL_NAMES:
        report = {
            "model_name": model_name,
            "format": model_format,
            "artifact_path": artifact_path,
            "image_path": image_path,
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
            "errors": "Video models are out of scope for this single-image compatibility matrix.",
            "load_ms": "",
            "predict_ms": "",
            "status": "unsupported_not_image_runtime",
        }
    elif not artifact_path:
        report = {
            "model_name": model_name,
            "format": model_format,
            "artifact_path": artifact_path,
            "image_path": image_path,
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
            "errors": "Artifact path is missing.",
            "load_ms": "",
            "predict_ms": "",
            "status": "no_local_artifact",
        }
    elif not image_path:
        report = {
            "model_name": model_name,
            "format": model_format,
            "artifact_path": artifact_path,
            "image_path": image_path,
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
            "errors": "Image path is missing.",
            "load_ms": "",
            "predict_ms": "",
            "status": "verified_fail",
        }
    else:
        report = validate_artifact(
            model_name=model_name,
            artifact_path=artifact_path,
            model_format=model_format,
            image_path=image_path,
            device=device,
            anomaly_threshold=threshold,
            probe_openvino_inferencer=probe_openvino_inferencer,
        )

    return matrix_row(row_id, artifact, report)


def matrix_row(row_id: str, artifact: dict[str, Any], report: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": row_id,
        "model_name": report.get("model_name", artifact.get("model_name", "")),
        "format": report.get("format", artifact.get("format", "")),
        "artifact_path": report.get("artifact_path", artifact.get("artifact_path", "")),
        "image_path": report.get("image_path", artifact.get("image_path", "")),
        "backend": report.get("selected_backend", ""),
        "load_success": report.get("load_success", False),
        "predict_success": report.get("predict_success", False),
        "raw_output_type": report.get("output_object_type", ""),
        "output_keys_or_attrs": report.get("output_keys_or_attributes", ""),
        "score": report.get("score", ""),
        "label": report.get("label", ""),
        "heatmap_shape": report.get("heatmap_shape", ""),
        "pred_mask_shape": report.get("pred_mask_shape", ""),
        "status": report.get("status", "unknown_needs_investigation"),
        "warnings": report.get("warnings", ""),
        "errors": report.get("errors", ""),
        "load_ms": report.get("load_ms", ""),
        "predict_ms": report.get("predict_ms", ""),
        "notes": artifact_notes(artifact),
        "support_level": artifact.get("support_level", ""),
        "expected_status": artifact.get("expected_status", ""),
        "tags": format_tags(artifact.get("tags", "")),
    }


def artifact_notes(artifact: dict[str, Any]) -> str:
    notes = str(artifact.get("notes", "") or "")
    expected_status = str(artifact.get("expected_status", "") or "")
    if expected_status == "needs_special_handling" and "needs_special_handling" not in notes:
        notes = f"{notes} needs_special_handling".strip()
    return notes


def format_tags(tags: Any) -> str:
    if isinstance(tags, (list, tuple)):
        return ",".join(str(tag) for tag in tags)
    return str(tags or "")


def error_row(artifact: dict[str, Any], row_index: int, exc: Exception) -> dict[str, Any]:
    row_id = str(artifact.get("id") or f"row_{row_index}")
    report = {
        "model_name": artifact.get("model_name", ""),
        "format": artifact.get("format", ""),
        "artifact_path": artifact.get("artifact_path", ""),
        "image_path": artifact.get("image_path", ""),
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
        "errors": str(exc),
        "load_ms": "",
        "predict_ms": "",
        "status": "verified_fail",
    }
    return matrix_row(row_id, artifact, report)


def write_json_report(rows: list[dict[str, Any]], output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"summary": summary_counts(rows), "artifacts": rows}
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def write_csv_report(rows: list[dict[str, Any]], output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=ROW_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in ROW_FIELDS})


def summary_counts(rows: list[dict[str, Any]]) -> dict[str, Any]:
    status_counts = Counter(str(row.get("status", "")) for row in rows)
    model_counts = Counter(str(row.get("model_name", "")) for row in rows)
    format_counts = Counter(str(row.get("format", "")) for row in rows)
    return {
        "total_artifacts": len(rows),
        "verified_pass": status_counts.get("verified_pass", 0),
        "verified_fail": status_counts.get("verified_fail", 0),
        "no_local_artifact": status_counts.get("no_local_artifact", 0),
        "unsupported_not_image_runtime": status_counts.get("unsupported_not_image_runtime", 0),
        "needs_special_handling": status_counts.get("needs_special_handling", 0),
        "by_model_name": dict(sorted(model_counts.items())),
        "by_format": dict(sorted(format_counts.items())),
        "by_status": dict(sorted(status_counts.items())),
    }


def print_summary(rows: list[dict[str, Any]]) -> None:
    summary = summary_counts(rows)
    print("Compatibility Matrix Summary")
    print(f"total artifacts: {summary['total_artifacts']}")
    for key in (
        "verified_pass",
        "verified_fail",
        "no_local_artifact",
        "unsupported_not_image_runtime",
        "needs_special_handling",
    ):
        print(f"{key}: {summary[key]}")
    print("by model_name:")
    for model_name, count in summary["by_model_name"].items():
        print(f"  {model_name}: {count}")
    print("by format:")
    for model_format, count in summary["by_format"].items():
        print(f"  {model_format}: {count}")
    print("")
    print("id | model_name | format | backend | status | score | label | notes")
    print("--- | --- | --- | --- | --- | --- | --- | ---")
    for row in rows:
        print(
            f"{row['id']} | {row['model_name']} | {row['format']} | {row['backend']} | "
            f"{row['status']} | {row['score']} | {row['label']} | {row['notes']}"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run image-only Anomalib artifact compatibility validation from a manifest.",
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-json")
    parser.add_argument("--output-csv")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--fail-on-verified-fail", action="store_true")
    parser.add_argument("--probe-openvino-inferencer", action="store_true")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--limit", type=int)
    parser.add_argument("--filter-model")
    parser.add_argument("--filter-format", choices=("ckpt", "torch_export", "openvino"))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest = load_manifest(args.manifest)
    rows = run_matrix(
        manifest,
        continue_on_error=args.continue_on_error,
        probe_openvino_inferencer=args.probe_openvino_inferencer,
        device_override=args.device,
        limit=args.limit,
        filter_model=args.filter_model,
        filter_format=args.filter_format,
    )
    if args.output_json:
        write_json_report(rows, args.output_json)
    if args.output_csv:
        write_csv_report(rows, args.output_csv)
    print_summary(rows)
    if args.fail_on_verified_fail and any(row["status"] == "verified_fail" for row in rows):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
