from __future__ import annotations

import csv
import statistics
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Iterable

import cv2

from anomaly_inspection.core.anomaly_inferencer import AnomalyInferencer
from anomaly_inspection.core.config import InspectionConfig
from anomaly_inspection.core.pipeline import IMAGE_EXTENSIONS


@dataclass
class ValidationRow:
    image_path: str
    category: str
    ground_truth_label: str
    pred_score: float | None
    pred_label: bool | None
    correct: bool
    error: str = ""
    inference_seconds: float = 0.0
    backend_name: str = ""


def iter_labeled_images(
    test_root: str | Path,
    normal_folders: list[str],
    abnormal_folders: list[str],
) -> Iterable[tuple[Path, str, str]]:
    test_root = Path(test_root)
    folder_labels = {name: "normal" for name in normal_folders}
    folder_labels.update({name: "abnormal" for name in abnormal_folders})
    for folder_name, label in folder_labels.items():
        folder = test_root / folder_name
        if not folder.is_dir():
            raise ValueError(f"Expected test folder does not exist: {folder}")
        for path in sorted(folder.rglob("*")):
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                yield path, folder_name, label


def run_anomaly_validation(
    config: InspectionConfig,
    test_root: str | Path,
    normal_folders: list[str],
    abnormal_folders: list[str],
    output_dir: str | Path,
) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    image_items = list(iter_labeled_images(test_root, normal_folders, abnormal_folders))
    inferencer = AnomalyInferencer(
        model_path=config.model.path,
        anomaly_threshold=config.model.anomaly_threshold,
        device=config.model.device,
        model_format=config.model.format,
        anomalib_model=config.model.anomalib_model,
        checkpoint_inference_mode=config.model.checkpoint_inference_mode,
    )
    inferencer.load()

    rows: list[ValidationRow] = []
    for image_path, category, gt_label in image_items:
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            rows.append(
                ValidationRow(
                    image_path=str(image_path),
                    category=category,
                    ground_truth_label=gt_label,
                    pred_score=None,
                    pred_label=None,
                    correct=False,
                    error="Could not read image.",
                    backend_name=inferencer.backend_name or "",
                )
            )
            continue
        start = perf_counter()
        result = inferencer.predict(image)
        elapsed = perf_counter() - start
        gt_abnormal = gt_label == "abnormal"
        correct = result.pred_label == gt_abnormal if result.pred_label is not None else False
        rows.append(
            ValidationRow(
                image_path=str(image_path),
                category=category,
                ground_truth_label=gt_label,
                pred_score=result.anomaly_score,
                pred_label=result.pred_label,
                correct=correct,
                error=result.error_message or "",
                inference_seconds=elapsed,
                backend_name=result.backend_name or inferencer.backend_name or "",
            )
        )

    predictions_csv = output_dir / "predictions.csv"
    write_predictions_csv(rows, predictions_csv)
    valid_rows = [row for row in rows if not row.error and row.pred_score is not None]
    summary = build_validation_summary(valid_rows)
    folder_metrics = {
        folder: compute_metrics([row for row in valid_rows if row.category == folder])
        for folder in normal_folders + abnormal_folders
    }
    distributions = {
        folder: score_distribution([row.pred_score for row in valid_rows if row.category == folder])
        for folder in normal_folders + abnormal_folders
    }
    distributions["all_abnormal"] = score_distribution(
        [row.pred_score for row in valid_rows if row.ground_truth_label == "abnormal"]
    )
    sweep_rows = threshold_sweep(valid_rows)
    sweep_csv = output_dir / "threshold_sweep.csv"
    write_dict_csv(sweep_rows, sweep_csv)
    best_f1 = max(sweep_rows, key=lambda row: (row["f1"], row["recall"], -row["false_positive_rate"])) if sweep_rows else {}
    current_05 = compute_metrics(valid_rows, threshold=0.5)
    false_negatives = sorted(
        [row for row in valid_rows if row.ground_truth_label == "abnormal" and row.pred_label is False],
        key=lambda row: row.pred_score if row.pred_score is not None else -1,
        reverse=True,
    )
    false_positives = sorted(
        [row for row in valid_rows if row.ground_truth_label == "normal" and row.pred_label is True],
        key=lambda row: row.pred_score if row.pred_score is not None else -1,
        reverse=True,
    )
    write_error_list(false_negatives, output_dir / "false_negatives.csv")
    write_error_list(false_positives, output_dir / "false_positives.csv")

    report = {
        "prediction_csv": str(predictions_csv),
        "threshold_sweep_csv": str(sweep_csv),
        "false_negatives_csv": str(output_dir / "false_negatives.csv"),
        "false_positives_csv": str(output_dir / "false_positives.csv"),
        "total_images": len(valid_rows),
        "backend_name": inferencer.backend_name,
        "overall": summary,
        "threshold_0_5": current_05,
        "best_f1": best_f1,
        "folder_metrics": folder_metrics,
        "score_distributions": distributions,
        "false_negative_count": len(false_negatives),
        "false_positive_count": len(false_positives),
        "mean_inference_seconds": statistics.mean([row.inference_seconds for row in valid_rows])
        if valid_rows
        else 0.0,
    }
    write_summary_txt(report, output_dir / "summary.txt")
    return report


def compute_metrics(rows: list[ValidationRow], threshold: float | None = None) -> dict:
    tp = tn = fp = fn = 0
    for row in rows:
        gt = row.ground_truth_label == "abnormal"
        pred = row.pred_score >= threshold if threshold is not None and row.pred_score is not None else row.pred_label is True
        if gt and pred:
            tp += 1
        elif not gt and not pred:
            tn += 1
        elif not gt and pred:
            fp += 1
        else:
            fn += 1
    total = tp + tn + fp + fn
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "total": total,
        "TP": tp,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "accuracy": (tp + tn) / total if total else 0.0,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "false_negative_rate": fn / (tp + fn) if tp + fn else 0.0,
        "false_positive_rate": fp / (fp + tn) if fp + tn else 0.0,
    }


def build_validation_summary(rows: list[ValidationRow]) -> dict:
    return compute_metrics(rows)


def score_distribution(values: list[float | None]) -> dict:
    clean = [float(value) for value in values if value is not None]
    if not clean:
        return {"count": 0, "min": "", "max": "", "mean": "", "median": "", "std": ""}
    return {
        "count": len(clean),
        "min": min(clean),
        "max": max(clean),
        "mean": statistics.mean(clean),
        "median": statistics.median(clean),
        "std": statistics.pstdev(clean),
    }


def threshold_sweep(rows: list[ValidationRow]) -> list[dict]:
    return [{"threshold": i / 100.0, **compute_metrics(rows, threshold=i / 100.0)} for i in range(101)]


def write_predictions_csv(rows: list[ValidationRow], output_path: Path) -> None:
    fieldnames = [
        "image_path",
        "category",
        "ground_truth_label",
        "pred_score",
        "pred_label",
        "correct",
        "error",
        "inference_seconds",
        "backend_name",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row.__dict__)


def write_dict_csv(rows: list[dict], output_path: Path) -> None:
    if not rows:
        output_path.write_text("", encoding="utf-8")
        return
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_error_list(rows: list[ValidationRow], output_path: Path) -> None:
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["image_path", "category", "ground_truth_label", "pred_score", "pred_label"])
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "image_path": row.image_path,
                    "category": row.category,
                    "ground_truth_label": row.ground_truth_label,
                    "pred_score": row.pred_score,
                    "pred_label": row.pred_label,
                }
            )


def write_summary_txt(report: dict, output_path: Path) -> None:
    lines = []
    for key in (
        "prediction_csv",
        "threshold_sweep_csv",
        "false_negatives_csv",
        "false_positives_csv",
        "total_images",
        "backend_name",
        "mean_inference_seconds",
    ):
        lines.append(f"{key}={report[key]}")
    for section in ("overall", "threshold_0_5", "best_f1"):
        lines.append(f"{section}: {format_metrics(report[section])}")
    lines.append("folder_metrics:")
    for folder, metrics in report["folder_metrics"].items():
        lines.append(f"  {folder}: {format_metrics(metrics)}")
    lines.append("score_distributions:")
    for folder, distribution in report["score_distributions"].items():
        lines.append(f"  {folder}: {format_metrics(distribution)}")
    lines.append(f"false_negative_count={report['false_negative_count']}")
    lines.append(f"false_positive_count={report['false_positive_count']}")
    output_path.write_text("\n".join(lines), encoding="utf-8")


def format_metrics(values: dict) -> str:
    return ", ".join(f"{key}={value:.6f}" if isinstance(value, float) else f"{key}={value}" for key, value in values.items())
