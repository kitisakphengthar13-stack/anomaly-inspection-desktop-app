from __future__ import annotations

import csv
import re
import shutil
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Iterable, Optional

import cv2

from inspection.anomaly_inferencer import AnomalyInferencer
from inspection.config import InspectionConfig, make_project_slug, normalize_project_name
from inspection.presence_checker import PresenceChecker
from inspection.result_types import FinalResult, InspectionResult, PresenceStatus
from inspection.visualization import save_annotated_image
from inspection.zone_io import assert_zone_shape_matches, load_zones

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
CSV_FIELDNAMES = [
    "run_id",
    "timestamp",
    "inspection_job",
    "inspection_job_slug",
    "inspection_mode",
    "image_name",
    "image_path",
    "final_result",
    "presence_status",
    "foreground_ratio",
    "mean_diff",
    "largest_blob_area",
    "changed_pixel_count",
    "zone_pixel_count",
    "anomaly_ran",
    "anomaly_backend",
    "anomaly_pred_label",
    "anomaly_score",
    "fallback_anomaly_threshold",
    "annotated_image_path",
    "heatmap_path",
    "presence_mask_path",
    "presence_time_ms",
    "anomaly_time_ms",
    "total_time_ms",
    "error_message",
]


class InspectionPipeline:
    def __init__(self, config: InspectionConfig, anomaly_inferencer: Optional[AnomalyInferencer] = None):
        self.config = config
        self.run_id = make_run_id()
        self.zone_config = load_zones(config.presence.zones_path)
        self.presence_checker = PresenceChecker(config.presence, self.zone_config)
        assert_zone_shape_matches(self.zone_config, self.presence_checker.reference_image.shape, "reference image")
        self.anomaly_inferencer = anomaly_inferencer
        self.last_folder_output_dir: Path | None = None

    def _get_anomaly_inferencer(self) -> AnomalyInferencer:
        if self.anomaly_inferencer is None:
            self.anomaly_inferencer = AnomalyInferencer(
                model_path=self.config.model.path,
                anomaly_threshold=self.config.model.anomaly_threshold,
                device=self.config.model.device,
                model_format=self.config.model.format,
                anomalib_model=self.config.model.anomalib_model,
                checkpoint_inference_mode=self.config.model.checkpoint_inference_mode,
            )
        return self.anomaly_inferencer

    def prepare_anomaly_backend(self) -> None:
        """Load the anomaly backend without running an inspection."""
        self._get_anomaly_inferencer().load()

    def inspect_image(
        self,
        image_path: str | Path,
        output_dir: str | Path,
        write_csv_log: bool = True,
        *,
        artifact_stem: str | None = None,
        run_id: str | None = None,
        inspection_mode: str = "image",
    ) -> InspectionResult:
        image_path = Path(image_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        stem = image_path.stem
        result_run_id = run_id or (make_run_id() if write_csv_log else self.run_id)
        if artifact_stem is None:
            artifact_stem = single_inspection_artifact_stem(stem, result_run_id) if write_csv_log else stem
        presence_mask_path = output_dir / "presence_masks" / f"{artifact_stem}_presence_mask.png"
        heatmap_path = output_dir / "heatmaps" / f"{artifact_stem}_heatmap.png"
        annotated_path = output_dir / "annotated" / f"{artifact_stem}_annotated.png"
        total_start = perf_counter()
        result = InspectionResult(
            image_path=str(image_path),
            final_result=FinalResult.ERROR,
            run_id=result_run_id,
            timestamp=current_timestamp(),
            inspection_job=self.inspection_job_name,
            inspection_job_slug=self.inspection_job_slug,
            inspection_mode=inspection_mode,
            anomaly_threshold=self.config.model.anomaly_threshold,
        )

        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            result.error_message = f"Could not read inspection image: {image_path}"
            result.total_time_ms = elapsed_ms(total_start)
            self._write_single_image_log(result, output_dir, write_csv_log)
            return result

        anomaly_heatmap = None
        try:
            debug_mask = presence_mask_path if self.config.output.save_presence_mask else None
            presence_start = perf_counter()
            presence = self.presence_checker.check(image, debug_mask_path=debug_mask)
            presence_time_ms = elapsed_ms(presence_start)
            result = InspectionResult(
                image_path=str(image_path),
                final_result=FinalResult.NO_PART,
                run_id=result_run_id,
                timestamp=result.timestamp,
                inspection_job=self.inspection_job_name,
                inspection_job_slug=self.inspection_job_slug,
                inspection_mode=inspection_mode,
                presence_status=presence.status,
                foreground_ratio=presence.foreground_ratio,
                mean_diff=presence.mean_diff,
                largest_blob_area=presence.largest_blob_area,
                changed_pixel_count=presence.changed_pixel_count,
                zone_pixel_count=presence.zone_pixel_count,
                anomaly_threshold=self.config.model.anomaly_threshold,
                presence_mask_path=presence.presence_mask_path,
                presence_time_ms=presence_time_ms,
                anomaly_time_ms=0.0,
            )

            if presence.status == PresenceStatus.NO_PART:
                result.final_result = FinalResult.NO_PART
            else:
                result.anomaly_ran = True
                anomaly_start = perf_counter()
                anomaly = self._get_anomaly_inferencer().predict(
                    image,
                    heatmap_path=heatmap_path if self.config.output.save_heatmap else None,
                )
                result.anomaly_time_ms = elapsed_ms(anomaly_start)
                result.heatmap_path = anomaly.heatmap_path
                anomaly_heatmap = anomaly.heatmap
                if anomaly.error_message:
                    result.final_result = FinalResult.ERROR
                    result.error_message = anomaly.error_message
                else:
                    result.anomaly_score = anomaly.anomaly_score
                    result.anomaly_pred_label = anomaly.pred_label
                    result.anomaly_backend = anomaly.backend_name
                    result.anomaly_model_path = anomaly.model_path
                    if anomaly.pred_label is True:
                        result.final_result = FinalResult.NG
                    elif anomaly.pred_label is False:
                        result.final_result = FinalResult.OK
                    else:
                        result.final_result = (
                            FinalResult.NG
                            if anomaly.anomaly_score is not None
                            and anomaly.anomaly_score >= self.config.model.anomaly_threshold
                            else FinalResult.OK
                        )

        except Exception as exc:
            result.final_result = FinalResult.ERROR
            result.error_message = str(exc)

        if self.config.output.save_annotated:
            try:
                result.annotated_image_path = save_annotated_image(
                    image,
                    result,
                    annotated_path,
                    anomaly_heatmap=anomaly_heatmap,
                )
            except Exception as exc:
                result.error_message = f"{result.error_message or ''} Annotated output failed: {exc}".strip()
                if result.final_result != FinalResult.ERROR:
                    result.final_result = FinalResult.ERROR

        if self.config.output.organize_by_result:
            self._copy_to_result_folder(image_path, result, output_dir, artifact_stem)
        result.total_time_ms = elapsed_ms(total_start)
        self._write_single_image_log(result, output_dir, write_csv_log)
        if self.config.output.show_images:
            show_result_images(result)
        return result

    def inspect_folder(self, input_dir: str | Path, output_dir: str | Path) -> list[InspectionResult]:
        input_dir = Path(input_dir)
        output_base_dir = Path(output_dir)
        output_base_dir.mkdir(parents=True, exist_ok=True)
        folder_run_id = make_run_id()
        output_dir = create_folder_run_output_dir(output_base_dir, folder_run_id)
        self.last_folder_output_dir = output_dir
        image_paths = list(iter_image_files(input_dir))
        artifact_stems = unique_folder_artifact_stems(image_paths, input_dir)
        results = [
            self.inspect_image(
                path,
                output_dir,
                write_csv_log=False,
                artifact_stem=artifact_stems[path],
                run_id=folder_run_id,
                inspection_mode="folder",
            )
            for path in image_paths
        ]
        if self.config.output.save_csv_log:
            write_summary_csv(results, output_dir / "summary.csv")
        return results

    def _write_single_image_log(self, result: InspectionResult, output_dir: Path, write_csv_log: bool) -> None:
        if write_csv_log and self.config.output.save_csv_log:
            append_inspection_log_csv(result, output_dir / "inspection_log.csv")

    @property
    def inspection_job_name(self) -> str:
        return normalize_project_name(self.config.project.name)

    @property
    def inspection_job_slug(self) -> str:
        return make_project_slug(self.inspection_job_name)

    @staticmethod
    def _copy_to_result_folder(image_path: Path, result: InspectionResult, output_dir: Path, artifact_stem: str) -> None:
        result_dir = output_dir / result.final_result.value
        result_dir.mkdir(parents=True, exist_ok=True)
        source = Path(result.annotated_image_path) if result.annotated_image_path else image_path
        if source.exists():
            destination_name = source.name if result.annotated_image_path else f"{artifact_stem}{image_path.suffix}"
            shutil.copy2(source, result_dir / destination_name)


def iter_image_files(input_dir: str | Path) -> Iterable[Path]:
    input_dir = Path(input_dir)
    for path in sorted(input_dir.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            yield path


def single_inspection_artifact_stem(image_stem: str, run_id: str) -> str:
    return f"{image_stem}_{run_id}"


def folder_artifact_stem(relative_image_path: str | Path) -> str:
    relative_path = Path(relative_image_path)
    raw_parts = (*relative_path.parent.parts, relative_path.stem)
    joined = "_".join(part for part in raw_parts if part not in ("", "."))
    stem = re.sub(r"[^A-Za-z0-9]+", "_", joined)
    stem = re.sub(r"_+", "_", stem).strip("_")
    return stem or "image"


def unique_folder_artifact_stems(image_paths: list[Path], input_dir: str | Path) -> dict[Path, str]:
    input_dir = Path(input_dir)
    used: set[str] = set()
    stems: dict[Path, str] = {}
    for image_path in image_paths:
        base_stem = folder_artifact_stem(image_path.relative_to(input_dir))
        candidate = base_stem
        suffix = 2
        while candidate in used:
            candidate = f"{base_stem}_{suffix}"
            suffix += 1
        used.add(candidate)
        stems[image_path] = candidate
    return stems


def folder_run_output_dir(output_base_dir: str | Path, run_id: str) -> Path:
    return Path(output_base_dir) / f"run_{run_id}"


def create_folder_run_output_dir(output_base_dir: str | Path, run_id: str) -> Path:
    base_path = folder_run_output_dir(output_base_dir, run_id)
    candidate = base_path
    suffix = 1
    while candidate.exists():
        candidate = Path(f"{base_path}_{suffix:03d}")
        suffix += 1
    candidate.mkdir(parents=True, exist_ok=False)
    return candidate


def write_summary_csv(results: list[InspectionResult], output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()
        for result in results:
            writer.writerow(result.to_csv_row())


def append_inspection_log_csv(result: InspectionResult, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and output_path.stat().st_size > 0:
        _upgrade_csv_schema_if_needed(output_path, result)
    should_write_header = not output_path.exists() or output_path.stat().st_size == 0
    with output_path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_FIELDNAMES)
        if should_write_header:
            writer.writeheader()
        writer.writerow(result.to_csv_row())


def _upgrade_csv_schema_if_needed(output_path: Path, result: InspectionResult) -> None:
    with output_path.open("r", newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        fieldnames = reader.fieldnames or []
        if fieldnames == CSV_FIELDNAMES:
            return
        rows = list(reader)

    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_FIELDNAMES, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(_expanded_legacy_csv_row(row, result))


def _expanded_legacy_csv_row(row: dict[str, str | None], result: InspectionResult) -> dict[str, str]:
    expanded = {field: (row.get(field) or "") for field in CSV_FIELDNAMES}
    expanded["inspection_job"] = expanded["inspection_job"] or result.inspection_job or ""
    expanded["inspection_job_slug"] = expanded["inspection_job_slug"] or result.inspection_job_slug or ""
    expanded["inspection_mode"] = expanded["inspection_mode"] or result.inspection_mode or ""
    return expanded


def make_run_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")


def current_timestamp() -> str:
    return datetime.now().isoformat(timespec="seconds")


def elapsed_ms(start: float) -> float:
    return (perf_counter() - start) * 1000.0


def show_result_images(result: InspectionResult) -> None:
    windows = [
        ("Annotated", result.annotated_image_path),
        ("Heatmap", result.heatmap_path),
        ("Presence mask", result.presence_mask_path),
    ]
    shown = False
    for title, path in windows:
        if not path or not Path(path).exists():
            continue
        image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if image is None:
            continue
        cv2.imshow(title, image)
        shown = True
    if shown:
        cv2.waitKey(0)
        cv2.destroyAllWindows()
