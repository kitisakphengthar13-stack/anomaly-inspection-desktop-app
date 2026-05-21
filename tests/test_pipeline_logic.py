import csv
from pathlib import Path

import cv2
import numpy as np

from inspection.config import InspectionConfig, ModelConfig, OutputConfig, PresenceConfig, ProjectConfig
from inspection.pipeline import (
    CSV_FIELDNAMES,
    InspectionPipeline,
    append_inspection_log_csv,
    folder_artifact_stem,
    folder_run_output_dir,
    single_inspection_artifact_stem,
    unique_folder_artifact_stems,
    write_summary_csv,
)
from inspection.result_types import FinalResult, InspectionResult
from inspection.zone_io import save_zones


class StubAnomalyInferencer:
    def __init__(self, score, pred_label=None, heatmap=None):
        self.score = score
        self.pred_label = pred_label
        self.heatmap = heatmap
        self.calls = 0

    def predict(self, image, heatmap_path=None):
        from inspection.result_types import AnomalyResult

        self.calls += 1
        saved_heatmap = None
        if heatmap_path:
            from pathlib import Path

            heatmap_path = Path(heatmap_path)
            heatmap_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(heatmap_path), np.zeros(image.shape[:2], dtype=np.uint8))
            saved_heatmap = str(heatmap_path)
        return AnomalyResult(
            anomaly_score=self.score,
            pred_label=self.pred_label,
            heatmap_path=saved_heatmap,
            heatmap=self.heatmap,
        )


def make_csv_result(image_name: str, final_result: FinalResult = FinalResult.OK) -> InspectionResult:
    return InspectionResult(
        image_path=image_name,
        final_result=final_result,
        run_id=f"run_{image_name}",
        timestamp="2026-05-15T10:00:00",
        anomaly_threshold=0.5,
    )


def make_config(tmp_path, reference_path, zones_path, threshold=0.5):
    return InspectionConfig(
        project=ProjectConfig(name="Default Job"),
        model=ModelConfig(path=tmp_path / "missing.ckpt", anomaly_threshold=threshold, device="cpu", format="auto"),
        presence=PresenceConfig(
            reference_image_path=reference_path,
            zones_path=zones_path,
            pixel_diff_threshold=20,
            min_foreground_ratio=0.05,
            min_blob_area=20,
            blur_kernel_size=0,
            morphology_kernel_size=0,
            use_largest_blob_filter=True,
        ),
        output=OutputConfig(
            save_annotated=False,
            save_heatmap=False,
            save_presence_mask=False,
            organize_by_result=False,
            show_images=False,
            save_csv_log=False,
        ),
        config_path=tmp_path / "inspection.yaml",
    )


def make_pipeline_assets(tmp_path):
    reference = np.full((100, 100, 3), 255, dtype=np.uint8)
    reference_path = tmp_path / "reference.png"
    zones_path = tmp_path / "zones.json"
    cv2.imwrite(str(reference_path), reference)
    save_zones(zones_path, 100, 100, [[(10, 10), (90, 10), (90, 90), (10, 90)]])
    empty_path = tmp_path / "empty.png"
    part_path = tmp_path / "part.png"
    cv2.imwrite(str(empty_path), reference)
    part = reference.copy()
    part[30:70, 30:70] = (0, 0, 255)
    cv2.imwrite(str(part_path), part)
    return reference_path, zones_path, empty_path, part_path


def test_no_part_skips_anomaly_inference(tmp_path):
    reference_path, zones_path, empty_path, _ = make_pipeline_assets(tmp_path)
    stub = StubAnomalyInferencer(score=0.99)
    pipeline = InspectionPipeline(make_config(tmp_path, reference_path, zones_path), anomaly_inferencer=stub)

    result = pipeline.inspect_image(empty_path, tmp_path / "out")

    assert result.final_result == FinalResult.NO_PART
    assert stub.calls == 0


def test_part_present_low_score_returns_ok(tmp_path):
    reference_path, zones_path, _, part_path = make_pipeline_assets(tmp_path)
    stub = StubAnomalyInferencer(score=0.1, pred_label=None)
    pipeline = InspectionPipeline(make_config(tmp_path, reference_path, zones_path, threshold=0.5), anomaly_inferencer=stub)

    result = pipeline.inspect_image(part_path, tmp_path / "out")

    assert result.final_result == FinalResult.OK
    assert stub.calls == 1


def test_part_present_high_score_returns_ng(tmp_path):
    reference_path, zones_path, _, part_path = make_pipeline_assets(tmp_path)
    stub = StubAnomalyInferencer(score=0.9, pred_label=None)
    pipeline = InspectionPipeline(make_config(tmp_path, reference_path, zones_path, threshold=0.5), anomaly_inferencer=stub)

    result = pipeline.inspect_image(part_path, tmp_path / "out")

    assert result.final_result == FinalResult.NG
    assert stub.calls == 1


def test_part_present_pred_label_true_returns_ng(tmp_path):
    reference_path, zones_path, _, part_path = make_pipeline_assets(tmp_path)
    stub = StubAnomalyInferencer(score=0.1, pred_label=True)
    pipeline = InspectionPipeline(make_config(tmp_path, reference_path, zones_path, threshold=0.5), anomaly_inferencer=stub)

    result = pipeline.inspect_image(part_path, tmp_path / "out")

    assert result.final_result == FinalResult.NG
    assert result.anomaly_score == 0.1
    assert result.anomaly_pred_label is True


def test_part_present_pred_label_false_returns_ok(tmp_path):
    reference_path, zones_path, _, part_path = make_pipeline_assets(tmp_path)
    stub = StubAnomalyInferencer(score=0.9, pred_label=False)
    pipeline = InspectionPipeline(make_config(tmp_path, reference_path, zones_path, threshold=0.5), anomaly_inferencer=stub)

    result = pipeline.inspect_image(part_path, tmp_path / "out")

    assert result.final_result == FinalResult.OK
    assert result.anomaly_score == 0.9
    assert result.anomaly_pred_label is False


def test_show_images_displays_saved_outputs(tmp_path, monkeypatch):
    reference_path, zones_path, _, part_path = make_pipeline_assets(tmp_path)
    base = make_config(tmp_path, reference_path, zones_path)
    config = InspectionConfig(
        project=base.project,
        model=base.model,
        presence=base.presence,
        output=OutputConfig(
            save_annotated=True,
            save_heatmap=False,
            save_presence_mask=True,
            organize_by_result=False,
            show_images=True,
            save_csv_log=False,
        ),
        config_path=base.config_path,
    )
    stub = StubAnomalyInferencer(score=0.1, pred_label=False)
    shown = []
    waits = []
    closes = []

    monkeypatch.setattr(cv2, "imshow", lambda title, image: shown.append(title))
    monkeypatch.setattr(cv2, "waitKey", lambda delay: waits.append(delay) or 0)
    monkeypatch.setattr(cv2, "destroyAllWindows", lambda: closes.append(True))

    pipeline = InspectionPipeline(config, anomaly_inferencer=stub)
    result = pipeline.inspect_image(part_path, tmp_path / "out")

    assert result.final_result == FinalResult.OK
    assert shown == ["Annotated", "Presence mask"]
    assert waits == [0]
    assert closes == [True]


def test_inspect_image_writes_csv_log_when_enabled(tmp_path):
    reference_path, zones_path, _, part_path = make_pipeline_assets(tmp_path)
    base = make_config(tmp_path, reference_path, zones_path)
    config = InspectionConfig(
        project=base.project,
        model=base.model,
        presence=base.presence,
        output=OutputConfig(
            save_annotated=False,
            save_heatmap=False,
            save_presence_mask=False,
            organize_by_result=False,
            show_images=False,
            save_csv_log=True,
        ),
        config_path=base.config_path,
    )
    pipeline = InspectionPipeline(config, anomaly_inferencer=StubAnomalyInferencer(score=0.1, pred_label=False))

    result = pipeline.inspect_image(part_path, tmp_path / "out")
    log_path = tmp_path / "out" / "inspection_log.csv"

    assert result.final_result == FinalResult.OK
    assert log_path.exists()
    with log_path.open("r", encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    assert len(rows) == 1
    row = rows[0]
    assert list(row.keys()) == CSV_FIELDNAMES
    assert row["run_id"]
    assert row["timestamp"]
    assert row["inspection_job"] == "Default Job"
    assert row["inspection_job_slug"] == "default_job"
    assert row["inspection_mode"] == "image"
    assert row["image_name"] == "part.png"
    assert row["final_result"] == "OK"
    assert row["anomaly_ran"] == "true"
    assert row["fallback_anomaly_threshold"] == "0.500000"
    assert row["presence_time_ms"] != ""
    assert row["anomaly_time_ms"] != ""
    assert row["total_time_ms"] != ""


def test_append_inspection_log_csv_creates_file_with_header_and_row(tmp_path):
    log_path = tmp_path / "inspection_log.csv"

    append_inspection_log_csv(make_csv_result("first.png"), log_path)

    with log_path.open("r", encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    assert len(rows) == 1
    assert rows[0]["image_name"] == "first.png"
    assert log_path.read_text(encoding="utf-8").count(",".join(CSV_FIELDNAMES)) == 1


def test_append_inspection_log_csv_preserves_existing_rows_and_header_once(tmp_path):
    log_path = tmp_path / "inspection_log.csv"

    append_inspection_log_csv(make_csv_result("first.png"), log_path)
    append_inspection_log_csv(make_csv_result("second.png", FinalResult.NG), log_path)

    text = log_path.read_text(encoding="utf-8")
    with log_path.open("r", encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    assert [row["image_name"] for row in rows] == ["first.png", "second.png"]
    assert rows[1]["final_result"] == "NG"
    assert text.count(",".join(CSV_FIELDNAMES)) == 1


def test_append_inspection_log_csv_writes_header_for_empty_existing_file(tmp_path):
    log_path = tmp_path / "inspection_log.csv"
    log_path.write_text("", encoding="utf-8")

    append_inspection_log_csv(make_csv_result("first.png"), log_path)

    with log_path.open("r", encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    assert len(rows) == 1
    assert rows[0]["image_name"] == "first.png"
    assert log_path.read_text(encoding="utf-8").count(",".join(CSV_FIELDNAMES)) == 1


def test_inspect_image_appends_csv_log_when_invoked_repeatedly(tmp_path):
    reference_path, zones_path, _, part_path = make_pipeline_assets(tmp_path)
    base = make_config(tmp_path, reference_path, zones_path)
    config = InspectionConfig(
        project=base.project,
        model=base.model,
        presence=base.presence,
        output=OutputConfig(
            save_annotated=False,
            save_heatmap=False,
            save_presence_mask=False,
            organize_by_result=False,
            show_images=False,
            save_csv_log=True,
        ),
        config_path=base.config_path,
    )
    pipeline = InspectionPipeline(config, anomaly_inferencer=StubAnomalyInferencer(score=0.1, pred_label=False))

    pipeline.inspect_image(part_path, tmp_path / "out")
    pipeline.inspect_image(part_path, tmp_path / "out")

    log_path = tmp_path / "out" / "inspection_log.csv"
    with log_path.open("r", encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    assert len(rows) == 2
    assert [row["image_name"] for row in rows] == ["part.png", "part.png"]
    assert log_path.read_text(encoding="utf-8").count(",".join(CSV_FIELDNAMES)) == 1


def test_append_inspection_log_csv_upgrades_legacy_header_in_place(tmp_path):
    log_path = tmp_path / "inspection_log.csv"
    legacy_fields = [field for field in CSV_FIELDNAMES if field not in {"inspection_job", "inspection_job_slug", "inspection_mode"}]
    with log_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=legacy_fields)
        writer.writeheader()
        writer.writerow(
            {field: "" for field in legacy_fields}
            | {
                "run_id": "old",
                "timestamp": "2026-05-15T10:00:00",
                "image_name": "old.png",
                "image_path": "old.png",
                "final_result": "OK",
            }
        )
    result = make_csv_result("new.png")
    result.inspection_job = "Metal Surface"
    result.inspection_job_slug = "metal_surface"
    result.inspection_mode = "image"

    append_inspection_log_csv(result, log_path)

    with log_path.open("r", encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    assert list(rows[0].keys()) == CSV_FIELDNAMES
    assert len(rows) == 2
    assert rows[0]["image_name"] == "old.png"
    assert rows[0]["inspection_job"] == "Metal Surface"
    assert rows[0]["inspection_job_slug"] == "metal_surface"
    assert rows[0]["inspection_mode"] == "image"
    assert rows[1]["image_name"] == "new.png"


def test_pipeline_can_write_camera_mode_metadata(tmp_path):
    reference_path, zones_path, _, part_path = make_pipeline_assets(tmp_path)
    base = make_config(tmp_path, reference_path, zones_path)
    config = InspectionConfig(
        project=ProjectConfig(name="Metal Surface"),
        model=base.model,
        presence=base.presence,
        output=OutputConfig(
            save_annotated=False,
            save_heatmap=False,
            save_presence_mask=False,
            organize_by_result=False,
            show_images=False,
            save_csv_log=True,
        ),
        config_path=base.config_path,
    )
    pipeline = InspectionPipeline(config, anomaly_inferencer=StubAnomalyInferencer(score=0.1, pred_label=False))

    pipeline.inspect_image(part_path, tmp_path / "out", inspection_mode="camera")

    with (tmp_path / "out" / "inspection_log.csv").open("r", encoding="utf-8", newline="") as file:
        row = next(csv.DictReader(file))
    assert row["inspection_job"] == "Metal Surface"
    assert row["inspection_job_slug"] == "metal_surface"
    assert row["inspection_mode"] == "camera"


def test_single_inspection_artifact_stem_includes_run_id():
    assert single_inspection_artifact_stem("001", "20260515_153012_123456") == "001_20260515_153012_123456"


def test_folder_artifact_stem_uses_relative_path_context():
    assert folder_artifact_stem("ok/001.png") == "ok_001"
    assert folder_artifact_stem("ng/001.png") == "ng_001"
    assert folder_artifact_stem("line 1/Part-A.png") == "line_1_Part_A"


def test_unique_folder_artifact_stems_suffixes_sanitized_collisions(tmp_path):
    input_dir = tmp_path / "input"
    first = input_dir / "a-b" / "001.png"
    second = input_dir / "a" / "b-001.png"
    first.parent.mkdir(parents=True)
    second.parent.mkdir(parents=True)
    first.write_bytes(b"")
    second.write_bytes(b"")

    stems = unique_folder_artifact_stems([first, second], input_dir)

    assert stems[first] == "a_b_001"
    assert stems[second] == "a_b_001_2"


def test_folder_run_output_dir_uses_run_prefix():
    assert folder_run_output_dir("outputs/transistor/folder", "20260515_173012_123456") == Path(
        "outputs/transistor/folder/run_20260515_173012_123456"
    )


def test_repeated_single_image_runs_keep_unique_artifact_paths_and_csv_traceability(tmp_path):
    reference_path, zones_path, _, part_path = make_pipeline_assets(tmp_path)
    base = make_config(tmp_path, reference_path, zones_path)
    config = InspectionConfig(
        project=base.project,
        model=base.model,
        presence=base.presence,
        output=OutputConfig(
            save_annotated=True,
            save_heatmap=True,
            save_presence_mask=True,
            organize_by_result=True,
            show_images=False,
            save_csv_log=True,
        ),
        config_path=base.config_path,
    )
    pipeline = InspectionPipeline(config, anomaly_inferencer=StubAnomalyInferencer(score=0.1, pred_label=False))

    first = pipeline.inspect_image(part_path, tmp_path / "out")
    second = pipeline.inspect_image(part_path, tmp_path / "out")

    assert first.annotated_image_path != second.annotated_image_path
    assert first.heatmap_path != second.heatmap_path
    assert first.presence_mask_path != second.presence_mask_path
    assert Path(first.annotated_image_path).exists()
    assert Path(second.annotated_image_path).exists()
    assert Path(first.heatmap_path).exists()
    assert Path(second.heatmap_path).exists()
    assert Path(first.presence_mask_path).exists()
    assert Path(second.presence_mask_path).exists()
    assert Path(first.annotated_image_path).name == f"part_{first.run_id}_annotated.png"
    assert Path(second.heatmap_path).name == f"part_{second.run_id}_heatmap.png"

    ok_copies = sorted((tmp_path / "out" / "OK").glob("*.png"))
    assert len(ok_copies) == 2
    assert {path.name for path in ok_copies} == {
        Path(first.annotated_image_path).name,
        Path(second.annotated_image_path).name,
    }

    log_path = tmp_path / "out" / "inspection_log.csv"
    with log_path.open("r", encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    assert len(rows) == 2
    assert rows[0]["annotated_image_path"] == first.annotated_image_path
    assert rows[1]["annotated_image_path"] == second.annotated_image_path
    assert rows[0]["heatmap_path"] == first.heatmap_path
    assert rows[1]["heatmap_path"] == second.heatmap_path
    assert rows[0]["presence_mask_path"] == first.presence_mask_path
    assert rows[1]["presence_mask_path"] == second.presence_mask_path


def test_annotated_contours_use_runtime_heatmap_when_heatmap_output_disabled(tmp_path):
    reference_path, zones_path, _, part_path = make_pipeline_assets(tmp_path)
    heatmap = np.zeros((20, 20), dtype=np.float32)
    heatmap[6:14, 6:14] = 1.0
    base = make_config(tmp_path, reference_path, zones_path)
    config = InspectionConfig(
        project=base.project,
        model=base.model,
        presence=base.presence,
        output=OutputConfig(
            save_annotated=True,
            save_heatmap=False,
            save_presence_mask=False,
            organize_by_result=False,
            show_images=False,
            save_csv_log=False,
        ),
        config_path=base.config_path,
    )
    pipeline = InspectionPipeline(
        config,
        anomaly_inferencer=StubAnomalyInferencer(score=0.9, pred_label=True, heatmap=heatmap),
    )

    result = pipeline.inspect_image(part_path, tmp_path / "out")

    assert result.final_result == FinalResult.NG
    assert result.heatmap_path is None
    assert result.annotated_image_path is not None
    annotated = cv2.imread(result.annotated_image_path, cv2.IMREAD_COLOR)
    assert annotated is not None
    red_pixels = (annotated[:, :, 2] > 180) & (annotated[:, :, 1] < 80) & (annotated[:, :, 0] < 80)
    assert np.count_nonzero(red_pixels) > 0


def test_repeated_single_image_runs_keep_unique_result_copies_without_annotated_output(tmp_path):
    reference_path, zones_path, _, part_path = make_pipeline_assets(tmp_path)
    base = make_config(tmp_path, reference_path, zones_path)
    config = InspectionConfig(
        project=base.project,
        model=base.model,
        presence=base.presence,
        output=OutputConfig(
            save_annotated=False,
            save_heatmap=False,
            save_presence_mask=False,
            organize_by_result=True,
            show_images=False,
            save_csv_log=True,
        ),
        config_path=base.config_path,
    )
    pipeline = InspectionPipeline(config, anomaly_inferencer=StubAnomalyInferencer(score=0.1, pred_label=False))

    first = pipeline.inspect_image(part_path, tmp_path / "out")
    second = pipeline.inspect_image(part_path, tmp_path / "out")

    ok_copies = sorted((tmp_path / "out" / "OK").glob("*.png"))
    assert len(ok_copies) == 2
    assert {path.name for path in ok_copies} == {f"part_{first.run_id}.png", f"part_{second.run_id}.png"}


def test_inspect_image_does_not_write_csv_log_when_disabled(tmp_path):
    reference_path, zones_path, _, part_path = make_pipeline_assets(tmp_path)
    pipeline = InspectionPipeline(make_config(tmp_path, reference_path, zones_path), anomaly_inferencer=StubAnomalyInferencer(score=0.1, pred_label=False))

    pipeline.inspect_image(part_path, tmp_path / "out")

    assert not (tmp_path / "out" / "inspection_log.csv").exists()


def test_write_summary_csv_overwrites_batch_summary(tmp_path):
    summary_path = tmp_path / "summary.csv"

    write_summary_csv([make_csv_result("old.png")], summary_path)
    write_summary_csv([make_csv_result("new.png")], summary_path)

    with summary_path.open("r", encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    assert len(rows) == 1
    assert rows[0]["image_name"] == "new.png"


def test_inspect_folder_respects_csv_log_option(tmp_path):
    reference_path, zones_path, empty_path, part_path = make_pipeline_assets(tmp_path)
    batch_dir = tmp_path / "batch"
    batch_dir.mkdir()
    empty = cv2.imread(str(empty_path), cv2.IMREAD_COLOR)
    part = cv2.imread(str(part_path), cv2.IMREAD_COLOR)
    cv2.imwrite(str(batch_dir / "empty.png"), empty)
    cv2.imwrite(str(batch_dir / "part.png"), part)
    base = make_config(tmp_path, reference_path, zones_path)
    config = InspectionConfig(
        project=base.project,
        model=base.model,
        presence=base.presence,
        output=OutputConfig(
            save_annotated=False,
            save_heatmap=False,
            save_presence_mask=False,
            organize_by_result=False,
            show_images=False,
            save_csv_log=True,
        ),
        config_path=base.config_path,
    )
    pipeline = InspectionPipeline(config, anomaly_inferencer=StubAnomalyInferencer(score=0.1, pred_label=False))

    results = pipeline.inspect_folder(batch_dir, tmp_path / "out_enabled")

    assert len(results) == 2
    assert pipeline.last_folder_output_dir is not None
    assert pipeline.last_folder_output_dir.parent == tmp_path / "out_enabled"
    assert pipeline.last_folder_output_dir.name.startswith("run_")
    summary_path = pipeline.last_folder_output_dir / "summary.csv"
    assert summary_path.exists()
    with summary_path.open("r", encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    assert len(rows) == 2
    assert len({row["run_id"] for row in rows}) == 1
    assert {row["inspection_job"] for row in rows} == {"Default Job"}
    assert {row["inspection_job_slug"] for row in rows} == {"default_job"}
    assert {row["inspection_mode"] for row in rows} == {"folder"}

    disabled = make_config(tmp_path, reference_path, zones_path)
    pipeline = InspectionPipeline(disabled, anomaly_inferencer=StubAnomalyInferencer(score=0.1, pred_label=False))
    pipeline.inspect_folder(batch_dir, tmp_path / "out_disabled")

    assert not (tmp_path / "out_disabled" / "summary.csv").exists()
    assert pipeline.last_folder_output_dir is not None
    assert not (pipeline.last_folder_output_dir / "summary.csv").exists()


def test_repeated_folder_runs_create_distinct_run_directories(tmp_path):
    reference_path, zones_path, _, part_path = make_pipeline_assets(tmp_path)
    batch_dir = tmp_path / "batch"
    batch_dir.mkdir()
    cv2.imwrite(str(batch_dir / "part.png"), cv2.imread(str(part_path), cv2.IMREAD_COLOR))
    base = make_config(tmp_path, reference_path, zones_path)
    config = InspectionConfig(
        project=base.project,
        model=base.model,
        presence=base.presence,
        output=OutputConfig(
            save_annotated=False,
            save_heatmap=False,
            save_presence_mask=False,
            organize_by_result=False,
            show_images=False,
            save_csv_log=True,
        ),
        config_path=base.config_path,
    )

    first_pipeline = InspectionPipeline(config, anomaly_inferencer=StubAnomalyInferencer(score=0.1, pred_label=False))
    first_pipeline.inspect_folder(batch_dir, tmp_path / "out")
    second_pipeline = InspectionPipeline(config, anomaly_inferencer=StubAnomalyInferencer(score=0.1, pred_label=False))
    second_pipeline.inspect_folder(batch_dir, tmp_path / "out")

    assert first_pipeline.last_folder_output_dir != second_pipeline.last_folder_output_dir
    assert (first_pipeline.last_folder_output_dir / "summary.csv").exists()
    assert (second_pipeline.last_folder_output_dir / "summary.csv").exists()


def test_folder_inspection_writes_artifacts_inside_run_directory(tmp_path):
    reference_path, zones_path, empty_path, part_path = make_pipeline_assets(tmp_path)
    batch_dir = tmp_path / "batch"
    batch_dir.mkdir()
    cv2.imwrite(str(batch_dir / "empty.png"), cv2.imread(str(empty_path), cv2.IMREAD_COLOR))
    cv2.imwrite(str(batch_dir / "part.png"), cv2.imread(str(part_path), cv2.IMREAD_COLOR))
    base = make_config(tmp_path, reference_path, zones_path)
    config = InspectionConfig(
        project=base.project,
        model=base.model,
        presence=base.presence,
        output=OutputConfig(
            save_annotated=True,
            save_heatmap=False,
            save_presence_mask=True,
            organize_by_result=True,
            show_images=False,
            save_csv_log=True,
        ),
        config_path=base.config_path,
    )
    pipeline = InspectionPipeline(config, anomaly_inferencer=StubAnomalyInferencer(score=0.1, pred_label=False))

    results = pipeline.inspect_folder(batch_dir, tmp_path / "out")

    assert len(results) == 2
    assert pipeline.last_folder_output_dir is not None
    run_dir = pipeline.last_folder_output_dir
    assert (run_dir / "annotated" / "part_annotated.png").exists()
    assert (run_dir / "presence_masks" / "part_presence_mask.png").exists()
    assert (run_dir / "summary.csv").exists()
    assert not (tmp_path / "out" / "summary.csv").exists()
    assert not (tmp_path / "out" / "annotated" / "part_annotated.png").exists()


def test_folder_inspection_avoids_duplicate_stem_artifact_collisions(tmp_path):
    reference_path, zones_path, _, part_path = make_pipeline_assets(tmp_path)
    batch_dir = tmp_path / "batch"
    ok_dir = batch_dir / "ok"
    ng_dir = batch_dir / "ng"
    ok_dir.mkdir(parents=True)
    ng_dir.mkdir(parents=True)
    part = cv2.imread(str(part_path), cv2.IMREAD_COLOR)
    cv2.imwrite(str(ok_dir / "001.png"), part)
    cv2.imwrite(str(ng_dir / "001.png"), part)
    base = make_config(tmp_path, reference_path, zones_path)
    config = InspectionConfig(
        project=base.project,
        model=base.model,
        presence=base.presence,
        output=OutputConfig(
            save_annotated=True,
            save_heatmap=True,
            save_presence_mask=True,
            organize_by_result=True,
            show_images=False,
            save_csv_log=True,
        ),
        config_path=base.config_path,
    )
    pipeline = InspectionPipeline(config, anomaly_inferencer=StubAnomalyInferencer(score=0.1, pred_label=False))

    results = pipeline.inspect_folder(batch_dir, tmp_path / "out")

    run_dir = pipeline.last_folder_output_dir
    assert len(results) == 2
    assert (run_dir / "annotated" / "ok_001_annotated.png").exists()
    assert (run_dir / "annotated" / "ng_001_annotated.png").exists()
    assert (run_dir / "heatmaps" / "ok_001_heatmap.png").exists()
    assert (run_dir / "heatmaps" / "ng_001_heatmap.png").exists()
    assert (run_dir / "presence_masks" / "ok_001_presence_mask.png").exists()
    assert (run_dir / "presence_masks" / "ng_001_presence_mask.png").exists()
    ok_copies = sorted((run_dir / "OK").glob("*.png"))
    assert {path.name for path in ok_copies} == {"ok_001_annotated.png", "ng_001_annotated.png"}
    with (run_dir / "summary.csv").open("r", encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    assert {Path(row["annotated_image_path"]).name for row in rows} == {
        "ok_001_annotated.png",
        "ng_001_annotated.png",
    }


def test_folder_result_copies_are_collision_safe_without_annotated_output(tmp_path):
    reference_path, zones_path, _, part_path = make_pipeline_assets(tmp_path)
    batch_dir = tmp_path / "batch"
    left_dir = batch_dir / "left"
    right_dir = batch_dir / "right"
    left_dir.mkdir(parents=True)
    right_dir.mkdir(parents=True)
    part = cv2.imread(str(part_path), cv2.IMREAD_COLOR)
    cv2.imwrite(str(left_dir / "001.png"), part)
    cv2.imwrite(str(right_dir / "001.png"), part)
    base = make_config(tmp_path, reference_path, zones_path)
    config = InspectionConfig(
        project=base.project,
        model=base.model,
        presence=base.presence,
        output=OutputConfig(
            save_annotated=False,
            save_heatmap=False,
            save_presence_mask=False,
            organize_by_result=True,
            show_images=False,
            save_csv_log=True,
        ),
        config_path=base.config_path,
    )
    pipeline = InspectionPipeline(config, anomaly_inferencer=StubAnomalyInferencer(score=0.1, pred_label=False))

    pipeline.inspect_folder(batch_dir, tmp_path / "out")

    run_dir = pipeline.last_folder_output_dir
    ok_copies = sorted((run_dir / "OK").glob("*.png"))
    assert {path.name for path in ok_copies} == {"left_001.png", "right_001.png"}
