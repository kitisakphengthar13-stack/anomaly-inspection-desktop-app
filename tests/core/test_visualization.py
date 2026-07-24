import cv2
import numpy as np

from anomaly_inspection.core.result_types import AnomalyResult, FinalResult, InspectionResult, PresenceStatus
from anomaly_inspection.core.visualization import (
    build_defect_mask_from_heatmap,
    contour_thickness_for_image,
    extract_defect_contours,
    normalize_heatmap_to_uint8,
    save_annotated_image,
    save_heatmap_overlay,
)


def make_result(final_result: FinalResult) -> InspectionResult:
    return InspectionResult(
        image_path="part.png",
        final_result=final_result,
        presence_status=PresenceStatus.PART_PRESENT,
        foreground_ratio=0.1234,
        largest_blob_area=1234.0,
        anomaly_score=0.2345,
        anomaly_threshold=0.5,
        anomaly_pred_label=final_result == FinalResult.NG,
    )


def make_heatmap_with_defect(shape: tuple[int, int] = (64, 64)) -> np.ndarray:
    heatmap = np.zeros(shape, dtype=np.float32)
    heatmap[20:42, 24:46] = 1.0
    return heatmap


def test_anomaly_result_supports_runtime_heatmap_data():
    heatmap = make_heatmap_with_defect()

    result = AnomalyResult(anomaly_score=0.9, pred_label=True, heatmap_path="heatmap.png", heatmap=heatmap)

    assert result.heatmap is heatmap
    assert result.heatmap_path == "heatmap.png"


def test_anomaly_result_runtime_heatmap_is_not_serialized_to_csv():
    result = InspectionResult(
        image_path="part.png",
        final_result=FinalResult.NG,
        anomaly_threshold=0.5,
        heatmap_path="heatmap.png",
    )

    row = result.to_csv_row()

    assert "heatmap" not in row
    assert row["heatmap_path"] == "heatmap.png"


def test_normalize_heatmap_to_uint8_resizes_to_source_resolution():
    heatmap = make_heatmap_with_defect((64, 64))

    normalized = normalize_heatmap_to_uint8(heatmap, (80, 120, 3))

    assert normalized is not None
    assert normalized.shape == (80, 120)
    assert normalized.dtype == np.uint8
    assert normalized.max() == 255
    assert normalized.min() == 0


def test_normalize_heatmap_to_uint8_handles_all_zero_heatmap():
    normalized = normalize_heatmap_to_uint8(np.zeros((12, 12), dtype=np.float32), (60, 60, 3))

    assert normalized is not None
    assert normalized.shape == (60, 60)
    assert np.count_nonzero(normalized) == 0


def test_build_defect_mask_from_heatmap_thresholds_strong_regions():
    heatmap = make_heatmap_with_defect()

    mask = build_defect_mask_from_heatmap(heatmap, (128, 128, 3), threshold_ratio=0.65)

    assert mask is not None
    assert mask.shape == (128, 128)
    assert mask.dtype == np.uint8
    assert np.count_nonzero(mask) > 0


def test_extract_defect_contours_filters_tiny_regions():
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[10:12, 10:12] = 255
    mask[40:70, 40:70] = 255

    contours = extract_defect_contours(mask, min_area=100)

    assert len(contours) == 1
    assert cv2.contourArea(contours[0]) >= 100


def test_annotated_ng_with_heatmap_draws_red_contour_only(tmp_path):
    image = np.full((128, 128, 3), 40, dtype=np.uint8)
    output_path = tmp_path / "annotated_ng.png"

    saved_path = save_annotated_image(image, make_result(FinalResult.NG), output_path, make_heatmap_with_defect())

    saved = cv2.imread(saved_path, cv2.IMREAD_COLOR)
    assert saved is not None
    assert saved.shape == image.shape
    red_pixels = (saved[:, :, 2] > 180) & (saved[:, :, 1] < 80) & (saved[:, :, 0] < 80)
    yellow_pixels = (saved[:, :, 1] > 180) & (saved[:, :, 2] > 180) & (saved[:, :, 0] < 80)
    assert np.count_nonzero(red_pixels) > 0
    assert np.count_nonzero(yellow_pixels) == 0


def test_annotated_non_ng_results_remain_clean_source_image(tmp_path):
    image = np.full((128, 128, 3), 40, dtype=np.uint8)
    heatmap = make_heatmap_with_defect()

    for final_result in (FinalResult.OK, FinalResult.NO_PART, FinalResult.ERROR):
        output_path = tmp_path / f"{final_result.value.lower()}_annotated.png"
        saved_path = save_annotated_image(image, make_result(final_result), output_path, heatmap)
        saved = cv2.imread(saved_path, cv2.IMREAD_COLOR)
        assert saved is not None
        assert np.array_equal(saved, image)


def test_annotated_ng_without_usable_heatmap_remains_clean_source_image(tmp_path):
    image = np.full((128, 128, 3), 40, dtype=np.uint8)
    output_path = tmp_path / "annotated_ng_no_heatmap.png"

    saved_path = save_annotated_image(image, make_result(FinalResult.NG), output_path, None)

    saved = cv2.imread(saved_path, cv2.IMREAD_COLOR)
    assert saved is not None
    assert np.array_equal(saved, image)


def test_annotated_image_preserves_source_resolution(tmp_path):
    image = np.full((720, 1280, 3), 30, dtype=np.uint8)
    output_path = tmp_path / "annotated.png"

    saved_path = save_annotated_image(image, make_result(FinalResult.NG), output_path, make_heatmap_with_defect())

    saved = cv2.imread(saved_path, cv2.IMREAD_COLOR)
    assert saved is not None
    assert saved.shape == image.shape


def test_contour_thickness_scales_with_source_resolution():
    assert contour_thickness_for_image((480, 640, 3)) == 2
    assert contour_thickness_for_image((1024, 1024, 3)) == 3
    assert contour_thickness_for_image((1440, 2560, 3)) == 4


def test_heatmap_overlay_still_resizes_heatmap_to_source_resolution(tmp_path):
    image = np.full((720, 1280, 3), 30, dtype=np.uint8)
    heatmap = np.zeros((32, 48), dtype=np.float32)
    output_path = tmp_path / "heatmap.png"

    saved_path = save_heatmap_overlay(image, heatmap, output_path)

    saved = cv2.imread(saved_path, cv2.IMREAD_COLOR)
    assert saved is not None
    assert saved.shape == image.shape
