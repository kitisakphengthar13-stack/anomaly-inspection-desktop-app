import cv2
import numpy as np
import pytest

from anomaly_inspection.core.config import PresenceConfig
from anomaly_inspection.core.presence_checker import PresenceChecker
from anomaly_inspection.core.result_types import PresenceStatus
from anomaly_inspection.core.zone_io import PolygonZone, ZoneConfig


def make_checker(tmp_path, reference):
    reference_path = tmp_path / "reference.png"
    cv2.imwrite(str(reference_path), reference)
    config = PresenceConfig(
        reference_image_path=reference_path,
        zones_path=tmp_path / "zones.json",
        pixel_diff_threshold=20,
        min_foreground_ratio=0.05,
        min_blob_area=20,
        blur_kernel_size=0,
        morphology_kernel_size=0,
        use_largest_blob_filter=True,
    )
    zones = ZoneConfig(
        image_width=reference.shape[1],
        image_height=reference.shape[0],
        zones=[PolygonZone(id="zone_1", points=[(10, 10), (90, 10), (90, 90), (10, 90)])],
    )
    return PresenceChecker(config, zones)


def test_empty_current_image_vs_reference_returns_no_part(tmp_path):
    reference = np.full((100, 100, 3), 255, dtype=np.uint8)
    checker = make_checker(tmp_path, reference)

    result = checker.check(reference.copy())

    assert result.status == PresenceStatus.NO_PART
    assert result.foreground_ratio == 0
    assert result.largest_blob_area == 0


def test_colored_rectangle_inside_polygon_returns_part_present(tmp_path):
    reference = np.full((100, 100, 3), 255, dtype=np.uint8)
    current = reference.copy()
    current[30:70, 30:70] = (0, 0, 255)
    checker = make_checker(tmp_path, reference)

    result = checker.check(current)

    assert result.status == PresenceStatus.PART_PRESENT
    assert result.foreground_ratio > 0.05
    assert result.largest_blob_area >= 20


def test_image_shape_mismatch_raises_clear_error(tmp_path):
    reference = np.full((100, 100, 3), 255, dtype=np.uint8)
    current = np.full((80, 100, 3), 255, dtype=np.uint8)
    checker = make_checker(tmp_path, reference)

    with pytest.raises(ValueError, match="do not match reference image dimensions"):
        checker.check(current)
