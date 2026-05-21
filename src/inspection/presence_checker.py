from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from inspection.config import PresenceConfig
from inspection.result_types import PresenceResult, PresenceStatus
from inspection.zone_io import ZoneConfig, assert_zone_shape_matches, zones_to_mask


class PresenceChecker:
    def __init__(self, config: PresenceConfig, zone_config: ZoneConfig):
        self.config = config
        self.zone_config = zone_config
        self.reference_image = self._read_image(config.reference_image_path, "reference image")
        assert_zone_shape_matches(zone_config, self.reference_image.shape, "reference image")
        self.zone_mask = zones_to_mask(zone_config)
        self.zone_pixel_count = int(np.count_nonzero(self.zone_mask))
        if self.zone_pixel_count <= 0:
            raise ValueError("Combined polygon mask has zero pixels.")

    @staticmethod
    def _read_image(path: str | Path, label: str) -> np.ndarray:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Could not read {label}: {path}")
        return image

    def check_image_path(self, image_path: str | Path, debug_mask_path: Optional[str | Path] = None) -> PresenceResult:
        current = self._read_image(image_path, "inspection image")
        return self.check(current, debug_mask_path=debug_mask_path)

    def check(self, current_image: np.ndarray, debug_mask_path: Optional[str | Path] = None) -> PresenceResult:
        if current_image.shape[:2] != self.reference_image.shape[:2]:
            raise ValueError(
                "Inspection image dimensions "
                f"({current_image.shape[1]}x{current_image.shape[0]}) do not match reference image dimensions "
                f"({self.reference_image.shape[1]}x{self.reference_image.shape[0]})."
            )

        current = current_image
        reference = self.reference_image
        if self.config.blur_kernel_size and self.config.blur_kernel_size > 1:
            ksize = (self.config.blur_kernel_size, self.config.blur_kernel_size)
            current = cv2.GaussianBlur(current, ksize, 0)
            reference = cv2.GaussianBlur(reference, ksize, 0)

        diff = cv2.absdiff(current, reference)
        diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        masked_diff = cv2.bitwise_and(diff_gray, diff_gray, mask=self.zone_mask)

        _, changed = cv2.threshold(masked_diff, self.config.pixel_diff_threshold, 255, cv2.THRESH_BINARY)
        changed = cv2.bitwise_and(changed, changed, mask=self.zone_mask)

        if self.config.morphology_kernel_size and self.config.morphology_kernel_size > 1:
            kernel = np.ones((self.config.morphology_kernel_size, self.config.morphology_kernel_size), np.uint8)
            changed = cv2.morphologyEx(changed, cv2.MORPH_OPEN, kernel)
            changed = cv2.morphologyEx(changed, cv2.MORPH_CLOSE, kernel)

        changed_pixel_count = int(np.count_nonzero(changed))
        foreground_ratio = changed_pixel_count / float(self.zone_pixel_count)
        mean_diff = float(masked_diff[self.zone_mask > 0].mean())
        largest_blob_area = self._largest_blob_area(changed)

        ratio_pass = foreground_ratio >= self.config.min_foreground_ratio
        blob_pass = True
        if self.config.use_largest_blob_filter:
            blob_pass = largest_blob_area >= self.config.min_blob_area
        status = PresenceStatus.PART_PRESENT if ratio_pass and blob_pass else PresenceStatus.NO_PART

        saved_mask_path = None
        if debug_mask_path is not None:
            debug_mask_path = Path(debug_mask_path)
            debug_mask_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(debug_mask_path), changed)
            saved_mask_path = str(debug_mask_path)

        return PresenceResult(
            status=status,
            foreground_ratio=foreground_ratio,
            mean_diff=mean_diff,
            largest_blob_area=largest_blob_area,
            changed_pixel_count=changed_pixel_count,
            zone_pixel_count=self.zone_pixel_count,
            presence_mask_path=saved_mask_path,
        )

    @staticmethod
    def _largest_blob_area(binary_mask: np.ndarray) -> float:
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return 0.0
        return float(max(cv2.contourArea(contour) for contour in contours))
