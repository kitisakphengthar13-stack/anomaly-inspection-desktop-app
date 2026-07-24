from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from anomaly_inspection.core.result_types import FinalResult, InspectionResult


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def contour_thickness_for_image(image_shape: tuple[int, ...]) -> int:
    min_dim = min(image_shape[:2])
    return int(_clamp(round(min_dim / 360.0), 2, 5))


def normalize_heatmap_to_uint8(heatmap: np.ndarray | None, image_shape: tuple[int, ...]) -> Optional[np.ndarray]:
    if heatmap is None:
        return None
    heatmap_arr = np.asarray(heatmap)
    if heatmap_arr.size == 0:
        return None
    heatmap_arr = np.squeeze(heatmap_arr)
    if heatmap_arr.ndim == 3:
        heatmap_arr = np.squeeze(heatmap_arr)
        if heatmap_arr.ndim == 3:
            heatmap_arr = heatmap_arr[..., 0]
    if heatmap_arr.ndim != 2:
        return None

    target_height, target_width = image_shape[:2]
    heatmap_arr = np.nan_to_num(heatmap_arr.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    if heatmap_arr.shape[:2] != (target_height, target_width):
        heatmap_arr = cv2.resize(heatmap_arr, (target_width, target_height), interpolation=cv2.INTER_LINEAR)

    min_value = float(np.min(heatmap_arr))
    max_value = float(np.max(heatmap_arr))
    if max_value <= min_value:
        return np.zeros((target_height, target_width), dtype=np.uint8)
    normalized = (heatmap_arr - min_value) / (max_value - min_value)
    return np.clip(normalized * 255.0, 0, 255).astype(np.uint8)


def build_defect_mask_from_heatmap(
    heatmap: np.ndarray | None,
    image_shape: tuple[int, ...],
    threshold_ratio: float = 0.65,
) -> Optional[np.ndarray]:
    normalized = normalize_heatmap_to_uint8(heatmap, image_shape)
    if normalized is None:
        return None
    threshold_value = int(round(_clamp(threshold_ratio, 0.0, 1.0) * 255))
    _, mask = cv2.threshold(normalized, threshold_value, 255, cv2.THRESH_BINARY)
    if np.count_nonzero(mask) == 0:
        return mask
    kernel = np.ones((3, 3), np.uint8)
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)


def extract_defect_contours(mask: np.ndarray | None, min_area: int) -> list[np.ndarray]:
    if mask is None or mask.size == 0:
        return []
    mask_arr = np.asarray(mask)
    if mask_arr.ndim != 2:
        return []
    contours, _ = cv2.findContours(mask_arr.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return sorted(
        [contour for contour in contours if cv2.contourArea(contour) >= min_area],
        key=cv2.contourArea,
        reverse=True,
    )


def draw_defect_contours(image: np.ndarray, contours: list[np.ndarray]) -> np.ndarray:
    output = image.copy()
    if contours:
        cv2.drawContours(
            output,
            contours,
            contourIdx=-1,
            color=(0, 0, 255),
            thickness=contour_thickness_for_image(image.shape),
            lineType=cv2.LINE_AA,
        )
    return output


def save_annotated_image(
    image: np.ndarray,
    result: InspectionResult,
    output_path: str | Path,
    anomaly_heatmap: np.ndarray | None = None,
) -> str:
    annotated = image.copy()
    if result.final_result == FinalResult.NG:
        mask = build_defect_mask_from_heatmap(anomaly_heatmap, image.shape)
        image_area = int(image.shape[0] * image.shape[1])
        min_area = max(20, int(round(image_area * 0.0002)))
        contours = extract_defect_contours(mask, min_area)
        annotated = draw_defect_contours(annotated, contours)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), annotated)
    return str(output_path)


def save_heatmap_overlay(image: np.ndarray, heatmap: np.ndarray, output_path: str | Path) -> str:
    heatmap_arr = np.asarray(heatmap)
    heatmap_arr = np.squeeze(heatmap_arr)
    if heatmap_arr.ndim == 3:
        heatmap_arr = np.squeeze(heatmap_arr)
        if heatmap_arr.ndim == 3:
            heatmap_arr = heatmap_arr[..., 0]
    if heatmap_arr.ndim != 2:
        raise ValueError(f"Expected 2D heatmap after squeezing, got shape {heatmap_arr.shape}.")
    heatmap_arr = heatmap_arr.astype(np.float32)
    heatmap_arr -= float(np.min(heatmap_arr))
    max_value = float(np.max(heatmap_arr))
    if max_value > 0:
        heatmap_arr /= max_value
    heatmap_arr = (heatmap_arr * 255).astype(np.uint8)
    if heatmap_arr.shape[:2] != image.shape[:2]:
        heatmap_arr = cv2.resize(heatmap_arr, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)
    colored = cv2.applyColorMap(heatmap_arr, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(image, 0.55, colored, 0.45, 0)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), overlay)
    return str(output_path)


def try_save_heatmap(image: np.ndarray, heatmap: Optional[np.ndarray], output_path: str | Path) -> Optional[str]:
    if heatmap is None:
        return None
    try:
        return save_heatmap_overlay(image, heatmap, output_path)
    except Exception:
        return None
