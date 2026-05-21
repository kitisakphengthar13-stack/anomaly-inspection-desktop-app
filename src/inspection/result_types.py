from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np


class PresenceStatus(str, Enum):
    PART_PRESENT = "PART_PRESENT"
    NO_PART = "NO_PART"
    ERROR = "ERROR"


class FinalResult(str, Enum):
    NO_PART = "NO_PART"
    OK = "OK"
    NG = "NG"
    ERROR = "ERROR"


@dataclass
class PresenceResult:
    status: PresenceStatus
    foreground_ratio: float = 0.0
    mean_diff: float = 0.0
    largest_blob_area: float = 0.0
    changed_pixel_count: int = 0
    zone_pixel_count: int = 0
    presence_mask_path: Optional[str] = None
    error_message: Optional[str] = None


@dataclass
class AnomalyResult:
    anomaly_score: Optional[float] = None
    pred_label: Optional[bool] = None
    backend_name: Optional[str] = None
    model_path: Optional[str] = None
    heatmap_path: Optional[str] = None
    heatmap: Optional[np.ndarray] = None
    error_message: Optional[str] = None


@dataclass
class InspectionResult:
    image_path: str
    final_result: FinalResult
    run_id: Optional[str] = None
    timestamp: Optional[str] = None
    inspection_job: Optional[str] = None
    inspection_job_slug: Optional[str] = None
    inspection_mode: Optional[str] = None
    presence_status: Optional[PresenceStatus] = None
    foreground_ratio: Optional[float] = None
    mean_diff: Optional[float] = None
    largest_blob_area: Optional[float] = None
    changed_pixel_count: Optional[int] = None
    zone_pixel_count: Optional[int] = None
    anomaly_ran: bool = False
    anomaly_score: Optional[float] = None
    anomaly_pred_label: Optional[bool] = None
    anomaly_backend: Optional[str] = None
    anomaly_model_path: Optional[str] = None
    anomaly_threshold: Optional[float] = None
    annotated_image_path: Optional[str] = None
    heatmap_path: Optional[str] = None
    presence_mask_path: Optional[str] = None
    presence_time_ms: Optional[float] = None
    anomaly_time_ms: Optional[float] = None
    total_time_ms: Optional[float] = None
    error_message: Optional[str] = None

    def to_csv_row(self) -> dict:
        return {
            "run_id": self.run_id or "",
            "timestamp": self.timestamp or "",
            "inspection_job": self.inspection_job or "",
            "inspection_job_slug": self.inspection_job_slug or "",
            "inspection_mode": self.inspection_mode or "",
            "image_name": self.image_path.split("\\")[-1].split("/")[-1],
            "image_path": self.image_path,
            "final_result": self.final_result.value,
            "presence_status": self.presence_status.value if self.presence_status else "",
            "foreground_ratio": "" if self.foreground_ratio is None else f"{self.foreground_ratio:.6f}",
            "mean_diff": "" if self.mean_diff is None else f"{self.mean_diff:.6f}",
            "largest_blob_area": "" if self.largest_blob_area is None else f"{self.largest_blob_area:.2f}",
            "changed_pixel_count": "" if self.changed_pixel_count is None else str(self.changed_pixel_count),
            "zone_pixel_count": "" if self.zone_pixel_count is None else str(self.zone_pixel_count),
            "anomaly_ran": str(self.anomaly_ran).lower(),
            "anomaly_backend": self.anomaly_backend or "",
            "anomaly_pred_label": "" if self.anomaly_pred_label is None else str(self.anomaly_pred_label),
            "anomaly_score": "" if self.anomaly_score is None else f"{self.anomaly_score:.6f}",
            "fallback_anomaly_threshold": "" if self.anomaly_threshold is None else f"{self.anomaly_threshold:.6f}",
            "annotated_image_path": self.annotated_image_path or "",
            "heatmap_path": self.heatmap_path or "",
            "presence_mask_path": self.presence_mask_path or "",
            "presence_time_ms": "" if self.presence_time_ms is None else f"{self.presence_time_ms:.3f}",
            "anomaly_time_ms": "" if self.anomaly_time_ms is None else f"{self.anomaly_time_ms:.3f}",
            "total_time_ms": "" if self.total_time_ms is None else f"{self.total_time_ms:.3f}",
            "error_message": self.error_message or "",
        }
