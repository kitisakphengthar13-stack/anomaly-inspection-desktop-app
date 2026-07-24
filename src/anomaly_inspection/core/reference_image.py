from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def save_reference_image(frame: np.ndarray, output_path: str | Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), frame):
        raise ValueError(f"Could not save reference image to {output_path}")
    return output_path
