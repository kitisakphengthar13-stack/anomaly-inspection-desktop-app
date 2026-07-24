import cv2
import numpy as np

from anomaly_inspection.cli_support.reference_capture import save_reference_image


def test_save_reference_image_creates_parent_directory(tmp_path):
    frame = np.full((32, 48, 3), 127, dtype=np.uint8)
    output_path = tmp_path / "nested" / "empty_reference.png"

    saved_path = save_reference_image(frame, output_path)

    assert saved_path == output_path
    saved = cv2.imread(str(output_path), cv2.IMREAD_COLOR)
    assert saved is not None
    assert saved.shape[:2] == (32, 48)
