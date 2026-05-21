import json

import cv2
import numpy as np

from inspection.zone_editor import OpenCVZoneEditor


def make_editor(tmp_path):
    image_path = tmp_path / "reference.png"
    cv2.imwrite(str(image_path), np.zeros((80, 100, 3), dtype=np.uint8))
    return OpenCVZoneEditor(image_path, tmp_path / "zones.json")


def test_zone_editor_does_not_overwrite_with_empty_invalid_zones(tmp_path):
    editor = make_editor(tmp_path)
    editor.zones_path.write_text('{"keep": true}', encoding="utf-8")

    editor._save_zones()

    assert json.loads(editor.zones_path.read_text(encoding="utf-8")) == {"keep": True}
    assert editor.status == "No valid polygon to save."


def test_zone_editor_save_finalizes_valid_current_polygon(tmp_path):
    editor = make_editor(tmp_path)
    editor.current_points = [(10, 10), (40, 10), (40, 40)]

    editor._save_zones()

    data = json.loads(editor.zones_path.read_text(encoding="utf-8"))
    assert data["image_width"] == 100
    assert data["image_height"] == 80
    assert len(data["zones"]) == 1
    assert data["zones"][0]["points"] == [[10, 10], [40, 10], [40, 40]]
