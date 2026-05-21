import json

import pytest

from inspection.zone_io import load_zones


def write_zone_file(path, payload):
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_valid_zone_json_loads(tmp_path):
    path = tmp_path / "zones.json"
    write_zone_file(
        path,
        {
            "image_width": 100,
            "image_height": 80,
            "zones": [{"id": "zone_1", "type": "polygon", "points": [[1, 1], [50, 1], [50, 50]]}],
        },
    )

    zones = load_zones(path)

    assert zones.image_width == 100
    assert zones.image_height == 80
    assert len(zones.zones) == 1
    assert zones.zones[0].points == [(1, 1), (50, 1), (50, 50)]


def test_invalid_polygon_with_less_than_three_points_fails(tmp_path):
    path = tmp_path / "zones.json"
    write_zone_file(
        path,
        {
            "image_width": 100,
            "image_height": 80,
            "zones": [{"id": "bad", "type": "polygon", "points": [[1, 1], [50, 1]]}],
        },
    )

    with pytest.raises(ValueError, match="at least 3 points"):
        load_zones(path)


def test_out_of_bounds_point_fails(tmp_path):
    path = tmp_path / "zones.json"
    write_zone_file(
        path,
        {
            "image_width": 100,
            "image_height": 80,
            "zones": [{"id": "bad", "type": "polygon", "points": [[1, 1], [120, 1], [50, 50]]}],
        },
    )

    with pytest.raises(ValueError, match="outside image bounds"):
        load_zones(path)
