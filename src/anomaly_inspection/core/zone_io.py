from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import cv2
import numpy as np

Point = Tuple[int, int]


@dataclass(frozen=True)
class PolygonZone:
    id: str
    points: List[Point]
    type: str = "polygon"


@dataclass(frozen=True)
class ZoneConfig:
    image_width: int
    image_height: int
    zones: List[PolygonZone]


def _validate_point(point: Sequence[int], width: int, height: int, zone_id: str) -> Point:
    if len(point) != 2:
        raise ValueError(f"Zone '{zone_id}' contains an invalid point: {point!r}.")
    x, y = int(point[0]), int(point[1])
    if not (0 <= x < width and 0 <= y < height):
        raise ValueError(
            f"Zone '{zone_id}' point ({x}, {y}) is outside image bounds "
            f"width={width}, height={height}."
        )
    return x, y


def validate_zone_config(zone_config: ZoneConfig) -> None:
    if zone_config.image_width <= 0 or zone_config.image_height <= 0:
        raise ValueError("Zone image_width and image_height must be positive.")
    if not zone_config.zones:
        raise ValueError("Zone file must contain at least one polygon zone.")
    for zone in zone_config.zones:
        if zone.type != "polygon":
            raise ValueError(f"Unsupported zone type '{zone.type}' for zone '{zone.id}'.")
        if len(zone.points) < 3:
            raise ValueError(f"Zone '{zone.id}' must contain at least 3 points.")
        for point in zone.points:
            _validate_point(point, zone_config.image_width, zone_config.image_height, zone.id)


def load_zones(path: str | Path) -> ZoneConfig:
    path = Path(path)
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    try:
        width = int(data["image_width"])
        height = int(data["image_height"])
        raw_zones = data["zones"]
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Invalid zone file structure: {path}") from exc

    zones: List[PolygonZone] = []
    if not isinstance(raw_zones, list):
        raise ValueError("Zone file 'zones' field must be a list.")
    for idx, raw_zone in enumerate(raw_zones, start=1):
        if not isinstance(raw_zone, dict):
            raise ValueError(f"Zone entry #{idx} must be an object.")
        zone_id = str(raw_zone.get("id") or f"zone_{idx}")
        zone_type = str(raw_zone.get("type", "polygon"))
        raw_points = raw_zone.get("points")
        if not isinstance(raw_points, list):
            raise ValueError(f"Zone '{zone_id}' points must be a list.")
        points = [_validate_point(point, width, height, zone_id) for point in raw_points]
        zones.append(PolygonZone(id=zone_id, points=points, type=zone_type))

    zone_config = ZoneConfig(image_width=width, image_height=height, zones=zones)
    validate_zone_config(zone_config)
    return zone_config


def save_zones(path: str | Path, image_width: int, image_height: int, polygons: Iterable[Sequence[Point]]) -> None:
    zones = [
        {"id": f"zone_{idx}", "type": "polygon", "points": [[int(x), int(y)] for x, y in points]}
        for idx, points in enumerate(polygons, start=1)
    ]
    config = ZoneConfig(
        image_width=int(image_width),
        image_height=int(image_height),
        zones=[PolygonZone(id=item["id"], points=[tuple(point) for point in item["points"]]) for item in zones],
    )
    validate_zone_config(config)
    payload = {"image_width": int(image_width), "image_height": int(image_height), "zones": zones}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)


def zones_to_mask(zone_config: ZoneConfig) -> np.ndarray:
    mask = np.zeros((zone_config.image_height, zone_config.image_width), dtype=np.uint8)
    for zone in zone_config.zones:
        points = np.array(zone.points, dtype=np.int32)
        cv2.fillPoly(mask, [points], 255)
    return mask


def assert_zone_shape_matches(zone_config: ZoneConfig, image_shape: tuple[int, ...], label: str = "image") -> None:
    height, width = image_shape[:2]
    if width != zone_config.image_width or height != zone_config.image_height:
        raise ValueError(
            f"Zone dimensions ({zone_config.image_width}x{zone_config.image_height}) do not match "
            f"{label} dimensions ({width}x{height})."
        )
