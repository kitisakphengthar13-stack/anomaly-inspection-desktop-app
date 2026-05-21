from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import cv2

from inspection.toolbar_ui import (
    DisplayLayout,
    ToolbarButton,
    display_to_source,
    initial_canvas_size,
    render_interactive_canvas,
    toolbar_action_at,
    window_canvas_size,
)
from inspection.zone_io import save_zones

Point = Tuple[int, int]
ZONE_ACTIONS = (
    ("undo", "Undo Point"),
    ("finish", "Finish Polygon"),
    ("new", "New Polygon"),
    ("clear", "Clear Current"),
    ("save", "Save Zones"),
    ("quit", "Quit"),
)


class OpenCVZoneEditor:
    def __init__(self, image_path: str | Path, zones_path: str | Path):
        self.image_path = Path(image_path)
        self.zones_path = Path(zones_path)
        self.image = cv2.imread(str(self.image_path), cv2.IMREAD_COLOR)
        if self.image is None:
            raise ValueError(f"Could not read reference image: {self.image_path}")
        self.polygons: List[List[Point]] = []
        self.current_points: List[Point] = []
        self.buttons: list[ToolbarButton] = []
        self.layout: DisplayLayout | None = None
        self.canvas_size = initial_canvas_size(self.image.shape)
        self.status = "Click image to add points. Use toolbar to finish and save."
        self.unsaved_changes = False
        self.quit_armed = False
        self.window_name = "Polygon Zone Editor"

    def run(self) -> None:
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, self.canvas_size[0], self.canvas_size[1])
        cv2.setMouseCallback(self.window_name, self._on_mouse)

        while True:
            self.canvas_size = window_canvas_size(self.window_name, self.canvas_size)
            cv2.imshow(self.window_name, self._render(self.canvas_size))
            key = cv2.waitKey(20) & 0xFF
            if key in (27, ord("q"), ord("Q")):
                if self._request_quit():
                    break
            if not self.running:
                break
        cv2.destroyWindow(self.window_name)

    @property
    def running(self) -> bool:
        return not getattr(self, "_should_exit", False)

    def _on_mouse(self, event: int, x: int, y: int, flags: int, param: object) -> None:
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        action = toolbar_action_at(self.buttons, x, y)
        if action:
            self._handle_action(action)
            return
        source_point = display_to_source(self.layout, x, y) if self.layout else None
        if source_point is not None:
            self.current_points.append(source_point)
            self.status = f"Current polygon points: {len(self.current_points)}"
            self.unsaved_changes = True
            self.quit_armed = False

    def _finish_polygon(self) -> bool:
        if len(self.current_points) >= 3:
            self.polygons.append(self.current_points.copy())
            self.current_points.clear()
            self.status = f"Finished polygon. Total polygons: {len(self.polygons)}"
            self.unsaved_changes = True
            self.quit_armed = False
            return True
        self.status = "Need at least 3 points to finish polygon."
        return False

    def _handle_action(self, action: str) -> None:
        if action == "undo":
            if self.current_points:
                self.current_points.pop()
                self.status = f"Undid point. Current points: {len(self.current_points)}"
                self.unsaved_changes = True
            else:
                self.status = "No current point to undo."
        elif action == "finish":
            self._finish_polygon()
        elif action == "new":
            if self.current_points:
                self.status = "Finish current polygon before starting another."
            else:
                self.status = "Ready for a new polygon. Click image to add points."
        elif action == "clear":
            if self.current_points:
                self.current_points.clear()
                self.status = "Cleared current unfinished polygon."
                self.unsaved_changes = True
            else:
                self.status = "No current polygon points to clear."
        elif action == "save":
            self._save_zones()
        elif action == "quit":
            self._request_quit()

    def _save_zones(self) -> None:
        skipped_unfinished = False
        if len(self.current_points) >= 3:
            self._finish_polygon()
        elif self.current_points:
            skipped_unfinished = True

        if not self.polygons:
            self.status = "No valid polygon to save."
            return

        save_zones(
            self.zones_path,
            image_width=self.image.shape[1],
            image_height=self.image.shape[0],
            polygons=self.polygons,
        )
        self.unsaved_changes = False
        self.quit_armed = False
        self.status = f"Saved {len(self.polygons)} polygon zone(s) to {self.zones_path}"
        if skipped_unfinished:
            self.status += "; unfinished polygon was not saved."
        print(self.status)

    def _request_quit(self) -> bool:
        if self.unsaved_changes and not self.quit_armed:
            self.quit_armed = True
            self.status = "Unsaved changes. Click Quit again or press Q/Esc again to exit."
            return False
        self._should_exit = True
        return True

    def _render(self, canvas_size: tuple[int, int]):
        canvas = self.image.copy()
        for polygon in self.polygons:
            for idx, point in enumerate(polygon):
                cv2.circle(canvas, point, 4, (0, 255, 255), -1)
                cv2.line(canvas, point, polygon[(idx + 1) % len(polygon)], (0, 255, 255), 2)
        for idx, point in enumerate(self.current_points):
            cv2.circle(canvas, point, 4, (0, 255, 0), -1)
            if idx > 0:
                cv2.line(canvas, self.current_points[idx - 1], point, (0, 255, 0), 2)

        status = f"{self.status} Polygons: {len(self.polygons)} | Current points: {len(self.current_points)}"
        rendered, self.layout, self.buttons = render_interactive_canvas(canvas, ZONE_ACTIONS, status, canvas_size)
        return rendered


def run_zone_editor(image_path: str | Path, zones_path: str | Path) -> None:
    OpenCVZoneEditor(image_path, zones_path).run()
