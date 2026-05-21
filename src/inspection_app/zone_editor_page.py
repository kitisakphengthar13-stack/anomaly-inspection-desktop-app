from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from PySide6.QtCore import QPointF, Qt, Signal
from PySide6.QtGui import QBrush, QColor, QFont, QPen, QPixmap, QPolygonF
from PySide6.QtWidgets import (
    QFileDialog,
    QGraphicsEllipseItem,
    QGraphicsLineItem,
    QGraphicsPixmapItem,
    QGraphicsPolygonItem,
    QGraphicsScene,
    QGraphicsTextItem,
    QGraphicsView,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSizePolicy,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from inspection.zone_io import save_zones
from inspection_app.job_paths import (
    default_reference_image_path as job_reference_image_path,
    default_zones_path as job_zones_path,
)
from inspection_app.state import AppState
from inspection_app.theme import (
    page_margins,
    preview_surface_stylesheet,
    theme_dimensions,
    theme_spacing,
    zero_margins,
    zone_overlay_palette,
)
from inspection_app.ui_components import ActionButtonRow, ScrollablePane, SectionPanel, StatusBanner, set_button_icon, set_button_role

Point = tuple[int, int]
StatusCallback = Callable[[str], None]
ZonesSavedCallback = Callable[[Path], None]


@dataclass(frozen=True)
class ZoneCanvasStyle:
    finalized_pen_width: int
    current_pen_width: int
    point_diameter: int
    point_pen_width: int
    label_font_px: int


def _clamp_int(value: float, minimum: int, maximum: int) -> int:
    return max(minimum, min(maximum, int(round(value))))


def compute_zone_canvas_style(width: int, height: int) -> ZoneCanvasStyle:
    min_dim = max(1, min(int(width), int(height)))
    finalized_pen_width = _clamp_int(min_dim / 360.0, 2, 5)
    return ZoneCanvasStyle(
        finalized_pen_width=finalized_pen_width,
        current_pen_width=finalized_pen_width,
        point_diameter=_clamp_int(min_dim / 90.0, 7, 18),
        point_pen_width=_clamp_int(min_dim / 520.0, 1, 3),
        label_font_px=_clamp_int(min_dim / 45.0, 12, 32),
    )


def default_zones_output_path(state: AppState) -> Path:
    if state.zones_path is not None:
        return state.zones_path
    return job_zones_path(state.inspection_job_slug)


def normalize_zones_output_path(raw_path: str | Path) -> Path:
    path = Path(str(raw_path).strip())
    if not str(path):
        raise ValueError("Zones JSON output path is required.")
    if not path.suffix:
        path = path.with_suffix(".json")
    if path.suffix.lower() != ".json":
        raise ValueError(f"Unsupported zones file extension: {path.suffix}")
    return path


def prepare_polygons_for_save(
    finalized_polygons: list[list[Point]],
    current_points: list[Point],
) -> tuple[list[list[Point]], str | None]:
    polygons = [polygon.copy() for polygon in finalized_polygons]
    warning = None
    if len(current_points) >= 3:
        polygons.append(current_points.copy())
    elif current_points:
        warning = "Unfinished polygon has fewer than 3 points and was not saved."
    if not polygons:
        raise ValueError("No valid polygon to save.")
    return polygons, warning


def save_zone_session(
    zones_path: str | Path,
    image_width: int,
    image_height: int,
    finalized_polygons: list[list[Point]],
    current_points: list[Point],
) -> tuple[Path, int, str | None]:
    polygons, warning = prepare_polygons_for_save(finalized_polygons, current_points)
    path = normalize_zones_output_path(zones_path)
    save_zones(path, image_width=image_width, image_height=image_height, polygons=polygons)
    return path, len(polygons), warning


class ZoneGraphicsView(QGraphicsView):
    image_clicked = Signal(int, int)

    def __init__(self) -> None:
        super().__init__()
        self.image_width = 0
        self.image_height = 0
        self.setRenderHints(self.renderHints())
        self.setDragMode(QGraphicsView.DragMode.NoDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
        self.setStyleSheet(preview_surface_stylesheet())
        self.setMinimumHeight(theme_dimensions().zone_canvas_min_height)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    def refresh_theme(self) -> None:
        self.setStyleSheet(preview_surface_stylesheet())
        self.viewport().update()

    def set_image_size(self, width: int, height: int) -> None:
        self.image_width = width
        self.image_height = height
        self.setSceneRect(0, 0, width, height)
        self.fit_image()

    def fit_image(self) -> None:
        if self.image_width > 0 and self.image_height > 0:
            self.fitInView(0, 0, self.image_width, self.image_height, Qt.AspectRatioMode.KeepAspectRatio)

    def mousePressEvent(self, event) -> None:  # type: ignore[override]
        if event.button() == Qt.MouseButton.LeftButton and self.image_width > 0 and self.image_height > 0:
            scene_pos = self.mapToScene(event.position().toPoint())
            x = int(round(scene_pos.x()))
            y = int(round(scene_pos.y()))
            if 0 <= x < self.image_width and 0 <= y < self.image_height:
                self.image_clicked.emit(x, y)
                return
        super().mousePressEvent(event)

    def resizeEvent(self, event) -> None:  # type: ignore[override]
        super().resizeEvent(event)
        self.fit_image()


class ZoneEditorPage(QWidget):
    def __init__(
        self,
        state: AppState,
        status_callback: StatusCallback | None = None,
        zones_saved_callback: ZonesSavedCallback | None = None,
    ) -> None:
        super().__init__()
        self.state = state
        self._status_callback = status_callback
        self._zones_saved_callback = zones_saved_callback
        self._scene = QGraphicsScene(self)
        self._pixmap_item: QGraphicsPixmapItem | None = None
        self._overlay_items: list[object] = []
        self._image_width = 0
        self._image_height = 0
        self._finalized_polygons: list[list[Point]] = []
        self._current_points: list[Point] = []
        self._zone_canvas_style = compute_zone_canvas_style(0, 0)

        self._build_ui()
        self.refresh_from_state()
        self.reference_path_edit.textChanged.connect(self._sync_reference_path_state)
        self.zones_path_edit.textChanged.connect(self._sync_zones_path_state)
        self._set_feedback("Load reference, draw zones, save.", ok=True)

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        spacing = theme_spacing()
        layout.setContentsMargins(*page_margins())
        layout.setSpacing(spacing.setup_page_gap)
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        self.view = ZoneGraphicsView()
        self.view.setScene(self._scene)
        self.view.image_clicked.connect(self._add_point)

        self.feedback_label = StatusBanner()

        layout.addWidget(self._zone_workspace(), 1)

    def _zone_workspace(self) -> QSplitter:
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setObjectName("zoneEditorWorkspace")
        splitter.setChildrenCollapsible(False)
        splitter.addWidget(self._control_pane_scroll())
        splitter.addWidget(self._canvas_group())
        dimensions = theme_dimensions()
        splitter.setSizes([dimensions.camera_splitter_left_width, dimensions.camera_splitter_right_width])
        splitter.setStretchFactor(0, dimensions.camera_splitter_left_stretch)
        splitter.setStretchFactor(1, dimensions.camera_splitter_right_stretch)
        return splitter

    def _control_pane_scroll(self) -> ScrollablePane:
        return ScrollablePane(self._control_pane(), object_name="zoneEditorControlPaneScroll")

    def _control_pane(self) -> QWidget:
        widget = QWidget()
        widget.setObjectName("zoneEditorControlPane")
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(*zero_margins())
        layout.setSpacing(theme_spacing().left_pane_gap)
        layout.addWidget(self._paths_group())
        layout.addWidget(self._actions_group())
        layout.addStretch(1)
        return widget

    def _canvas_group(self) -> SectionPanel:
        panel = SectionPanel("Zone Canvas", compact=True)
        panel.setMinimumHeight(theme_dimensions().zone_canvas_min_height)
        panel.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        panel.content_layout.addWidget(self.view, 1)
        return panel

    def _paths_group(self) -> SectionPanel:
        group = SectionPanel("Reference", compact=True)
        layout = group.content_layout

        self.reference_path_edit = QLineEdit()
        self.zones_path_edit = QLineEdit()
        layout.addLayout(self._path_row(self.reference_path_edit, "Browse Reference...", self._browse_reference))
        layout.addLayout(self._path_row(self.zones_path_edit, "Choose Zones Path...", self._choose_zones_path))

        button_row = QHBoxLayout()
        self.load_button = QPushButton("Load Reference Image")
        set_button_icon(self.load_button, "load")
        self.load_button.clicked.connect(self.load_reference_image)
        self.reload_button = QPushButton("Reload Image")
        self.reload_button.clicked.connect(self.load_reference_image)
        self.fit_button = QPushButton("Fit Image")
        set_button_icon(self.fit_button, "fit")
        self.fit_button.clicked.connect(self.view.fit_image)
        for button in (self.load_button, self.reload_button, self.fit_button):
            set_button_role(button, "secondary")
        set_button_role(self.load_button, "primary")
        button_row.addWidget(self.load_button)
        button_row.addWidget(self.reload_button)
        button_row.addWidget(self.fit_button)
        button_row.addStretch(1)
        layout.addLayout(button_row)
        return group

    def _path_row(self, line_edit: QLineEdit, button_text: str, callback: Callable[[], None]) -> QHBoxLayout:
        row = QHBoxLayout()
        row.addWidget(line_edit, 1)
        button = QPushButton(button_text)
        button.clicked.connect(callback)
        row.addWidget(button)
        return row

    def _actions_group(self) -> SectionPanel:
        panel = SectionPanel("Zone Tools", compact=True)
        self.zone_state_banner = StatusBanner()
        panel.content_layout.addWidget(self.zone_state_banner)

        actions = (
            ("Undo Point", self.undo_point),
            ("Finish Polygon", self.finish_polygon),
            ("New Polygon", self.new_polygon),
            ("Clear Current", self.clear_current),
            ("Clear All Zones", self.clear_all_zones),
            ("Save Zones", self.save_zones),
        )
        buttons = []
        for label, callback in actions:
            button = QPushButton(label)
            button.clicked.connect(callback)
            set_button_role(button, "primary" if label == "Save Zones" else "secondary")
            buttons.append(button)
        panel.content_layout.addWidget(ActionButtonRow(buttons[:3]))
        panel.content_layout.addWidget(ActionButtonRow(buttons[3:]))
        panel.content_layout.addWidget(self.feedback_label)
        return panel

    def refresh_from_state(self) -> None:
        if self.state.reference_image_path is not None:
            self.reference_path_edit.setText(str(self.state.reference_image_path))
        if self.state.zones_path is not None:
            self.zones_path_edit.setText(str(self.state.zones_path))
        elif not self.zones_path_edit.text().strip():
            default_path = default_zones_output_path(self.state)
            self.zones_path_edit.setText(str(default_path))
            self.state.zones_path = default_path
            self.state.zones_path_auto = True

    def refresh_theme(self) -> None:
        self.view.refresh_theme()
        self._redraw_overlays()

    def load_reference_image(self) -> None:
        raw_path = self.reference_path_edit.text().strip()
        if not raw_path:
            self._set_feedback("Reference image path is required.", ok=False)
            return
        path = Path(raw_path)
        if not path.exists():
            self._set_feedback(f"Reference image does not exist: {path}", ok=False)
            return
        pixmap = QPixmap(str(path))
        if pixmap.isNull():
            self._set_feedback(f"Could not read reference image: {path}", ok=False)
            return

        self._scene.clear()
        self._overlay_items.clear()
        self._pixmap_item = self._scene.addPixmap(pixmap)
        self._pixmap_item.setZValue(0)
        self._image_width = pixmap.width()
        self._image_height = pixmap.height()
        self._zone_canvas_style = compute_zone_canvas_style(self._image_width, self._image_height)
        self.view.set_image_size(self._image_width, self._image_height)
        self._finalized_polygons.clear()
        self._current_points.clear()
        self.state.reference_image_path = path
        self.state.reference_path_auto = path == job_reference_image_path(self.state.inspection_job_slug)
        self._redraw_overlays()
        self._update_zone_state_banner()
        self._set_feedback(
            f"Loaded reference image {path} ({self._image_width}x{self._image_height}). Fresh drawing session started.",
            ok=True,
        )

    def undo_point(self) -> None:
        if not self._current_points:
            self._set_feedback("No current point to undo.", ok=False)
            return
        self._current_points.pop()
        self._redraw_overlays()
        self._update_zone_state_banner()
        self._set_feedback(f"Current polygon points: {len(self._current_points)}", ok=True)

    def finish_polygon(self) -> None:
        if len(self._current_points) < 3:
            self._set_feedback("A polygon needs at least 3 points.", ok=False)
            return
        self._finalized_polygons.append(self._current_points.copy())
        self._current_points.clear()
        self._redraw_overlays()
        self._update_zone_state_banner()
        self._set_feedback(f"Finished polygon. Total polygons: {len(self._finalized_polygons)}", ok=True)

    def new_polygon(self) -> None:
        if self._current_points:
            self._set_feedback("Finish or clear the current polygon before starting a new one.", ok=False)
            return
        self._set_feedback("Ready for a new polygon. Click the image to add points.", ok=True)

    def clear_current(self) -> None:
        if not self._current_points:
            self._set_feedback("No current polygon points to clear.", ok=True)
            return
        self._current_points.clear()
        self._redraw_overlays()
        self._update_zone_state_banner()
        self._set_feedback("Cleared current unfinished polygon.", ok=True)

    def clear_all_zones(self) -> None:
        self._finalized_polygons.clear()
        self._current_points.clear()
        self._redraw_overlays()
        self._update_zone_state_banner()
        self._set_feedback("Cleared all zones in the current drawing session.", ok=True)

    def save_zones(self) -> None:
        if self._image_width <= 0 or self._image_height <= 0:
            self._set_feedback("Load a reference image before saving zones.", ok=False)
            return
        raw_path = self.zones_path_edit.text().strip()
        if not raw_path:
            self._set_feedback("Zones JSON output path is required.", ok=False)
            return
        try:
            target_path = normalize_zones_output_path(raw_path)
            saved_path, count, warning = save_zone_session(
                target_path,
                self._image_width,
                self._image_height,
                self._finalized_polygons,
                self._current_points,
            )
            if len(self._current_points) >= 3:
                self._finalized_polygons.append(self._current_points.copy())
                self._current_points.clear()
            elif self._current_points:
                self._current_points.clear()
            self.zones_path_edit.setText(str(saved_path))
            self.state.zones_path = saved_path
            self.state.zones_path_auto = saved_path == job_zones_path(self.state.inspection_job_slug)
            if self._zones_saved_callback:
                self._zones_saved_callback(saved_path)
            self._redraw_overlays()
            self._update_zone_state_banner()
            message = f"Saved {count} polygon zone(s) to {saved_path}"
            if warning:
                message += f" {warning}"
            self._set_feedback(message, ok=True)
        except Exception as exc:
            self._set_feedback(str(exc), ok=False)

    def _sync_reference_path_state(self, value: str) -> None:
        text = value.strip()
        self.state.reference_image_path = Path(text) if text else None
        self.state.reference_path_auto = Path(text) == job_reference_image_path(self.state.inspection_job_slug) if text else True

    def _sync_zones_path_state(self, value: str) -> None:
        text = value.strip()
        self.state.zones_path = Path(text) if text else None
        self.state.zones_path_auto = Path(text) == job_zones_path(self.state.inspection_job_slug) if text else True

    def _browse_reference(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Choose reference image",
            self.reference_path_edit.text().strip(),
            "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff);;All files (*)",
        )
        if path:
            self.reference_path_edit.setText(path)

    def _choose_zones_path(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Choose zones JSON path",
            self.zones_path_edit.text().strip() or str(default_zones_output_path(self.state)),
            "JSON files (*.json);;All files (*)",
        )
        if path:
            target = Path(path)
            if not target.suffix:
                target = target.with_suffix(".json")
            self.zones_path_edit.setText(str(target))

    def _add_point(self, x: int, y: int) -> None:
        if self._image_width <= 0 or self._image_height <= 0:
            self._set_feedback("Load a reference image before drawing zones.", ok=False)
            return
        self._current_points.append((x, y))
        self._redraw_overlays()
        self._update_zone_state_banner()
        self._set_feedback(f"Current polygon points: {len(self._current_points)}", ok=True)

    def _redraw_overlays(self) -> None:
        for item in self._overlay_items:
            self._scene.removeItem(item)  # type: ignore[arg-type]
        self._overlay_items.clear()

        style = self._zone_canvas_style
        palette = zone_overlay_palette()
        finalized_color = QColor(*palette.finalized)
        current_color = QColor(*palette.current)
        finalized_pen = QPen(finalized_color, style.finalized_pen_width)
        finalized_brush = QBrush(QColor(*palette.finalized_fill))
        current_pen = QPen(current_color, style.current_pen_width)
        point_brush = QBrush(current_color)
        label_font = QFont()
        label_font.setPixelSize(style.label_font_px)

        for index, polygon in enumerate(self._finalized_polygons, start=1):
            points = [QPointF(x, y) for x, y in polygon]
            item = QGraphicsPolygonItem()
            item.setPolygon(QPolygonF(points))
            item.setPen(finalized_pen)
            item.setBrush(finalized_brush)
            item.setZValue(10)
            self._scene.addItem(item)
            self._overlay_items.append(item)
            label = QGraphicsTextItem(f"Zone {index}")
            label.setDefaultTextColor(QColor(*palette.finalized_label))
            label.setFont(label_font)
            label.setPos(points[0])
            label.setZValue(12)
            self._scene.addItem(label)
            self._overlay_items.append(label)
            for point in points:
                self._add_point_marker(point.x(), point.y(), finalized_color)

        for idx, (x, y) in enumerate(self._current_points):
            self._add_point_marker(x, y, current_color, point_brush)
            if idx > 0:
                prev_x, prev_y = self._current_points[idx - 1]
                line = QGraphicsLineItem(prev_x, prev_y, x, y)
                line.setPen(current_pen)
                line.setZValue(11)
                self._scene.addItem(line)
                self._overlay_items.append(line)

    def _add_point_marker(self, x: float, y: float, color: QColor, brush: QBrush | None = None) -> None:
        style = self._zone_canvas_style
        size = style.point_diameter
        item = QGraphicsEllipseItem(x - size / 2, y - size / 2, size, size)
        item.setPen(QPen(color, style.point_pen_width))
        item.setBrush(brush or QBrush(color))
        item.setZValue(12)
        self._scene.addItem(item)
        self._overlay_items.append(item)

    def _set_feedback(self, message: str, ok: bool) -> None:
        self._update_zone_state_banner()
        self.feedback_label.set_message(message, "success" if ok else "error")
        self.state.status_message = message
        if self._status_callback:
            self._status_callback(message)

    def _update_zone_state_banner(self) -> None:
        if not hasattr(self, "zone_state_banner"):
            return
        if self._image_width <= 0 or self._image_height <= 0:
            self.zone_state_banner.set_state("Reference not loaded", "Load a reference image to begin drawing.", "blocked", "warning")
            return
        if self._current_points:
            self.zone_state_banner.set_state(
                "Polygon in progress",
                f"{len(self._current_points)} point(s). Finish at 3 or more points.",
                "live",
                "info",
            )
            return
        if self._finalized_polygons:
            self.zone_state_banner.set_state(
                "Zones ready",
                f"{len(self._finalized_polygons)} finalized zone(s). Save when complete.",
                "ready",
                "success",
            )
            return
        self.zone_state_banner.set_state("Reference loaded", "Click the canvas to add polygon points.", "ready", "info")
