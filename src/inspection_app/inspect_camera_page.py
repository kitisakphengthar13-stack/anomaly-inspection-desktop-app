from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
import shiboken6
from PySide6.QtCore import QThread, Qt, QTimer, QUrl, Slot
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QFileDialog,
    QGridLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from inspection.result_types import FinalResult, InspectionResult
from inspection_app.formatting import (
    format_anomaly_decision,
    format_duration_ms,
    format_mean_pixel_difference,
    format_pixel_area,
    format_presence_status,
    format_ratio_percent,
    format_score,
)
from inspection_app.inspect_image_page import (
    InspectionWorker,
    inspection_csv_log_path,
    require_saved_config_path,
)
from inspection_app.job_paths import default_camera_output_dir as job_camera_output_dir
from inspection_app.layout_contracts import ActionSection, ControlRail, WorkbenchLayout
from inspection_app.camera_controller import CameraController, CameraControllerState
from inspection_app.reference_capture_page import (
    AspectImageLabel,
    camera_log,
    frame_to_pixmap,
    log_if_not_gui_thread,
)
from inspection_app.runtime import PreparedRuntimeManager
from inspection_app.state import AppState
from inspection_app.theme import page_margins, theme_dimensions, theme_spacing, zero_margins
from inspection_app.ui_locale import apply_app_locale
from inspection_app.ui_components import (
    PathPickerRow,
    ResultImageTabs,
    ResultSummaryPanel,
    SectionPanel,
    StatusBanner,
    set_button_icon,
    set_button_role,
)

StatusCallback = Callable[[str], None]


def default_camera_output_dir(state: AppState) -> Path:
    return state.camera_output_dir or job_camera_output_dir(state.inspection_job_slug)


def captured_image_path(output_dir: str | Path, now: datetime | None = None) -> Path:
    timestamp = (now or datetime.now()).strftime("%Y%m%d_%H%M%S_%f")
    return Path(output_dir) / "captured" / f"capture_{timestamp}.png"


def save_captured_inspection_frame(frame: np.ndarray | None, output_dir: str | Path) -> tuple[Path, tuple[int, int]]:
    if frame is None:
        raise ValueError("No captured frame is available to inspect.")
    path = captured_image_path(output_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), frame):
        raise ValueError(f"Could not save captured image: {path}")
    height, width = frame.shape[:2]
    return path, (width, height)


class _CenteredPreviewContainer(QWidget):
    def __init__(self, preview_label: AspectImageLabel) -> None:
        super().__init__()
        self.setObjectName("cameraPreviewContainer")
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.preview_label = preview_label
        self.preview_label.setParent(self)
        self.preview_label.show()
        self._deferred_geometry_update_pending = False
        self._deferred_geometry_update_timer = QTimer(self)
        self._deferred_geometry_update_timer.setSingleShot(True)
        self._deferred_geometry_update_timer.timeout.connect(self._run_deferred_preview_geometry_update)

    def resizeEvent(self, event) -> None:  # type: ignore[override]
        super().resizeEvent(event)
        self.update_preview_geometry()

    def update_preview_geometry(self) -> None:
        if not log_if_not_gui_thread("_CenteredPreviewContainer.update_preview_geometry"):
            return
        rect = self.contentsRect()
        if rect.width() <= 0 or rect.height() <= 0:
            self._defer_preview_geometry_update()
            return

        aspect_ratio = self._preview_aspect_ratio()
        width = rect.width()
        height = round(width / aspect_ratio)
        if height > rect.height():
            height = rect.height()
            width = round(height * aspect_ratio)

        width = max(1, width)
        height = max(1, height)
        x = rect.x() + (rect.width() - width) // 2
        y = rect.y() + (rect.height() - height) // 2
        self.preview_label.setGeometry(x, y, width, height)
        self.preview_label.update()

    def _defer_preview_geometry_update(self) -> None:
        if not log_if_not_gui_thread("_CenteredPreviewContainer._defer_preview_geometry_update"):
            return
        if self._deferred_geometry_update_pending:
            return
        self._deferred_geometry_update_pending = True
        self._deferred_geometry_update_timer.start(0)

    def _run_deferred_preview_geometry_update(self) -> None:
        if not log_if_not_gui_thread("_CenteredPreviewContainer._run_deferred_preview_geometry_update"):
            return
        self._deferred_geometry_update_pending = False
        self.update_preview_geometry()

    def _preview_aspect_ratio(self) -> float:
        pixmap = self.preview_label._source_pixmap
        if pixmap is not None and not pixmap.isNull() and pixmap.height() > 0:
            return pixmap.width() / pixmap.height()
        return 16 / 9


class InspectCameraPage(QWidget):
    def __init__(
        self,
        state: AppState,
        status_callback: StatusCallback | None = None,
        runtime_manager: PreparedRuntimeManager | None = None,
    ) -> None:
        super().__init__()
        self.state = state
        self._status_callback = status_callback
        self.runtime_manager = runtime_manager
        self._inspection_thread: QThread | None = None
        self._inspection_worker: InspectionWorker | None = None
        self._latest_frame: np.ndarray | None = None
        self._captured_frame: np.ndarray | None = None
        self._captured_image_path: Path | None = None
        self._state = "idle"
        self._logged_first_preview_update = False

        self._build_ui()
        self.camera_controller = CameraController(self._camera_settings)
        self.camera_controller.state_changed.connect(self._on_camera_controller_state_changed, Qt.ConnectionType.QueuedConnection)
        self.camera_controller.frame_ready.connect(self._on_controller_frame_ready, Qt.ConnectionType.QueuedConnection)
        self.camera_controller.resolution_changed.connect(self._on_controller_resolution_changed, Qt.ConnectionType.QueuedConnection)
        self.camera_controller.error.connect(self._on_controller_error, Qt.ConnectionType.QueuedConnection)
        self.camera_controller.stopped.connect(self._on_controller_stopped, Qt.ConnectionType.QueuedConnection)
        apply_app_locale(self)
        self.refresh_from_state()
        self._set_state("idle")
        self._set_feedback("Start camera, capture, inspect.", ok=True)

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        spacing = theme_spacing()
        dimensions = theme_dimensions()
        layout.setContentsMargins(*page_margins())
        layout.setSpacing(spacing.operations_page_gap)
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        self.preview_label = AspectImageLabel()
        self.preview_label.setMinimumHeight(dimensions.camera_preview_idle_min_height)
        self.feedback_label = StatusBanner()

        self.workspace = self._operator_workspace()
        layout.addWidget(self.workspace, 1)

    def _operator_workspace(self) -> WorkbenchLayout:
        splitter = WorkbenchLayout(object_name="cameraOperatorWorkspace", rail_position="left")
        splitter.set_control_rail(self._control_rail())
        splitter.set_main_workspace(self._visual_workspace_group())
        dimensions = theme_dimensions()
        splitter.setSizes([dimensions.camera_splitter_left_width, dimensions.camera_splitter_right_width])
        return splitter

    def _control_rail(self) -> ControlRail:
        rail = ControlRail(object_name="cameraControlPane")
        rail.add_fixed(self._camera_readiness_group())
        rail.add_fixed(self._operation_group())
        rail.set_scroll_body(self._secondary_control_pane(), object_name="cameraControlPaneScroll")
        return rail

    def _secondary_control_pane(self) -> QWidget:
        widget = QWidget()
        widget.setObjectName("cameraSecondaryPane")
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(*zero_margins())
        layout.setSpacing(theme_spacing().left_pane_gap)
        layout.addWidget(self._summary_group())
        layout.addWidget(self._setup_output_group())
        layout.addStretch(1)
        return widget

    def _camera_readiness_group(self) -> SectionPanel:
        panel = SectionPanel("Camera", compact=True)
        grid = QGridLayout()
        spacing = theme_spacing()
        grid.setContentsMargins(*zero_margins())
        grid.setHorizontalSpacing(spacing.compact_grid_horizontal_gap)
        grid.setVerticalSpacing(spacing.compact_grid_vertical_gap)

        self.camera_index_spin = QSpinBox()
        self.camera_index_spin.setRange(0, 32)
        self.camera_index_spin.setValue(0)
        grid.addWidget(QLabel("Camera index"), 0, 0)
        grid.addWidget(self.camera_index_spin, 0, 1)

        self.width_spin = QSpinBox()
        self.width_spin.setRange(0, 10000)
        self.width_spin.setSpecialValueText("Auto")
        self.height_spin = QSpinBox()
        self.height_spin.setRange(0, 10000)
        self.height_spin.setSpecialValueText("Auto")
        grid.addWidget(QLabel("Requested frame size"), 0, 2)
        grid.addWidget(self.width_spin, 0, 3)
        grid.addWidget(QLabel("x"), 0, 4)
        grid.addWidget(self.height_spin, 0, 5)
        grid.setColumnStretch(1, 1)
        grid.setColumnStretch(5, 1)
        panel.content_layout.addLayout(grid)
        return panel

    def _operation_group(self) -> SectionPanel:
        panel = SectionPanel("Operation", compact=True)
        self.operation_state_banner = StatusBanner()
        panel.content_layout.addWidget(self.operation_state_banner)

        self.start_button = QPushButton("Start Camera")
        set_button_icon(self.start_button, "start_camera")
        self.start_button.clicked.connect(self.start_camera)
        self.stop_button = QPushButton("Stop Camera")
        self.stop_button.clicked.connect(self.stop_camera)
        self.capture_button = QPushButton("Capture Part")
        set_button_icon(self.capture_button, "capture")
        self.capture_button.clicked.connect(self.capture_part)
        self.retake_button = QPushButton("Retake")
        self.retake_button.clicked.connect(self.retake)
        self.inspect_button = QPushButton("Inspect Captured Image")
        set_button_icon(self.inspect_button, "run")
        self.inspect_button.clicked.connect(self.inspect_captured_image)
        self.open_output_button = QPushButton("Open Output Folder")
        set_button_icon(self.open_output_button, "open")
        self.open_output_button.clicked.connect(self.open_output_folder)

        for button in (
            self.start_button,
            self.stop_button,
            self.capture_button,
            self.retake_button,
            self.inspect_button,
            self.open_output_button,
        ):
            set_button_role(button, "secondary")
        action_section = ActionSection(rows=2)
        action_section.add_row((self.start_button, self.capture_button, self.inspect_button, self.retake_button))
        action_section.add_row((self.stop_button, self.open_output_button))
        panel.content_layout.addWidget(action_section)
        panel.content_layout.addWidget(self.feedback_label)
        return panel

    def _setup_output_group(self) -> SectionPanel:
        panel = SectionPanel("Config and Output", compact=True)
        grid = QGridLayout()
        spacing = theme_spacing()
        grid.setContentsMargins(*zero_margins())
        grid.setHorizontalSpacing(spacing.compact_grid_horizontal_gap)
        grid.setVerticalSpacing(spacing.compact_grid_vertical_gap)

        self.config_path_edit = QLineEdit()
        self.config_path_edit.setReadOnly(True)
        grid.addWidget(QLabel("Saved config"), 0, 0)
        grid.addWidget(self.config_path_edit, 0, 1)

        self.output_dir_row = PathPickerRow("Choose Output Folder...", self._choose_output_dir)
        self.output_dir_edit = self.output_dir_row.line_edit
        self.output_dir_edit.textEdited.connect(self._mark_output_dir_override)
        grid.addWidget(QLabel("Output folder"), 1, 0)
        grid.addWidget(self.output_dir_row, 1, 1)

        grid.setColumnStretch(1, 1)
        panel.content_layout.addLayout(grid)
        return panel

    def _summary_group(self) -> ResultSummaryPanel:
        self.summary_panel = ResultSummaryPanel()
        self.result_badge = self.summary_panel.result_badge
        return self.summary_panel

    def _visual_workspace_group(self) -> SectionPanel:
        panel = SectionPanel("Camera View", compact=True)
        dimensions = theme_dimensions()
        panel.setMinimumHeight(dimensions.camera_workspace_min_height)
        panel.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        self.visual_stack = QStackedWidget()
        self.visual_stack.setObjectName("cameraVisualStack")
        self.visual_stack.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.preview_label.setMinimumSize(0, 0)
        self.preview_label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        self.preview_container = _CenteredPreviewContainer(self.preview_label)
        self.preview_container.setMinimumHeight(dimensions.camera_visual_preview_min_height)

        self.result_image_tabs = ResultImageTabs(
            min_height=dimensions.camera_result_image_tabs_min_height,
            tabs_min_height=dimensions.camera_result_tabs_min_height,
            preview_min_height=dimensions.camera_result_preview_min_height,
        )
        self.fit_image_button = self.result_image_tabs.fit_image_button
        self.tabs = self.result_image_tabs.tabs
        self.annotated_preview = self.result_image_tabs.annotated_preview
        self.heatmap_preview = self.result_image_tabs.heatmap_preview
        self.mask_preview = self.result_image_tabs.mask_preview
        self.visual_stack.addWidget(self.preview_container)
        self.visual_stack.addWidget(self.result_image_tabs)
        panel.content_layout.addWidget(self.visual_stack, 1)
        return panel

    def refresh_from_state(self) -> None:
        self.config_path_edit.setText("" if self.state.config_path is None else str(self.state.config_path))
        if self.state.camera_output_dir is None:
            self.output_dir_edit.setText(str(default_camera_output_dir(self.state)))
        else:
            self.output_dir_edit.setText(str(self.state.camera_output_dir))
        self._set_state(self._state)

    def refresh_theme(self) -> None:
        self.preview_label.refresh_theme()

    def start_camera(self) -> None:
        self._latest_frame = None
        self._captured_frame = None
        self._captured_image_path = None
        setattr(frame_to_pixmap, "_logged_first", False)
        self._logged_first_preview_update = False
        self.preview_label._logged_first_display = False
        self.preview_label._logged_first_source_set = False
        self.preview_label.clear_preview("Starting camera\nWaiting for the first live frame.")
        self._set_state("live_preview")
        self._set_feedback("Starting camera...", ok=True)
        self.camera_controller.start_preview()

    def stop_camera(self) -> None:
        self._latest_frame = None
        self.camera_controller.stop_preview()
        if self._captured_frame is not None:
            self._set_state("captured_preview")
            self._set_preview_frame(self._captured_frame)
            self._set_feedback("Camera stopped. Captured frame preserved.", ok=True)
            return
        self._set_state("idle")
        self.preview_label.clear_preview("Camera preview stopped.\nStart camera to resume.")
        self._set_feedback("Camera stopped.", ok=True)

    def capture_part(self) -> None:
        if self._latest_frame is None:
            self._set_feedback("No camera frame is available to capture yet.", ok=False)
            return
        self._captured_frame = self._latest_frame.copy()
        self._set_state("captured_preview")
        self._set_preview_frame(self._captured_frame)
        self._set_feedback("Frame captured.", ok=True)

    def retake(self) -> None:
        self._captured_frame = None
        self._captured_image_path = None
        if self.camera_controller.state() in {CameraControllerState.RELEASED.value, CameraControllerState.FAILED.value, CameraControllerState.SOFT_STOPPED.value}:
            self.start_camera()
        elif self.camera_controller.state() in {
            CameraControllerState.OPENING.value,
            CameraControllerState.WARMING.value,
            CameraControllerState.RUNNING.value,
            CameraControllerState.RECOVERING.value,
        }:
            self._set_state("live_preview")
            self._set_feedback("Retaking. Live preview resumed.", ok=True)
        else:
            self._set_feedback("Camera is stopping. Start again after it has stopped.", ok=False)

    def inspect_captured_image(self) -> None:
        try:
            config_path = require_saved_config_path(self.state)
            output_dir = Path(self.output_dir_edit.text().strip())
            if not str(output_dir):
                raise ValueError("Output directory is required.")
            output_dir.mkdir(parents=True, exist_ok=True)
            captured_path, (width, height) = save_captured_inspection_frame(self._captured_frame, output_dir)
        except Exception as exc:
            self._set_feedback(str(exc), ok=False)
            self.operation_state_banner.set_state("Error", str(exc), "error", "error")
            return

        derived_output_dir = job_camera_output_dir(self.state.inspection_job_slug)
        self.state.output_dir = output_dir
        if self.state.camera_output_dir is not None or output_dir != derived_output_dir:
            self.state.camera_output_dir = output_dir
        self._captured_image_path = captured_path
        self._set_state("inspecting")
        self._set_feedback(f"Inspecting {captured_path.name} ({width}x{height}).", ok=True)

        self._inspection_thread = QThread(self)
        self._inspection_worker = InspectionWorker(
            config_path,
            captured_path,
            output_dir,
            inspection_mode="camera",
            runtime_manager=self.runtime_manager,
        )
        self._inspection_worker.moveToThread(self._inspection_thread)
        self._inspection_thread.started.connect(self._inspection_worker.run)
        self._inspection_worker.status.connect(lambda message: self._set_feedback(message, ok=True))
        self._inspection_worker.completed.connect(self._on_inspection_completed)
        self._inspection_worker.failed.connect(self._on_inspection_failed)
        self._inspection_worker.finished.connect(self._inspection_thread.quit)
        self._inspection_worker.finished.connect(self._inspection_worker.deleteLater)
        self._inspection_thread.finished.connect(self._inspection_thread.deleteLater)
        self._inspection_thread.finished.connect(self._clear_inspection_refs)
        self._inspection_thread.start()

    def open_output_folder(self) -> None:
        output_dir = Path(self.output_dir_edit.text().strip() or default_camera_output_dir(self.state))
        if output_dir.exists():
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(output_dir.resolve())))
        else:
            self._set_feedback(f"Output folder does not exist yet: {output_dir}", ok=False)

    def shutdown(self) -> None:
        self.camera_controller.shutdown()
        self._stop_inspection_worker()

    def _choose_output_dir(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Choose output folder", self.output_dir_edit.text().strip())
        if path:
            self.output_dir_edit.setText(path)
            output_path = Path(path)
            self.state.camera_output_dir = output_path
            self.state.output_dir = output_path
            self._set_state(self._state)

    def _mark_output_dir_override(self, value: str) -> None:
        text = value.strip()
        self.state.camera_output_dir = Path(text) if text else None
        self._set_state(self._state)

    def _camera_settings(self) -> tuple[int, int, int]:
        return self.camera_index_spin.value(), self.width_spin.value(), self.height_spin.value()

    @Slot(object)
    def _on_controller_frame_ready(self, frame: object) -> None:
        if not isinstance(frame, np.ndarray):
            return
        self._latest_frame = frame
        if self._state == "result_ready":
            self._set_preview_frame(frame)
            return
        self._set_state("live_preview")
        self._show_camera_visual()
        self._set_preview_frame(frame)
        self._update_operation_state_banner(self._state)

    @Slot(int, int)
    def _on_controller_resolution_changed(self, width: int, height: int) -> None:
        requested_width = self.width_spin.value()
        requested_height = self.height_spin.value()
        if requested_width and requested_height and (width != requested_width or height != requested_height):
            self._set_feedback(
                f"Camera started. Requested {requested_width}x{requested_height}; actual frame size is {width}x{height}.",
                ok=True,
            )
        else:
            self._set_feedback(f"Camera started. Actual frame size is {width}x{height}.", ok=True)

    @Slot(str, str, str)
    def _on_camera_controller_state_changed(self, state: str, message: str, level: str) -> None:
        if state == CameraControllerState.SOFT_STOPPED.value:
            self._latest_frame = None
            if self._captured_frame is not None:
                self._set_state("captured_preview")
                self._set_preview_frame(self._captured_frame)
            else:
                self._set_state("idle")
                self.preview_label.clear_preview("Camera preview stopped.\nStart camera to resume.")
            self._set_feedback(message, ok=True)
            return
        if state in {CameraControllerState.OPENING.value, CameraControllerState.WARMING.value, CameraControllerState.RECOVERING.value}:
            self._set_state("live_preview")
        elif state in {CameraControllerState.RELEASED.value, CameraControllerState.FAILED.value}:
            self._set_state("idle")
        elif state == CameraControllerState.RUNNING.value:
            self._set_state("live_preview")
        banner_level = "warning" if level == "warning" else "info"
        title = "Reopening camera" if state == CameraControllerState.RECOVERING.value else "Camera"
        state = "blocked" if level == "warning" else "processing"
        self.operation_state_banner.set_state(title, message, state, banner_level)
        self._set_feedback(message, ok=level != "error")

    @Slot(str)
    def _on_controller_error(self, message: str) -> None:
        self._set_feedback(message, ok=False)
        self._set_state("idle")
        self.operation_state_banner.set_state("Error", message, "error", "error")

    @Slot()
    def _on_controller_stopped(self) -> None:
        if self._state == "live_preview":
            self._set_state("idle")

    def _on_inspection_completed(self, result: object, output_dir: object) -> None:
        if not isinstance(result, InspectionResult):
            self._set_feedback("Inspection failed: worker returned an invalid result.", ok=False)
            self._set_state("captured_preview")
            self.operation_state_banner.set_state("Error", "Worker returned an invalid inspection result.", "error", "error")
            return
        output_path = Path(output_dir)
        self._update_summary(result, output_path)
        self._update_previews(result)
        ok = result.final_result != FinalResult.ERROR
        self._set_state("result_ready")
        self._set_feedback(f"Camera inspection complete: {result.final_result.value}", ok=ok)

    def _on_inspection_failed(self, message: str) -> None:
        self._set_state("captured_preview")
        self._set_feedback(f"Inspection failed: {message}", ok=False)
        self.operation_state_banner.set_state("Error", f"Inspection failed: {message}", "error", "error")

    def _update_summary(self, result: InspectionResult, output_dir: Path) -> None:
        final = result.final_result.value
        csv_path = inspection_csv_log_path(result, output_dir)
        metrics = (
            ("Presence", format_presence_status(result.presence_status)),
            ("Anomaly decision", format_anomaly_decision(result.anomaly_pred_label)),
            ("Score", format_score(result.anomaly_score)),
            ("Inspection time", format_duration_ms(result.total_time_ms)),
            ("Foreground", format_ratio_percent(result.foreground_ratio)),
            ("Blob area", format_pixel_area(result.largest_blob_area)),
            ("Mean pixel difference", format_mean_pixel_difference(result.mean_diff)),
            ("Backend", result.anomaly_backend or ""),
            ("Presence check time", format_duration_ms(result.presence_time_ms)),
            ("Anomaly inference time", format_duration_ms(result.anomaly_time_ms)),
        )
        self.summary_panel.set_result(
            final,
            metrics,
            csv_log_path="" if csv_path is None else str(csv_path),
            error_message=result.error_message or "",
        )

    def _update_previews(self, result: InspectionResult) -> None:
        self.result_image_tabs.set_artifacts(
            annotated_path=result.annotated_image_path,
            heatmap_path=result.heatmap_path,
            presence_mask_path=result.presence_mask_path,
        )

    def _fit_active_preview(self) -> None:
        self.result_image_tabs.fit_active_preview()

    def _refit_active_preview_if_needed(self) -> None:
        self.result_image_tabs.refit_active_preview_if_needed()

    def _stop_inspection_worker(self) -> None:
        thread = self._inspection_thread
        self._inspection_thread = None
        self._inspection_worker = None
        if thread is None:
            return
        try:
            if shiboken6.isValid(thread):
                thread.quit()
                thread.wait()
        except RuntimeError:
            return

    def _clear_inspection_refs(self) -> None:
        self._inspection_thread = None
        self._inspection_worker = None

    def _set_state(self, state: str) -> None:
        self._state = state
        idle = state == "idle"
        live = state == "live_preview"
        captured = state == "captured_preview"
        inspecting = state == "inspecting"
        result_ready = state == "result_ready"
        controller_state = (
            self.camera_controller.state()
            if hasattr(self, "camera_controller")
            else CameraControllerState.RELEASED.value
        )
        camera_can_start = controller_state in {
            CameraControllerState.RELEASED.value,
            CameraControllerState.FAILED.value,
            CameraControllerState.SOFT_STOPPED.value,
        }
        camera_active = controller_state in {
            CameraControllerState.OPENING.value,
            CameraControllerState.WARMING.value,
            CameraControllerState.RUNNING.value,
            CameraControllerState.RECOVERING.value,
            CameraControllerState.SOFT_STOPPED.value,
        }
        camera_stopping = controller_state == CameraControllerState.STOPPING.value
        camera_ready = controller_state == CameraControllerState.RUNNING.value and self._latest_frame is not None

        self.start_button.setEnabled((idle or result_ready) and camera_can_start and not inspecting)
        self.stop_button.setEnabled((live or captured or (result_ready and camera_active)) and camera_active and not camera_stopping)
        self.capture_button.setEnabled(live and camera_ready)
        self.retake_button.setEnabled(captured or result_ready)
        self.inspect_button.setEnabled(captured and self.state.config_path is not None)
        self.open_output_button.setEnabled(not inspecting)
        for widget in (self.camera_index_spin, self.width_spin, self.height_spin):
            widget.setEnabled(idle and controller_state in {CameraControllerState.RELEASED.value, CameraControllerState.FAILED.value})
        self.output_dir_edit.setEnabled(not inspecting)
        if result_ready:
            self._show_result_visual()
        else:
            self._show_camera_visual()
        self._apply_button_roles(state)
        self._update_operation_state_banner(state)

    def _show_camera_visual(self) -> None:
        if hasattr(self, "visual_stack"):
            self.visual_stack.setCurrentWidget(self.preview_container)
            self.preview_container.update_preview_geometry()
            self.preview_label.update()

    def _show_result_visual(self) -> None:
        if hasattr(self, "visual_stack"):
            self.visual_stack.setCurrentWidget(self.result_image_tabs)

    def _set_preview_frame(self, frame: np.ndarray) -> None:
        pixmap = frame_to_pixmap(frame)
        if not self._logged_first_preview_update:
            camera_log(f"first inspect-camera preview update; pixmap={pixmap.width()}x{pixmap.height()}")
            self._logged_first_preview_update = True
        self.preview_label.set_frame_pixmap(pixmap)
        self.preview_container.update_preview_geometry()
        self.preview_container._defer_preview_geometry_update()

    def _apply_button_roles(self, state: str) -> None:
        for button in (
            self.start_button,
            self.stop_button,
            self.capture_button,
            self.retake_button,
            self.inspect_button,
            self.open_output_button,
        ):
            set_button_role(button, "secondary")
        if state == "idle":
            set_button_role(self.start_button, "primary")
        elif state == "live_preview":
            set_button_role(self.capture_button, "primary")
        elif state == "captured_preview":
            set_button_role(self.inspect_button, "primary")
        elif state == "result_ready":
            set_button_role(self.retake_button, "primary")

    def _update_operation_state_banner(self, state: str) -> None:
        if state == "idle":
            if self.state.config_path is None:
                self.operation_state_banner.set_state(
                    "Blocked",
                    "Open or save a config in Project Setup before camera inspection.",
                    "blocked",
                    "warning",
                )
            elif not self.output_dir_edit.text().strip():
                self.operation_state_banner.set_state("Blocked", "Choose an output folder before camera inspection.", "blocked", "warning")
            else:
                self.operation_state_banner.set_state("Ready", "Start camera to begin live preview.", "ready", "success")
        elif state == "live_preview":
            if self._latest_frame is None:
                self.operation_state_banner.set_state("Starting camera", "Waiting for first frame.", "processing", "info")
            else:
                self.operation_state_banner.set_state("Camera live", "Capture when the part is positioned.", "live", "info")
        elif state == "captured_preview":
            if self.state.config_path is None:
                self.operation_state_banner.set_state(
                    "Frame captured",
                    "Save a config in Project Setup before inspection.",
                    "captured",
                    "warning",
                )
            else:
                self.operation_state_banner.set_state("Ready to inspect", "Inspect the captured frame or retake it.", "captured", "success")
        elif state == "inspecting":
            self.operation_state_banner.set_state("Inspecting", "Running inspection.", "processing", "info")
        elif state == "result_ready":
            self.operation_state_banner.set_state("Complete", "Result available. Retake to inspect another frame.", "complete", "success")
        else:
            self.operation_state_banner.set_state("Error", "Camera workflow state is unavailable.", "error", "error")

    def _set_feedback(self, message: str, ok: bool) -> None:
        self.feedback_label.set_message(message, "success" if ok else "error")
        self.state.status_message = message
        if self._status_callback:
            self._status_callback(message)
