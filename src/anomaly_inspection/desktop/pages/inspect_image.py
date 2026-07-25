from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from threading import Event
from typing import Callable

import shiboken6
from PySide6.QtCore import QObject, QThread, Qt, QUrl, Signal, Slot
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QFileDialog,
    QGridLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from anomaly_inspection.core.config import InspectionConfig, load_config
from anomaly_inspection.core.pipeline import InspectionPipeline
from anomaly_inspection.core.result_types import FinalResult, InspectionResult
from anomaly_inspection.desktop.ui.formatting import (
    format_anomaly_decision,
    format_duration_ms,
    format_mean_pixel_difference,
    format_pixel_area,
    format_presence_status,
    format_ratio_percent,
    format_score,
)
from anomaly_inspection.desktop.job_paths import default_image_output_dir
from anomaly_inspection.desktop.ui.layout_contracts import ActionSection, ControlRail, WorkbenchLayout
from anomaly_inspection.desktop.runtime import PreparedRuntimeManager
from anomaly_inspection.desktop.state import AppState
from anomaly_inspection.desktop.ui.theme import page_margins, theme_dimensions, theme_spacing, zero_margins
from anomaly_inspection.desktop.ui.locale import apply_app_locale
from anomaly_inspection.desktop.ui.components import (
    ActionButtonRow,
    PathPickerRow,
    ResultImageTabs,
    ResultSummaryPanel,
    SectionPanel,
    StatusBanner,
    set_button_icon,
    set_button_role,
)

StatusCallback = Callable[[str], None]

IMAGE_FILTER = "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff);;All files (*)"
def default_inspect_output_dir(state: AppState) -> Path:
    return state.image_output_dir or default_image_output_dir(state.inspection_job_slug)


def require_saved_config_path(state: AppState) -> Path:
    if state.config_path is None:
        raise ValueError("Open or save a config in Project Setup before inspection.")
    path = Path(state.config_path)
    if not path.exists():
        raise ValueError(f"Selected config file does not exist: {path}")
    return path


def config_for_gui_inspection(config: InspectionConfig) -> InspectionConfig:
    return replace(config, output=replace(config.output, show_images=False))


def inspection_csv_log_path(result: InspectionResult, output_dir: str | Path) -> Path | None:
    path = Path(output_dir) / "inspection_log.csv"
    return path if path.exists() else None


class InspectionWorker(QObject):
    completed = Signal(object, object)
    failed = Signal(str)
    cancelled = Signal(str)
    status = Signal(str)
    finished = Signal()

    def __init__(
        self,
        config_path: Path,
        image_path: Path,
        output_dir: Path,
        inspection_mode: str = "image",
        runtime_manager: PreparedRuntimeManager | None = None,
    ) -> None:
        super().__init__()
        self.config_path = config_path
        self.image_path = image_path
        self.output_dir = output_dir
        self.inspection_mode = inspection_mode
        self.runtime_manager = runtime_manager
        self._cancel_requested = Event()

    def request_cancel(self) -> None:
        self._cancel_requested.set()

    def is_cancel_requested(self) -> bool:
        return self._cancel_requested.is_set()

    @Slot()
    def run(self) -> None:
        try:
            if self.is_cancel_requested():
                self.cancelled.emit("Inspection cancelled before it started.")
                return
            self.output_dir.mkdir(parents=True, exist_ok=True)
            if self.runtime_manager is not None:
                self.status.emit("Inspecting image with prepared runtime..." if self.runtime_manager.is_current(self.config_path) else "Loading runtime on demand...")
                result, used_prepared = self.runtime_manager.inspect_image(
                    self.config_path,
                    self.image_path,
                    self.output_dir,
                    inspection_mode=self.inspection_mode,
                )
                if used_prepared:
                    self.status.emit("Used prepared runtime.")
            else:
                self.status.emit("Loading config and inspection pipeline...")
                config = config_for_gui_inspection(load_config(self.config_path))
                pipeline = InspectionPipeline(config)
                self.status.emit("Inspecting image...")
                result = pipeline.inspect_image(self.image_path, self.output_dir, inspection_mode=self.inspection_mode)
            if self.is_cancel_requested():
                self.cancelled.emit("Inspection cancelled. The completed backend result was discarded.")
            else:
                self.completed.emit(result, self.output_dir)
        except Exception as exc:
            if self.is_cancel_requested():
                self.cancelled.emit("Inspection cancelled.")
            else:
                self.failed.emit(str(exc))
        finally:
            self.finished.emit()


class InspectImagePage(QWidget):
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
        self._thread: QThread | None = None
        self._worker: InspectionWorker | None = None
        self._last_output_dir: Path | None = None
        self._operation_state_key = "waiting"

        self._build_ui()
        apply_app_locale(self)
        self.refresh_from_state()
        self._set_feedback("Choose an image, then run inspection.", ok=True)

    def _build_ui(self) -> None:
        self._main_layout = QVBoxLayout(self)
        spacing = theme_spacing()
        self._main_layout.setContentsMargins(*page_margins())
        self._main_layout.setSpacing(spacing.operations_page_gap)
        self._main_layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        self.feedback_label = StatusBanner()

        self.workspace = self._workspace()

        self._main_layout.addWidget(self.workspace, 1)

    def _workspace(self) -> WorkbenchLayout:
        splitter = WorkbenchLayout(object_name="inspectImageWorkspace", rail_position="left")
        splitter.set_control_rail(self._control_rail())
        splitter.set_main_workspace(self._result_images_group())
        dimensions = theme_dimensions()
        splitter.setSizes([dimensions.inspect_splitter_left_width, dimensions.inspect_splitter_right_width])
        return splitter

    def _control_rail(self) -> ControlRail:
        rail = ControlRail(object_name="inspectImageLeftPane")
        rail.add_fixed(self._inputs_group())
        rail.add_fixed(self._action_group())
        rail.set_scroll_body(self._secondary_control_pane(), object_name="inspectImageLeftPaneScroll")
        return rail

    def _secondary_control_pane(self) -> QWidget:
        widget = QWidget()
        widget.setObjectName("inspectImageSecondaryPane")
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(*zero_margins())
        layout.setSpacing(theme_spacing().left_pane_gap)
        layout.addWidget(self._summary_group())
        layout.addWidget(self._setup_output_group())
        layout.addStretch(1)
        return widget

    def _inputs_group(self) -> SectionPanel:
        panel = SectionPanel("Input Image", compact=True, classic=True)
        self.inputs_panel = panel
        grid = QGridLayout()
        spacing = theme_spacing()
        grid.setContentsMargins(*zero_margins())
        grid.setHorizontalSpacing(spacing.compact_grid_horizontal_gap)
        grid.setVerticalSpacing(spacing.compact_grid_vertical_gap)

        self.image_path_row = PathPickerRow("Browse Image...", self._browse_image)
        self.image_path_edit = self.image_path_row.line_edit
        self.image_path_edit.textChanged.connect(lambda _text: self._refresh_operation_state())
        image_label = QLabel("&Image")
        image_label.setBuddy(self.image_path_edit)
        grid.addWidget(image_label, 0, 0)
        grid.addWidget(self.image_path_row, 0, 1)
        grid.setColumnStretch(1, 1)
        panel.content_layout.addLayout(grid)
        return panel

    def _action_group(self) -> SectionPanel:
        panel = SectionPanel("Run", compact=True)
        self.operation_state_banner = StatusBanner()
        panel.content_layout.addWidget(self.operation_state_banner)

        self.run_button = set_button_role(QPushButton("Run Inspection"), "primary")
        set_button_icon(self.run_button, "run")
        self.run_button.clicked.connect(self.run_inspection)
        self.cancel_button = set_button_role(QPushButton("Cancel"), "secondary")
        self.cancel_button.setEnabled(False)
        self.cancel_button.clicked.connect(self.cancel_inspection)
        action_section = ActionSection(rows=1)
        action_section.add_row((self.cancel_button, self.run_button), align_right=True)
        panel.content_layout.addWidget(action_section)
        panel.content_layout.addWidget(self.feedback_label)
        return panel

    def _setup_output_group(self) -> SectionPanel:
        panel = SectionPanel("Config and Output", compact=True, classic=True)
        grid = QGridLayout()
        spacing = theme_spacing()
        grid.setContentsMargins(*zero_margins())
        grid.setHorizontalSpacing(spacing.compact_grid_horizontal_gap)
        grid.setVerticalSpacing(spacing.compact_grid_vertical_gap)

        self.config_path_edit = QLineEdit()
        self.config_path_edit.setReadOnly(True)
        config_label = QLabel("Saved &config")
        config_label.setBuddy(self.config_path_edit)
        grid.addWidget(config_label, 0, 0)
        grid.addWidget(self.config_path_edit, 0, 1)

        self.output_dir_row = PathPickerRow("Choose Output Folder...", self._choose_output_dir)
        self.output_dir_edit = self.output_dir_row.line_edit
        self.output_dir_edit.textEdited.connect(self._mark_output_dir_override)
        output_label = QLabel("&Output folder")
        output_label.setBuddy(self.output_dir_edit)
        grid.addWidget(output_label, 1, 0)
        grid.addWidget(self.output_dir_row, 1, 1)

        self.open_output_button = set_button_role(QPushButton("Open Output Folder"), "secondary")
        set_button_icon(self.open_output_button, "open")
        self.open_output_button.clicked.connect(self.open_output_folder)
        grid.addWidget(ActionButtonRow((self.open_output_button,), align_right=True), 2, 1)
        grid.setColumnStretch(1, 1)
        panel.content_layout.addLayout(grid)
        return panel

    def _summary_group(self) -> ResultSummaryPanel:
        self.summary_panel = ResultSummaryPanel(min_height=theme_dimensions().inspect_summary_min_height)
        self.result_badge = self.summary_panel.result_badge
        return self.summary_panel

    def _result_images_group(self) -> SectionPanel:
        panel = SectionPanel("Result Images", compact=True)
        dimensions = theme_dimensions()
        panel.setMinimumHeight(dimensions.inspect_result_panel_min_height)
        panel.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        self.result_image_tabs = ResultImageTabs(
            min_height=dimensions.inspect_result_image_tabs_min_height,
            tabs_min_height=dimensions.inspect_result_tabs_min_height,
            preview_min_height=dimensions.inspect_result_preview_min_height,
        )
        self.fit_image_button = self.result_image_tabs.fit_image_button
        self.tabs = self.result_image_tabs.tabs
        self.annotated_preview = self.result_image_tabs.annotated_preview
        self.heatmap_preview = self.result_image_tabs.heatmap_preview
        self.mask_preview = self.result_image_tabs.mask_preview
        panel.content_layout.addWidget(self.result_image_tabs, 1)
        return panel

    def refresh_from_state(self) -> None:
        self.config_path_edit.setText("" if self.state.config_path is None else str(self.state.config_path))
        if self.state.image_output_dir is None:
            self.output_dir_edit.setText(str(default_inspect_output_dir(self.state)))
        else:
            self.output_dir_edit.setText(str(self.state.image_output_dir))
        self._refresh_operation_state()

    def run_inspection(self) -> None:
        try:
            config_path = require_saved_config_path(self.state)
            image_path = Path(self.image_path_edit.text().strip())
            if not str(image_path):
                raise ValueError("Inspection image path is required.")
            if not image_path.exists():
                raise ValueError(f"Inspection image does not exist: {image_path}")
            output_dir = Path(self.output_dir_edit.text().strip())
            if not str(output_dir):
                raise ValueError("Output directory is required.")
            output_dir.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            self._set_feedback(str(exc), ok=False)
            self._refresh_operation_state()
            return

        derived_output_dir = default_image_output_dir(self.state.inspection_job_slug)
        self.state.output_dir = output_dir
        if self.state.image_output_dir is not None or output_dir != derived_output_dir:
            self.state.image_output_dir = output_dir
        self._last_output_dir = output_dir
        self._set_running(True)
        self._set_feedback("Inspection started.", ok=True)
        self._thread = QThread(self)
        self._worker = InspectionWorker(config_path, image_path, output_dir, runtime_manager=self.runtime_manager)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.status.connect(self._on_worker_status, Qt.ConnectionType.QueuedConnection)
        self._worker.completed.connect(self._on_completed, Qt.ConnectionType.QueuedConnection)
        self._worker.failed.connect(self._on_failed, Qt.ConnectionType.QueuedConnection)
        self._worker.cancelled.connect(self._on_cancelled, Qt.ConnectionType.QueuedConnection)
        self._worker.finished.connect(self._thread.quit)
        self._worker.finished.connect(self._worker.deleteLater)
        self._thread.finished.connect(self._thread.deleteLater)
        self._thread.finished.connect(self._on_worker_thread_finished, Qt.ConnectionType.QueuedConnection)
        self._thread.start()

    def open_output_folder(self) -> None:
        output_dir = Path(self.output_dir_edit.text().strip() or default_inspect_output_dir(self.state))
        if output_dir.exists():
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(output_dir.resolve())))
        else:
            self._set_feedback(f"Output folder does not exist yet: {output_dir}", ok=False)

    def _browse_image(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Choose inspection image", self.image_path_edit.text().strip(), IMAGE_FILTER)
        if path:
            self.image_path_edit.setText(path)

    def _choose_output_dir(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Choose output folder", self.output_dir_edit.text().strip())
        if path:
            self.output_dir_edit.setText(path)
            output_path = Path(path)
            self.state.image_output_dir = output_path
            self.state.output_dir = output_path
            self._refresh_operation_state()

    def _mark_output_dir_override(self, value: str) -> None:
        text = value.strip()
        self.state.image_output_dir = Path(text) if text else None
        self._refresh_operation_state()

    @Slot(object, object)
    def _on_completed(self, result: object, output_dir: object) -> None:
        if not isinstance(result, InspectionResult):
            self._set_feedback("Inspection failed: worker returned an invalid result.", ok=False)
            return
        output_path = Path(output_dir)
        self._update_summary(result, output_path)
        self._update_previews(result)
        ok = result.final_result != FinalResult.ERROR
        self._set_feedback(f"Inspection complete: {result.final_result.value}", ok=ok)
        self._set_operation_state(
            "Complete",
            f"Inspection complete: {result.final_result.value}",
            "success" if ok else "error",
            "complete",
        )

    @Slot(str)
    def _on_failed(self, message: str) -> None:
        self._set_feedback(f"Inspection failed: {message}", ok=False)
        self._set_operation_state("Blocked", f"Inspection failed: {message}", "error", "blocked")

    @Slot(str)
    def _on_cancelled(self, message: str) -> None:
        self._set_feedback(message, ok=True)
        self._set_operation_state("Cancelled", message, "info", "cancelled")

    @Slot(str)
    def _on_worker_status(self, message: str) -> None:
        self._set_feedback(message, ok=True)

    @Slot()
    def _on_worker_thread_finished(self) -> None:
        self._set_running(False)
        self._clear_thread_refs()

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

    def _set_running(self, running: bool) -> None:
        self.run_button.setEnabled(not running)
        self.cancel_button.setEnabled(running)
        if running:
            self._set_operation_state("Inspecting", "Running inspection", "info", "processing")
        elif self._operation_state_key == "processing":
            self._refresh_operation_state()

    def cancel_inspection(self) -> None:
        worker = self._worker
        if worker is None:
            return
        worker.request_cancel()
        self.cancel_button.setEnabled(False)
        message = "Cancellation requested. Waiting for the current backend operation to finish."
        self._set_feedback(message, ok=True)
        self._set_operation_state("Cancelling", message, "info", "cancelling")

    def shutdown(self) -> bool:
        if self._thread is None:
            return True
        self.cancel_inspection()
        try:
            if shiboken6.isValid(self._thread):
                self._thread.requestInterruption()
        except RuntimeError:
            return True
        return False

    def _clear_thread_refs(self) -> None:
        self._thread = None
        self._worker = None

    def _set_feedback(self, message: str, ok: bool) -> None:
        self.feedback_label.set_message(message, "success" if ok else "error")
        self.state.status_message = message
        if self._status_callback:
            self._status_callback(message)

    def _refresh_operation_state(self) -> None:
        if self._thread is not None:
            self._set_operation_state("Inspecting", "Running inspection", "info", "processing")
            return

        config_path = self.state.config_path
        if config_path is None:
            self._set_operation_state("Blocked", "Open or save a config in Project Setup before inspection.", "warning", "blocked")
            return
        if not Path(config_path).exists():
            self._set_operation_state("Blocked", f"Selected config file does not exist: {config_path}", "warning", "blocked")
            return

        image_text = self.image_path_edit.text().strip()
        if not image_text:
            self._set_operation_state("Waiting", "Choose an image to inspect.", "info", "waiting")
            return
        image_path = Path(image_text)
        if not image_path.exists():
            self._set_operation_state("Blocked", f"Inspection image does not exist: {image_path}", "warning", "blocked")
            return

        if not self.output_dir_edit.text().strip():
            self._set_operation_state("Blocked", "Choose an output folder.", "warning", "blocked")
            return

        self._set_operation_state("Ready", "Ready to inspect the selected image.", "success", "ready")

    def _set_operation_state(self, title: str, message: str, level: str, key: str) -> None:
        self._operation_state_key = key
        self.operation_state_banner.set_state(title, message, key, level)
