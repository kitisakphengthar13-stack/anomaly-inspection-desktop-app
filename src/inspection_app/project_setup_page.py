from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from PySide6.QtCore import QObject, QThread, Qt, Signal, Slot
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from inspection.config import load_config
from inspection_app.config_io import (
    config_to_yaml_data,
    default_config_data,
    save_config_data,
    validate_config_data,
)
from inspection_app.job_paths import (
    LEGACY_REFERENCE_IMAGE_PATH,
    LEGACY_ZONES_PATH,
    default_logs_root_dir,
    default_reference_image_path,
    default_zones_path,
)
from inspection_app.runtime import (
    RUNTIME_ERROR,
    RUNTIME_NEEDS_REFRESH,
    RUNTIME_NOT_PREPARED,
    RUNTIME_PREPARING,
    RUNTIME_READY,
    PreparedRuntimeManager,
)
from inspection_app.state import AppState
from inspection_app.theme import page_margins, theme_spacing, zero_margins
from inspection_app.ui_components import ActionButtonRow, MetricGrid, PathPickerRow, SectionPanel, StatusBanner, set_button_icon, set_button_role

StatusCallback = Callable[[str], None]


class RuntimePrepareWorker(QObject):
    completed = Signal()
    failed = Signal(str)
    finished = Signal()

    def __init__(self, runtime_manager: PreparedRuntimeManager, config_path: Path) -> None:
        super().__init__()
        self.runtime_manager = runtime_manager
        self.config_path = config_path

    @Slot()
    def run(self) -> None:
        try:
            self.runtime_manager.prepare(self.config_path)
            self.completed.emit()
        except Exception as exc:
            self.runtime_manager.set_error(str(exc))
            self.failed.emit(str(exc))
        finally:
            self.finished.emit()


class ProjectSetupPage(QWidget):
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
        self._syncing_path_fields = False
        self._runtime_thread: QThread | None = None
        self._runtime_worker: RuntimePrepareWorker | None = None

        self._build_ui()
        self.populate_from_data(default_config_data(), explicit_config=False)
        self.job_name_edit.textChanged.connect(self._update_inspection_job_state)
        self.model_path_edit.textChanged.connect(lambda _text: self._refresh_readiness_summary())
        self.reference_path_edit.textChanged.connect(self._update_reference_path_state)
        self.zones_path_edit.textChanged.connect(self._update_zones_path_state)
        self.save_csv_log_checkbox.stateChanged.connect(lambda _state: self._refresh_readiness_summary())
        self.show_images_checkbox.stateChanged.connect(lambda _state: self._refresh_readiness_summary())
        for widget in (
            self.job_name_edit,
            self.model_path_edit,
            self.reference_path_edit,
            self.zones_path_edit,
        ):
            widget.textEdited.connect(lambda _text: self._mark_runtime_stale())
        for widget in (
            self.model_format_combo,
            self.device_combo,
        ):
            widget.currentIndexChanged.connect(lambda _index: self._mark_runtime_stale())
        for widget in (
            self.anomaly_threshold_spin,
            self.foreground_ratio_spin,
        ):
            widget.valueChanged.connect(lambda _value: self._mark_runtime_stale())
        for widget in (
            self.pixel_diff_spin,
            self.min_blob_area_spin,
            self.blur_kernel_spin,
            self.morphology_kernel_spin,
        ):
            widget.valueChanged.connect(lambda _value: self._mark_runtime_stale())
        for widget in (
            self.use_largest_blob_checkbox,
            self.save_annotated_checkbox,
            self.save_heatmap_checkbox,
            self.save_presence_mask_checkbox,
            self.organize_by_result_checkbox,
            self.show_images_checkbox,
            self.save_csv_log_checkbox,
        ):
            widget.stateChanged.connect(lambda _state: self._mark_runtime_stale())
        self.state.status_message = "Sample defaults loaded. Save a local config before inspection."

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        spacing = theme_spacing()
        layout.setContentsMargins(*page_margins())
        layout.setSpacing(spacing.setup_page_gap)
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        self.feedback_label = StatusBanner()

        layout.addWidget(self._readiness_group())
        layout.addWidget(self._setup_console(), 1)
        layout.addWidget(self.feedback_label)

    def _setup_console(self) -> QWidget:
        console = QWidget()
        console.setObjectName("setupConsole")
        layout = QHBoxLayout(console)
        spacing = theme_spacing()
        layout.setContentsMargins(*zero_margins())
        layout.setSpacing(spacing.section_gap)

        left_column = QWidget()
        left_column.setObjectName("setupConsoleColumn")
        left_column.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        left_layout = QVBoxLayout(left_column)
        left_layout.setContentsMargins(*zero_margins())
        left_layout.setSpacing(spacing.setup_page_gap)
        left_layout.addWidget(self._project_group())
        left_layout.addWidget(self._config_group())
        left_layout.addWidget(self._output_group())
        left_layout.addStretch(1)

        right_column = QWidget()
        right_column.setObjectName("setupConsoleColumn")
        right_column.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        right_layout = QVBoxLayout(right_column)
        right_layout.setContentsMargins(*zero_margins())
        right_layout.setSpacing(spacing.setup_page_gap)
        right_layout.addWidget(self._model_group())
        right_layout.addWidget(self._presence_group())
        right_layout.addStretch(1)

        layout.addWidget(left_column, 1)
        layout.addWidget(right_column, 1)
        return console

    def _readiness_group(self) -> SectionPanel:
        panel = SectionPanel("Readiness", compact=True)
        self.readiness_banner = StatusBanner()
        panel.content_layout.addWidget(self.readiness_banner)

        self.readiness_grid = MetricGrid()
        panel.content_layout.addWidget(self.readiness_grid)
        return panel

    def _project_group(self) -> SectionPanel:
        panel = SectionPanel("Inspection Job", compact=True)
        form = QFormLayout()
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)
        self.job_name_edit = QLineEdit()
        form.addRow("Inspection Job", self.job_name_edit)
        panel.content_layout.addLayout(form)
        return panel

    def _config_group(self) -> SectionPanel:
        panel = SectionPanel("Config File", compact=True)
        layout = panel.content_layout

        path_row = PathPickerRow("Open Config...", self.open_config)
        self.config_path_edit = path_row.line_edit
        self.config_path_edit.setPlaceholderText("No local config selected")
        layout.addWidget(path_row)

        save_button = set_button_role(QPushButton("Save Config"), "secondary")
        set_button_icon(save_button, "save")
        save_button.clicked.connect(self.save_config)
        save_as_button = set_button_role(QPushButton("Save Config As..."), "secondary")
        set_button_icon(save_as_button, "save")
        save_as_button.clicked.connect(self.save_config_as)
        validate_button = set_button_role(QPushButton("Validate Config"), "secondary")
        set_button_icon(validate_button, "ready")
        validate_button.clicked.connect(self.validate_current_config)
        layout.addWidget(ActionButtonRow((save_button, save_as_button, validate_button)))
        return panel

    def _runtime_row(self) -> QWidget:
        runtime_row = QWidget()
        runtime_row.setObjectName("runtimePreparationRow")
        row_layout = QHBoxLayout(runtime_row)
        row_layout.setContentsMargins(*zero_margins())
        row_layout.setSpacing(theme_spacing().control_gap)
        row_layout.setAlignment(Qt.AlignmentFlag.AlignVCenter)

        self.runtime_banner = StatusBanner()
        self.runtime_banner.set_inline_mode()
        row_layout.addWidget(self.runtime_banner, 1)

        self.prepare_runtime_button = set_button_role(QPushButton("Prepare Runtime"), "secondary")
        set_button_icon(self.prepare_runtime_button, "run")
        self.prepare_runtime_button.clicked.connect(self.prepare_runtime)
        row_layout.addWidget(self.prepare_runtime_button, 0, Qt.AlignmentFlag.AlignVCenter)

        self._refresh_runtime_status()
        return runtime_row

    def _model_group(self) -> SectionPanel:
        panel = SectionPanel("Model", compact=True)
        form = QFormLayout()
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)

        model_path_row = PathPickerRow("Browse Model...", self._browse_model)
        self.model_path_edit = model_path_row.line_edit
        form.addRow("Model path", model_path_row)

        self.model_format_combo = QComboBox()
        self.model_format_combo.addItems(["auto", "ckpt", "torch_export"])
        form.addRow("Model artifact type", self.model_format_combo)

        self.anomaly_threshold_spin = QDoubleSpinBox()
        self.anomaly_threshold_spin.setRange(0.0, 1_000_000_000.0)
        self.anomaly_threshold_spin.setDecimals(3)
        self.anomaly_threshold_spin.setSingleStep(0.05)
        form.addRow("Decision threshold", self.anomaly_threshold_spin)

        self.device_combo = QComboBox()
        self.device_combo.addItems(["auto", "cuda", "cpu"])
        form.addRow("Device", self.device_combo)
        panel.content_layout.addLayout(form)

        runtime_label = QLabel("Runtime")
        runtime_label.setObjectName("sectionPanelSubtitle")
        panel.content_layout.addWidget(runtime_label)
        panel.content_layout.addWidget(self._runtime_row())
        return panel

    def _presence_group(self) -> SectionPanel:
        panel = SectionPanel("Presence Gate", compact=True)
        form = QFormLayout()
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)

        reference_path_row = PathPickerRow("Browse Reference...", self._browse_reference)
        self.reference_path_edit = reference_path_row.line_edit
        form.addRow("Reference image", reference_path_row)

        zones_path_row = PathPickerRow("Browse Zones...", self._browse_zones)
        self.zones_path_edit = zones_path_row.line_edit
        form.addRow("Zones JSON", zones_path_row)
        panel.content_layout.addLayout(form)

        self.presence_tuning_button = set_button_role(QPushButton("Presence tuning"), "secondary")
        self.presence_tuning_button.setObjectName("presenceTuningButton")
        self.presence_tuning_button.setCheckable(True)
        self.presence_tuning_button.toggled.connect(self._set_presence_tuning_visible)
        panel.content_layout.addWidget(self.presence_tuning_button)

        self.presence_tuning_widget = QWidget()
        self.presence_tuning_widget.setObjectName("presenceTuningFields")
        tuning_form = QFormLayout(self.presence_tuning_widget)
        tuning_form.setContentsMargins(*zero_margins())
        tuning_form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)

        self.pixel_diff_spin = QSpinBox()
        self.pixel_diff_spin.setRange(0, 255)
        tuning_form.addRow("Pixel difference threshold", self.pixel_diff_spin)

        self.foreground_ratio_spin = QDoubleSpinBox()
        self.foreground_ratio_spin.setRange(0.0, 1.0)
        self.foreground_ratio_spin.setDecimals(3)
        self.foreground_ratio_spin.setSingleStep(0.01)
        tuning_form.addRow("Minimum foreground", self.foreground_ratio_spin)

        self.min_blob_area_spin = QSpinBox()
        self.min_blob_area_spin.setRange(0, 1_000_000_000)
        tuning_form.addRow("Minimum blob area (px^2)", self.min_blob_area_spin)

        self.blur_kernel_spin = QSpinBox()
        self.blur_kernel_spin.setRange(0, 9999)
        tuning_form.addRow("Blur kernel size", self.blur_kernel_spin)

        self.morphology_kernel_spin = QSpinBox()
        self.morphology_kernel_spin.setRange(0, 9999)
        tuning_form.addRow("Morphology kernel size", self.morphology_kernel_spin)

        self.use_largest_blob_checkbox = QCheckBox("Use largest blob filter")
        tuning_form.addRow("", self.use_largest_blob_checkbox)
        self.presence_tuning_widget.setVisible(False)
        panel.content_layout.addWidget(self.presence_tuning_widget)
        return panel

    def _output_group(self) -> SectionPanel:
        panel = SectionPanel("Outputs")

        self.save_annotated_checkbox = QCheckBox("Save annotated image")
        self.save_heatmap_checkbox = QCheckBox("Save heatmap")
        self.save_presence_mask_checkbox = QCheckBox("Save presence mask")
        self.organize_by_result_checkbox = QCheckBox("Organize by result")
        self.show_images_checkbox = QCheckBox("Show images after inspection")
        self.save_csv_log_checkbox = QCheckBox("Save CSV inspection log")

        grid = QGridLayout()
        spacing = theme_spacing()
        grid.setHorizontalSpacing(spacing.compact_grid_horizontal_gap)
        grid.setVerticalSpacing(spacing.compact_grid_vertical_gap)
        for index, checkbox in enumerate((
            self.save_annotated_checkbox,
            self.save_heatmap_checkbox,
            self.save_presence_mask_checkbox,
            self.organize_by_result_checkbox,
            self.show_images_checkbox,
            self.save_csv_log_checkbox,
        )):
            grid.addWidget(checkbox, index // 2, index % 2)
        panel.content_layout.addLayout(grid)
        return panel

    def _set_presence_tuning_visible(self, visible: bool) -> None:
        self.presence_tuning_widget.setVisible(visible)

    def open_config(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Open inspection config", "", "YAML files (*.yaml *.yml);;All files (*)")
        if not path:
            return
        try:
            config = load_config(path)
            self.populate_from_data(config_to_yaml_data(config), explicit_config=True)
            self._set_config_path(Path(path))
            self.state.config_loaded = True
            self.state.last_validation_message = "Config loaded successfully."
            self._mark_runtime_stale()
            self._set_feedback(f"Loaded config: {path}", ok=True)
        except Exception as exc:
            self.state.config_loaded = False
            self.state.last_validation_message = str(exc)
            self._set_feedback(f"Failed to load config: {exc}", ok=False)

    def save_config(self) -> None:
        config_path = self._current_config_path()
        if config_path is None:
            self._set_feedback("Choose Save Config As before saving a new config.", ok=False)
            return
        self._save_to_path(config_path)

    def save_config_as(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Save inspection config", "", "YAML files (*.yaml *.yml);;All files (*)")
        if not path:
            return
        self._save_to_path(Path(path))

    def validate_current_config(self) -> None:
        data = self.to_config_data()
        try:
            validate_config_data(data, self._current_config_path(), require_existing_files=True)
            message = "Config validation passed. Required files exist."
            self.state.last_validation_message = message
            self._set_feedback(message, ok=True)
        except Exception as exc:
            self.state.last_validation_message = str(exc)
            self._set_feedback(f"Config validation failed: {exc}", ok=False)

    def populate_from_data(self, data: dict[str, Any], *, explicit_config: bool = True) -> None:
        project = data.get("project", {}) or {}
        model = data.get("model", {}) or {}
        presence = data.get("presence", {}) or {}
        output = data.get("output", {}) or {}

        self.job_name_edit.setText(str(project.get("name", "Default Job")).strip() or "Default Job")
        self._update_inspection_job_state(self.job_name_edit.text())
        self.model_path_edit.setText(str(model.get("path", model.get("checkpoint_path", ""))))
        self._set_combo_value(self.model_format_combo, str(model.get("format", "auto")))
        self.anomaly_threshold_spin.setValue(float(model.get("anomaly_threshold", 0.5)))
        self._set_combo_value(self.device_combo, str(model.get("device", "auto")))

        reference_value = _presence_path_value(presence.get("reference_image_path"))
        zones_value = _presence_path_value(presence.get("zones_path"))
        if explicit_config:
            self._set_reference_path(reference_value, is_auto=False)
            self._set_zones_path(zones_value, is_auto=False)
        else:
            self._set_reference_path(
                self._default_reference_path() if _should_replace_with_job_default(reference_value, LEGACY_REFERENCE_IMAGE_PATH) else reference_value,
                is_auto=_should_replace_with_job_default(reference_value, LEGACY_REFERENCE_IMAGE_PATH),
            )
            self._set_zones_path(
                self._default_zones_path() if _should_replace_with_job_default(zones_value, LEGACY_ZONES_PATH) else zones_value,
                is_auto=_should_replace_with_job_default(zones_value, LEGACY_ZONES_PATH),
            )
        self.pixel_diff_spin.setValue(int(presence.get("pixel_diff_threshold", 30)))
        self.foreground_ratio_spin.setValue(float(presence.get("min_foreground_ratio", 0.08)))
        self.min_blob_area_spin.setValue(int(presence.get("min_blob_area", 500)))
        self.blur_kernel_spin.setValue(int(presence.get("blur_kernel_size", 5)))
        self.morphology_kernel_spin.setValue(int(presence.get("morphology_kernel_size", 5)))
        self.use_largest_blob_checkbox.setChecked(bool(presence.get("use_largest_blob_filter", True)))

        self.save_annotated_checkbox.setChecked(bool(output.get("save_annotated", True)))
        self.save_heatmap_checkbox.setChecked(bool(output.get("save_heatmap", True)))
        self.save_presence_mask_checkbox.setChecked(bool(output.get("save_presence_mask", True)))
        self.organize_by_result_checkbox.setChecked(bool(output.get("organize_by_result", True)))
        self.show_images_checkbox.setChecked(bool(output.get("show_images", True)))
        self.save_csv_log_checkbox.setChecked(bool(output.get("save_csv_log", True)))
        self._refresh_readiness_summary()

    def to_config_data(self) -> dict[str, Any]:
        return {
            "project": {
                "name": self.job_name_edit.text().strip() or "Default Job",
            },
            "model": {
                "path": self.model_path_edit.text().strip(),
                "format": self.model_format_combo.currentText(),
                "anomaly_threshold": self.anomaly_threshold_spin.value(),
                "device": self.device_combo.currentText(),
            },
            "presence": {
                "reference_image_path": self.reference_path_edit.text().strip(),
                "zones_path": self.zones_path_edit.text().strip(),
                "pixel_diff_threshold": self.pixel_diff_spin.value(),
                "min_foreground_ratio": self.foreground_ratio_spin.value(),
                "min_blob_area": self.min_blob_area_spin.value(),
                "blur_kernel_size": self.blur_kernel_spin.value(),
                "morphology_kernel_size": self.morphology_kernel_spin.value(),
                "use_largest_blob_filter": self.use_largest_blob_checkbox.isChecked(),
            },
            "output": {
                "save_annotated": self.save_annotated_checkbox.isChecked(),
                "save_heatmap": self.save_heatmap_checkbox.isChecked(),
                "save_presence_mask": self.save_presence_mask_checkbox.isChecked(),
                "organize_by_result": self.organize_by_result_checkbox.isChecked(),
                "show_images": self.show_images_checkbox.isChecked(),
                "save_csv_log": self.save_csv_log_checkbox.isChecked(),
            },
        }

    def set_reference_image_path(self, path: str | Path) -> None:
        path_value = Path(path)
        self._set_reference_path(path_value, is_auto=path_value == self._default_reference_path())
        self._mark_runtime_stale()

    def set_zones_path(self, path: str | Path) -> None:
        path_value = Path(path)
        self._set_zones_path(path_value, is_auto=path_value == self._default_zones_path())
        self._mark_runtime_stale()

    def refresh_from_state(self) -> None:
        if self.state.reference_image_path is not None and self.reference_path_edit.text().strip() != str(self.state.reference_image_path):
            self._set_reference_path(self.state.reference_image_path, is_auto=self.state.reference_path_auto)
        if self.state.zones_path is not None and self.zones_path_edit.text().strip() != str(self.state.zones_path):
            self._set_zones_path(self.state.zones_path, is_auto=self.state.zones_path_auto)
        self._refresh_readiness_summary()

    def _save_to_path(self, path: Path) -> None:
        try:
            data = self.to_config_data()
            validate_config_data(data, path)
            save_config_data(path, data)
            self._set_config_path(path)
            self.state.config_loaded = True
            self.state.last_validation_message = "Config saved and validated successfully."
            self._mark_runtime_stale()
            self._set_feedback(f"Saved config: {path}", ok=True)
        except Exception as exc:
            self.state.last_validation_message = str(exc)
            self._set_feedback(f"Failed to save config: {exc}", ok=False)

    def _browse_model(self) -> None:
        self._browse_file(self.model_path_edit, "Choose anomaly model", "Model files (*.ckpt *.pt);;All files (*)")

    def _browse_reference(self) -> None:
        self._browse_file(self.reference_path_edit, "Choose reference image", "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff);;All files (*)")

    def _browse_zones(self) -> None:
        self._browse_file(self.zones_path_edit, "Choose zones JSON", "JSON files (*.json);;All files (*)")

    def _browse_file(self, target: QLineEdit, title: str, filter_text: str) -> None:
        path, _ = QFileDialog.getOpenFileName(self, title, "", filter_text)
        if path:
            target.setText(path)

    def _set_config_path(self, path: Path) -> None:
        resolved = path.resolve()
        self.state.config_path = resolved
        self.config_path_edit.setText(str(resolved))
        self._refresh_readiness_summary()

    def _current_config_path(self) -> Path | None:
        raw_path = self.config_path_edit.text().strip()
        return Path(raw_path) if raw_path else self.state.config_path

    def _set_feedback(self, message: str, ok: bool, level: str | None = None) -> None:
        if ok:
            self.feedback_label.clear()
        else:
            self.feedback_label.set_message(message, level or "error")
        self.state.status_message = message
        if self._status_callback:
            self._status_callback(message)

    def prepare_runtime(self) -> None:
        if self.runtime_manager is None:
            self._set_feedback("Runtime preparation is not available.", ok=False)
            return
        if self._runtime_thread is not None:
            return
        config_path = self._current_config_path()
        if config_path is None:
            self._set_feedback("Save a local config before preparing the runtime.", ok=False)
            return
        if not config_path.exists():
            self._set_feedback(f"Selected config file does not exist: {config_path}", ok=False)
            return
        self.runtime_manager.mark_preparing()
        self._refresh_runtime_status()
        self.prepare_runtime_button.setEnabled(False)
        self.state.status_message = "Preparing runtime..."
        if self._status_callback:
            self._status_callback("Preparing runtime...")
        self._runtime_thread = QThread(self)
        self._runtime_worker = RuntimePrepareWorker(self.runtime_manager, config_path)
        self._runtime_worker.moveToThread(self._runtime_thread)
        self._runtime_thread.started.connect(self._runtime_worker.run)
        self._runtime_worker.completed.connect(self._on_runtime_prepared)
        self._runtime_worker.failed.connect(self._on_runtime_prepare_failed)
        self._runtime_worker.finished.connect(self._runtime_thread.quit)
        self._runtime_worker.finished.connect(self._runtime_worker.deleteLater)
        self._runtime_thread.finished.connect(self._runtime_thread.deleteLater)
        self._runtime_thread.finished.connect(self._clear_runtime_worker)
        self._runtime_thread.start()

    def _on_runtime_prepared(self) -> None:
        self._refresh_runtime_status()
        message = "Runtime ready. Model backend is prepared for inspection."
        self.state.status_message = message
        if self._status_callback:
            self._status_callback(message)

    def _on_runtime_prepare_failed(self, message: str) -> None:
        self._refresh_runtime_status()
        self._set_feedback(f"Runtime preparation failed: {message}", ok=False)

    def _clear_runtime_worker(self) -> None:
        self._runtime_thread = None
        self._runtime_worker = None
        self.prepare_runtime_button.setEnabled(True)
        self._refresh_runtime_status()

    def _mark_runtime_stale(self) -> None:
        if self.runtime_manager is None:
            return
        self.runtime_manager.mark_stale()
        self._refresh_runtime_status()

    def _refresh_runtime_status(self) -> None:
        if not hasattr(self, "runtime_banner"):
            return
        if self.runtime_manager is None:
            self.runtime_banner.set_state("Not prepared", "Runtime preparation is not available.", "idle", "info")
            set_button_role(self.prepare_runtime_button, "secondary")
            return
        state = self.runtime_manager.state
        if state == RUNTIME_READY:
            self.runtime_banner.set_state("Ready", "Prepared runtime matches the saved config.", "ready", "success")
            set_button_role(self.prepare_runtime_button, "secondary")
        elif state == RUNTIME_PREPARING:
            self.runtime_banner.set_state("Preparing", "Loading config, pipeline, and model backend.", "processing", "info")
            set_button_role(self.prepare_runtime_button, "secondary")
        elif state == RUNTIME_NEEDS_REFRESH:
            self.runtime_banner.set_state("Needs refresh", "Prepare runtime again after config changes.", "waiting", "warning")
            set_button_role(self.prepare_runtime_button, "primary")
        elif state == RUNTIME_ERROR:
            self.runtime_banner.set_state("Error", self.runtime_manager.error_message or "Runtime preparation failed.", "error", "error")
            set_button_role(self.prepare_runtime_button, "secondary")
        else:
            self.runtime_banner.set_state("Not prepared", "Prepare runtime to reduce first inspection latency.", "idle", "info")
            set_button_role(self.prepare_runtime_button, "primary")

    def shutdown(self) -> None:
        thread = self._runtime_thread
        if thread is None:
            return
        thread.quit()
        thread.wait()
        self._runtime_thread = None
        self._runtime_worker = None

    def _update_inspection_job_state(self, value: str) -> None:
        old_reference_auto = self.state.reference_path_auto
        old_zones_auto = self.state.zones_path_auto
        self.state.set_inspection_job_name(value)
        if hasattr(self, "reference_path_edit"):
            if old_reference_auto or not self.reference_path_edit.text().strip():
                self._set_reference_path(self._default_reference_path(), is_auto=True)
            if old_zones_auto or not self.zones_path_edit.text().strip():
                self._set_zones_path(self._default_zones_path(), is_auto=True)
        self._refresh_readiness_summary()

    def _update_reference_path_state(self, value: str) -> None:
        text = value.strip()
        self.state.reference_image_path = Path(text) if text else None
        if not self._syncing_path_fields:
            self.state.reference_path_auto = False if text else True
        self._refresh_readiness_summary()

    def _update_zones_path_state(self, value: str) -> None:
        text = value.strip()
        self.state.zones_path = Path(text) if text else None
        if not self._syncing_path_fields:
            self.state.zones_path_auto = False if text else True
        self._refresh_readiness_summary()

    def _set_reference_path(self, path: str | Path | None, *, is_auto: bool) -> None:
        self._syncing_path_fields = True
        try:
            text = "" if path is None else str(path)
            self.reference_path_edit.setText(text)
            self.state.reference_image_path = Path(text) if text else None
            self.state.reference_path_auto = is_auto
        finally:
            self._syncing_path_fields = False
        self._refresh_readiness_summary()

    def _set_zones_path(self, path: str | Path | None, *, is_auto: bool) -> None:
        self._syncing_path_fields = True
        try:
            text = "" if path is None else str(path)
            self.zones_path_edit.setText(text)
            self.state.zones_path = Path(text) if text else None
            self.state.zones_path_auto = is_auto
        finally:
            self._syncing_path_fields = False
        self._refresh_readiness_summary()

    def _default_reference_path(self) -> Path:
        return default_reference_image_path(self.state.inspection_job_slug)

    def _default_zones_path(self) -> Path:
        return default_zones_path(self.state.inspection_job_slug)

    def _refresh_readiness_summary(self) -> None:
        if not hasattr(self, "readiness_banner"):
            return
        config_path = self._current_config_path()
        model_path = Path(self.model_path_edit.text().strip()) if self.model_path_edit.text().strip() else None
        reference_path = self.state.reference_image_path
        zones_path = self.state.zones_path
        output_root = default_logs_root_dir(self.state.inspection_job_slug)

        items = (
            ("Config file", self._path_state_text(config_path, allow_missing=False)),
            ("Model artifact", self._path_state_text(model_path, allow_missing=False)),
            ("Reference image", self._path_state_text(reference_path, allow_missing=False)),
            ("Zones JSON", self._path_state_text(zones_path, allow_missing=False)),
            ("Output root", str(output_root)),
            ("CSV log", "Enabled" if self.save_csv_log_checkbox.isChecked() else "Disabled"),
        )
        self.readiness_grid.set_metrics(items)

        missing = [
            label
            for label, path in (
                ("config", config_path),
                ("model", model_path),
                ("reference", reference_path),
                ("zones", zones_path),
            )
            if path is None or not path.exists()
        ]
        if missing:
            self.readiness_banner.set_state("Incomplete", f"Missing {', '.join(missing)}.", "blocked", "warning")
        else:
            self.readiness_banner.set_state("Ready", "Config, model, reference image, and zones file are available.", "ready", "success")

    @staticmethod
    def _path_state_text(path: Path | None, *, allow_missing: bool) -> str:
        if path is None or not str(path).strip():
            return "Not selected"
        if path.exists():
            return f"Available: {path}"
        if allow_missing:
            return str(path)
        return f"Missing: {path}"

    @staticmethod
    def _set_combo_value(combo: QComboBox, value: str) -> None:
        index = combo.findText(value.lower(), Qt.MatchFlag.MatchFixedString)
        combo.setCurrentIndex(index if index >= 0 else 0)


def _presence_path_value(value: object) -> str | None:
    text = str(value or "").strip()
    return text or None


def _should_replace_with_job_default(path: str | Path | None, legacy_path: Path) -> bool:
    return path is None or Path(path) == legacy_path
