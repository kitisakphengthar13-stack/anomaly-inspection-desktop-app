from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
import shiboken6
from PySide6.QtCore import QObject, QThread, Qt, QTimer, Signal, Slot
from PySide6.QtGui import QImage, QPainter, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSizePolicy,
    QSplitter,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from inspection.reference_capture import save_reference_image
from inspection_app.icons import icon_pixmap, state_icon
from inspection_app.job_paths import default_reference_image_path as job_reference_image_path
from inspection_app.state import AppState
from inspection_app.theme import page_margins, preview_surface_stylesheet, theme_dimensions, theme_spacing, zero_margins
from inspection_app.ui_components import ActionButtonRow, ScrollablePane, SectionPanel, StatusBanner, set_button_icon, set_button_role

StatusCallback = Callable[[str], None]
ReferenceSavedCallback = Callable[[Path], None]
CaptureFactory = Callable[..., object]

PREVIEW_INTERVAL_SECONDS = 1.0 / 12.0
BLANK_STARTUP_FRAME_LOG_LIMIT = 5
STARTUP_BLANK_RECOVERY_FRAMES = 6
STARTUP_BLANK_FINAL_FAIL_FRAMES = 6
MAX_BLANK_RECOVERY_ATTEMPTS = 1


def camera_log(message: str) -> None:
    print(f"[camera] {message}", flush=True)


def camera_backend_candidates(platform_name: str | None = None) -> list[tuple[str, int | None]]:
    platform_value = platform_name or sys.platform
    if platform_value.startswith("win"):
        return [("DSHOW", cv2.CAP_DSHOW), ("MSMF", cv2.CAP_MSMF)]
    return [("DEFAULT", None)]


def _open_camera_capture_with_backend(
    camera_index: int,
    capture_factory: CaptureFactory | None = None,
    platform_name: str | None = None,
) -> tuple[object, str, str, int | None]:
    factory = capture_factory or cv2.VideoCapture
    candidates = camera_backend_candidates(platform_name)
    attempted = []
    for backend_name, backend_id in candidates:
        attempted.append(backend_name)
        camera_log(f"trying camera index {camera_index} with backend {backend_name}")
        try:
            capture = factory(camera_index) if backend_id is None else factory(camera_index, backend_id)
        except Exception as exc:
            camera_log(f"backend {backend_name} constructor failed: {exc}")
            continue
        try:
            opened = bool(capture.isOpened())
            camera_log(f"backend {backend_name} isOpened={opened}")
            if opened:
                selected = backend_name
                try:
                    selected = f"{backend_name}/{capture.getBackendName()}"
                except Exception:
                    pass
                camera_log(f"opened camera index {camera_index} with backend {selected}")
                return capture, selected, backend_name, backend_id
        except Exception as exc:
            camera_log(f"backend {backend_name} open check failed: {exc}")
        try:
            capture.release()
        except Exception:
            pass
    raise RuntimeError(f"Unable to open camera index {camera_index} using {' or '.join(attempted)}.")


def open_camera_capture(
    camera_index: int,
    capture_factory: CaptureFactory | None = None,
    platform_name: str | None = None,
) -> tuple[object, str]:
    capture, selected, _, _ = _open_camera_capture_with_backend(camera_index, capture_factory, platform_name)
    return capture, selected


def reopen_camera_capture_with_backend(
    camera_index: int,
    backend_name: str,
    backend_id: int | None,
    capture_factory: CaptureFactory | None = None,
) -> tuple[object, str]:
    factory = capture_factory or cv2.VideoCapture
    camera_log(f"reopening camera index {camera_index} with backend {backend_name}")
    capture = factory(camera_index) if backend_id is None else factory(camera_index, backend_id)
    try:
        opened = bool(capture.isOpened())
        camera_log(f"reopened backend {backend_name} isOpened={opened}")
        if opened:
            selected = backend_name
            try:
                selected = f"{backend_name}/{capture.getBackendName()}"
            except Exception:
                pass
            camera_log(f"reopened camera index {camera_index} with backend {selected}")
            return capture, selected
    except Exception as exc:
        camera_log(f"reopened backend {backend_name} open check failed: {exc}")
    try:
        capture.release()
    except Exception:
        pass
    raise RuntimeError(f"Unable to reopen camera index {camera_index} using {backend_name}.")


def default_reference_output_path(state: AppState) -> Path:
    if state.reference_image_path is not None:
        return state.reference_image_path
    return job_reference_image_path(state.inspection_job_slug)


def normalize_reference_output_path(raw_path: str | Path) -> Path:
    path = Path(str(raw_path).strip())
    if not str(path):
        raise ValueError("Output reference image path is required.")
    if not path.suffix:
        path = path.with_suffix(".png")
    if path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}:
        raise ValueError(f"Unsupported image extension for reference image: {path.suffix}")
    return path


def save_captured_reference_frame(frame: np.ndarray | None, output_path: str | Path) -> tuple[Path, tuple[int, int]]:
    if frame is None:
        raise ValueError("No captured frame is available to save.")
    path = normalize_reference_output_path(output_path)
    saved_path = save_reference_image(frame, path)
    height, width = frame.shape[:2]
    return saved_path, (width, height)


def frame_to_pixmap(frame: np.ndarray) -> QPixmap:
    if not getattr(frame_to_pixmap, "_logged_first", False):
        camera_log(f"first frame_to_pixmap for frame shape {frame.shape}")
        setattr(frame_to_pixmap, "_logged_first", True)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    height, width, channels = rgb.shape
    bytes_per_line = channels * width
    image = QImage(rgb.data, width, height, bytes_per_line, QImage.Format.Format_RGB888).copy()
    return QPixmap.fromImage(image)


def _frame_diagnostics(frame: np.ndarray) -> str:
    try:
        return (
            f"frame shape={frame.shape}; dtype={frame.dtype}; "
            f"min={frame.min()}; max={frame.max()}; mean={float(frame.mean()):.2f}"
        )
    except Exception as exc:
        return f"frame diagnostics unavailable: {exc}"


def is_blank_startup_frame(frame: np.ndarray, *, max_value: int = 2, mean_max: float = 1.0) -> bool:
    try:
        return int(frame.max()) <= max_value and float(frame.mean()) <= mean_max
    except Exception:
        return False


def is_gui_thread() -> bool:
    app = QApplication.instance()
    return app is not None and QThread.currentThread() is app.thread()


def log_if_not_gui_thread(context: str) -> bool:
    if is_gui_thread():
        return True
    camera_log(f"GUI thread violation ignored in {context}")
    return False


class CameraWorker(QObject):
    frame_ready = Signal(int, object)
    actual_resolution = Signal(int, int, int)
    status = Signal(int, str, str)
    error = Signal(int, str)
    finished = Signal(int)

    def __init__(
        self,
        camera_index: int,
        width: int = 0,
        height: int = 0,
        capture_factory: CaptureFactory | None = None,
        session_id: int = 0,
        blank_recovery_frames: int = STARTUP_BLANK_RECOVERY_FRAMES,
        blank_final_fail_frames: int = STARTUP_BLANK_FINAL_FAIL_FRAMES,
        max_blank_recovery_attempts: int = MAX_BLANK_RECOVERY_ATTEMPTS,
        blank_max_value: int = 2,
        blank_mean_max: float = 1.0,
        preview_interval_seconds: float = PREVIEW_INTERVAL_SECONDS,
    ) -> None:
        super().__init__()
        self.session_id = session_id
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self._capture_factory = capture_factory
        self._running = False
        self._blank_recovery_frames = blank_recovery_frames
        self._blank_final_fail_frames = blank_final_fail_frames
        self._max_blank_recovery_attempts = max_blank_recovery_attempts
        self._blank_max_value = blank_max_value
        self._blank_mean_max = blank_mean_max
        self._preview_interval_seconds = preview_interval_seconds

    @Slot()
    def run(self) -> None:
        capture = None
        backend_name = ""
        backend_label = ""
        backend_id: int | None = None
        recovery_attempts = 0
        startup_frame_count = 0
        blank_frame_count = 0
        first_valid_frame_seen = False
        reported_resolution = False
        logged_first_read = False
        logged_first_emit = False
        last_emit_time = 0.0
        try:
            camera_log(
                f"worker started: session={self.session_id}; camera_index={self.camera_index}, "
                f"requested_size={self.width}x{self.height}"
            )
            capture, backend_name, backend_label, backend_id = _open_camera_capture_with_backend(
                self.camera_index,
                self._capture_factory,
            )
            self._configure_capture(capture)

            self._running = True
            while self._running:
                ok, frame = capture.read()
                if not ok or frame is None:
                    camera_log(f"camera frame read failed; session={self.session_id}")
                    self.error.emit(self.session_id, "Could not read camera frame.")
                    break
                if not logged_first_read:
                    camera_log(
                        f"first successful capture.read() from {backend_name}; "
                        f"session={self.session_id}; frame shape={frame.shape}"
                    )
                    logged_first_read = True
                if not first_valid_frame_seen:
                    startup_frame_count += 1
                    blank_startup = is_blank_startup_frame(
                        frame,
                        max_value=self._blank_max_value,
                        mean_max=self._blank_mean_max,
                    )
                    if startup_frame_count <= BLANK_STARTUP_FRAME_LOG_LIMIT:
                        camera_log(
                            "camera worker startup frame stats; "
                            f"session={self.session_id}; backend={backend_name}; "
                            f"frame_index={startup_frame_count}; blank_frames={blank_frame_count}; "
                            f"{_frame_diagnostics(frame)}; blank_startup={blank_startup}"
                        )
                    if blank_startup:
                        blank_frame_count += 1
                        if blank_frame_count == 1:
                            self.status.emit(self.session_id, "Warming up camera...", "info")
                        if blank_frame_count == self._blank_recovery_frames - 1 and recovery_attempts < self._max_blank_recovery_attempts:
                            camera_log(
                                f"camera worker blank startup recovery pending; session={self.session_id}; "
                                f"backend={backend_name}; blank_frames={blank_frame_count}"
                            )
                            self.status.emit(
                                self.session_id,
                                "Blank startup frames detected; reopening camera...",
                                "warning",
                            )
                        if blank_frame_count >= self._blank_recovery_frames:
                            if recovery_attempts < self._max_blank_recovery_attempts:
                                if not self._running:
                                    break
                                recovery_attempts += 1
                                camera_log(
                                    f"camera worker blank startup recovery; session={self.session_id}; "
                                    f"backend={backend_name}; attempt={recovery_attempts}"
                                )
                                self.status.emit(
                                    self.session_id,
                                    "Blank startup frames detected; reopening camera...",
                                    "warning",
                                )
                                try:
                                    capture.release()
                                    camera_log(f"camera capture released before blank recovery; session={self.session_id}")
                                except Exception as exc:
                                    camera_log(f"camera release before blank recovery failed; session={self.session_id}; error={exc}")
                                capture, backend_name = reopen_camera_capture_with_backend(
                                    self.camera_index,
                                    backend_label,
                                    backend_id,
                                    self._capture_factory,
                                )
                                self._configure_capture(capture)
                                if not self._running:
                                    break
                                startup_frame_count = 0
                                blank_frame_count = 0
                                reported_resolution = False
                                logged_first_read = False
                                logged_first_emit = False
                                last_emit_time = 0.0
                                continue
                        if recovery_attempts >= self._max_blank_recovery_attempts and blank_frame_count >= self._blank_final_fail_frames:
                            message = (
                                "Camera opened but frames stayed blank. "
                                "Please retry or check exposure, lens, privacy shutter, or driver."
                            )
                            camera_log(
                                f"camera worker blank startup recovery failed; session={self.session_id}; "
                                f"backend={backend_name}; blank_frames={blank_frame_count}"
                            )
                            self.error.emit(self.session_id, message)
                            break
                        time.sleep(0.005)
                        continue
                    first_valid_frame_seen = True
                    camera_log(
                        f"camera worker first valid frame; session={self.session_id}; backend={backend_name}; "
                        f"startup_frames={startup_frame_count}; blank_frames={blank_frame_count}; "
                        f"recovery_attempts={recovery_attempts}"
                    )
                if not reported_resolution:
                    actual_height, actual_width = frame.shape[:2]
                    self.actual_resolution.emit(self.session_id, actual_width, actual_height)
                    reported_resolution = True
                now = time.monotonic()
                if now - last_emit_time >= self._preview_interval_seconds:
                    if not logged_first_emit:
                        camera_log(f"first frame_ready emit; session={self.session_id}")
                        logged_first_emit = True
                    self.frame_ready.emit(self.session_id, frame.copy())
                    last_emit_time = now
                time.sleep(0.005)
        except Exception as exc:
            camera_log(f"camera worker error; session={self.session_id}; error={exc}")
            self.error.emit(self.session_id, str(exc))
        finally:
            if capture is not None:
                try:
                    capture.release()
                    camera_log(f"camera capture released; session={self.session_id}")
                except Exception as exc:
                    camera_log(f"camera release failed; session={self.session_id}; error={exc}")
            self.finished.emit(self.session_id)

    @Slot()
    def stop(self) -> None:
        self._running = False

    def _configure_capture(self, capture: object) -> None:
        if self.width > 0:
            camera_log(f"requesting camera width {self.width}; session={self.session_id}")
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        if self.height > 0:
            camera_log(f"requesting camera height {self.height}; session={self.session_id}")
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)


class AspectImageLabel(QWidget):
    def __init__(
        self,
        *,
        image_alignment: Qt.AlignmentFlag = Qt.AlignmentFlag.AlignCenter,
        preserve_aspect_size: bool = False,
    ) -> None:
        super().__init__()
        self.setMinimumHeight(theme_dimensions().camera_preview_base_min_height)
        vertical_policy = QSizePolicy.Policy.Preferred if preserve_aspect_size else QSizePolicy.Policy.Expanding
        self.setSizePolicy(QSizePolicy.Policy.Expanding, vertical_policy)
        self.setStyleSheet(preview_surface_stylesheet())
        self._source_pixmap: QPixmap | None = None
        self._message = "Camera preview is idle."
        self._logged_first_display = False
        self._image_alignment = image_alignment
        self._preserve_aspect_size = preserve_aspect_size
        self._deferred_update_pending = False
        self._deferred_update_timer = QTimer(self)
        self._deferred_update_timer.setSingleShot(True)
        self._deferred_update_timer.timeout.connect(self._run_deferred_update)

    def refresh_theme(self) -> None:
        if not log_if_not_gui_thread("AspectImageLabel.refresh_theme"):
            return
        self.setStyleSheet(preview_surface_stylesheet())
        self.update()

    def set_frame_pixmap(self, pixmap: QPixmap) -> None:
        if not log_if_not_gui_thread("AspectImageLabel.set_frame_pixmap"):
            return
        if not getattr(self, "_logged_first_source_set", False):
            camera_log("first preview source pixmap set")
            self._logged_first_source_set = True
        self._source_pixmap = pixmap
        self._message = ""
        self.updateGeometry()
        self.update()
        self._defer_update()

    def clear_preview(self, message: str = "Camera preview is idle.") -> None:
        if not log_if_not_gui_thread("AspectImageLabel.clear_preview"):
            return
        self._source_pixmap = None
        self._message = message
        self.updateGeometry()
        self.update()

    def resizeEvent(self, event) -> None:  # type: ignore[override]
        super().resizeEvent(event)
        if self._source_pixmap is not None and not self._source_pixmap.isNull():
            self.update()

    def hasHeightForWidth(self) -> bool:  # type: ignore[override]
        return self._preserve_aspect_size

    def heightForWidth(self, width: int) -> int:  # type: ignore[override]
        if not self._preserve_aspect_size:
            return super().heightForWidth(width)
        if width <= 0:
            return self.minimumHeight()
        if self._source_pixmap is not None and not self._source_pixmap.isNull():
            height = round(width * self._source_pixmap.height() / self._source_pixmap.width())
        else:
            height = round(width * 9 / 16)
        return max(self.minimumHeight(), height)

    def paintEvent(self, event) -> None:  # type: ignore[override]
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
        rect = self.rect()
        painter.fillRect(rect, self.palette().window())
        if rect.width() <= 0 or rect.height() <= 0:
            return
        if self._source_pixmap is None or self._source_pixmap.isNull():
            painter.setPen(Qt.GlobalColor.lightGray)
            text = _camera_empty_state_text(self._message)
            icon_name = "processing" if text.lower().startswith("starting") else "camera"
            pixmap = icon_pixmap(state_icon(icon_name), 24)
            if pixmap.isNull():
                painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, text)
            else:
                text_rect = painter.fontMetrics().boundingRect(rect, Qt.AlignmentFlag.AlignCenter, text)
                painter.drawPixmap(
                    rect.center().x() - pixmap.width() // 2,
                    text_rect.top() - pixmap.height() - 8,
                    pixmap,
                )
                painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, text)
            return
        if not getattr(self, "_logged_first_display", False):
            camera_log("first preview display update")
            self._logged_first_display = True
        scaled = self._source_pixmap.scaled(
            rect.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        if scaled.isNull():
            self._defer_update()
            return
        x = rect.x() + (rect.width() - scaled.width()) // 2
        if self._image_alignment & Qt.AlignmentFlag.AlignTop:
            y = rect.y()
        elif self._image_alignment & Qt.AlignmentFlag.AlignBottom:
            y = rect.y() + rect.height() - scaled.height()
        else:
            y = rect.y() + (rect.height() - scaled.height()) // 2
        painter.drawPixmap(x, y, scaled)

    def _defer_update(self) -> None:
        if not log_if_not_gui_thread("AspectImageLabel._defer_update"):
            return
        if self._deferred_update_pending:
            return
        self._deferred_update_pending = True
        self._deferred_update_timer.start(0)

    def _run_deferred_update(self) -> None:
        if not log_if_not_gui_thread("AspectImageLabel._run_deferred_update"):
            return
        self._deferred_update_pending = False
        self.update()


def _camera_empty_state_text(message: str) -> str:
    text = message.strip() or "Camera preview is idle."
    return text


class ReferenceCapturePage(QWidget):
    def __init__(
        self,
        state: AppState,
        status_callback: StatusCallback | None = None,
        reference_saved_callback: ReferenceSavedCallback | None = None,
    ) -> None:
        super().__init__()
        self.state = state
        self._status_callback = status_callback
        self._reference_saved_callback = reference_saved_callback
        self._thread: QThread | None = None
        self._worker: CameraWorker | None = None
        self._latest_frame: np.ndarray | None = None
        self._captured_frame: np.ndarray | None = None
        self._state = "idle"
        self._last_actual_resolution: tuple[int, int] | None = None
        self._camera_lifecycle = "stopped"
        self._camera_session_id = 0
        self._startup_frame_count = 0
        self._blank_startup_frame_count = 0
        self._first_valid_frame_seen = False

        self._build_ui()
        self.refresh_from_state()
        self._set_state("idle")
        self._set_feedback("Set target, start camera, capture reference.", ok=True)

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        spacing = theme_spacing()
        layout.setContentsMargins(*page_margins())
        layout.setSpacing(spacing.setup_page_gap)
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        self.preview_label = AspectImageLabel()
        self.preview_label.setMinimumHeight(theme_dimensions().reference_preview_min_height)
        self.feedback_label = StatusBanner()

        layout.addWidget(self._capture_workspace(), 1)

    def _capture_workspace(self) -> QSplitter:
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setObjectName("referenceCaptureWorkspace")
        splitter.setChildrenCollapsible(False)
        splitter.addWidget(self._control_pane_scroll())
        splitter.addWidget(self._preview_group())
        dimensions = theme_dimensions()
        splitter.setSizes([dimensions.camera_splitter_left_width, dimensions.camera_splitter_right_width])
        splitter.setStretchFactor(0, dimensions.camera_splitter_left_stretch)
        splitter.setStretchFactor(1, dimensions.camera_splitter_right_stretch)
        return splitter

    def _control_pane_scroll(self) -> ScrollablePane:
        return ScrollablePane(self._control_pane(), object_name="referenceCaptureControlPaneScroll")

    def _control_pane(self) -> QWidget:
        widget = QWidget()
        widget.setObjectName("referenceCaptureControlPane")
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(*zero_margins())
        layout.setSpacing(theme_spacing().left_pane_gap)
        layout.addWidget(self._settings_group())
        layout.addWidget(self._actions_group())
        layout.addWidget(self._save_target_group())
        layout.addStretch(1)
        return widget

    def _preview_group(self) -> SectionPanel:
        panel = SectionPanel("Reference Preview", compact=True)
        panel.setMinimumHeight(theme_dimensions().reference_preview_min_height)
        panel.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        panel.content_layout.addWidget(self.preview_label, 1)
        return panel

    def _settings_group(self) -> SectionPanel:
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

    def _actions_group(self) -> SectionPanel:
        panel = SectionPanel("Capture", compact=True)
        self.capture_state_banner = StatusBanner()
        panel.content_layout.addWidget(self.capture_state_banner)

        self.start_button = QPushButton("Start Camera")
        set_button_icon(self.start_button, "start_camera")
        self.start_button.clicked.connect(self.start_camera)
        self.stop_button = QPushButton("Stop Camera")
        self.stop_button.clicked.connect(self.stop_camera)
        self.capture_button = QPushButton("Capture Background")
        set_button_icon(self.capture_button, "capture")
        self.capture_button.clicked.connect(self.capture_background)
        self.save_button = QPushButton("Use as Reference")
        set_button_icon(self.save_button, "save")
        self.save_button.clicked.connect(self.use_as_reference)
        self.retake_button = QPushButton("Retake")
        self.retake_button.clicked.connect(self.retake)

        for button in (self.start_button, self.stop_button, self.capture_button, self.save_button, self.retake_button):
            set_button_role(button, "secondary")
        panel.content_layout.addWidget(ActionButtonRow((self.start_button, self.capture_button, self.save_button, self.retake_button)))
        panel.content_layout.addWidget(ActionButtonRow((self.stop_button,)))
        panel.content_layout.addWidget(self.feedback_label)
        return panel

    def _save_target_group(self) -> SectionPanel:
        panel = SectionPanel("Save Target", compact=True)
        self.output_path_edit = QLineEdit()
        self.output_path_edit.setMinimumWidth(0)
        browse_button = QPushButton("Choose Save Path...")
        set_button_icon(browse_button, "browse")
        browse_button.clicked.connect(self.choose_output_path)
        path_row = QWidget()
        path_layout = QHBoxLayout(path_row)
        path_layout.setContentsMargins(*zero_margins())
        path_layout.addWidget(self.output_path_edit, 1)
        path_layout.addWidget(browse_button)
        panel.content_layout.addWidget(path_row)
        return panel

    def refresh_from_state(self) -> None:
        current = self.output_path_edit.text().strip()
        if current and not self.state.reference_path_auto:
            return
        self.output_path_edit.setText(str(default_reference_output_path(self.state)))

    def refresh_theme(self) -> None:
        self.preview_label.refresh_theme()

    def choose_output_path(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save reference image",
            self.output_path_edit.text().strip() or "data/reference/empty_reference.png",
            "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff);;All files (*)",
        )
        if path:
            try:
                self.output_path_edit.setText(str(normalize_reference_output_path(path)))
            except Exception as exc:
                self._set_feedback(str(exc), ok=False)

    def start_camera(self) -> None:
        if self._camera_lifecycle in {"starting", "running", "stopping"} or self._thread is not None:
            camera_log(
                f"reference capture start ignored; lifecycle={self._camera_lifecycle}; "
                f"session={self._camera_session_id}"
            )
            return
        self._camera_session_id += 1
        session_id = self._camera_session_id
        self._camera_lifecycle = "starting"
        self._startup_frame_count = 0
        self._blank_startup_frame_count = 0
        self._first_valid_frame_seen = False
        self._latest_frame = None
        self._captured_frame = None
        self._last_actual_resolution = None
        setattr(frame_to_pixmap, "_logged_first", False)
        self._logged_first_frame_ready = False
        self.preview_label._logged_first_display = False
        self.preview_label._logged_first_source_set = False
        self._set_state("live_preview")
        self.preview_label.clear_preview("Starting camera\nWaiting for the first live frame.")
        self._set_feedback("Starting camera...", ok=True)
        camera_log(f"start camera clicked; creating camera thread; session={session_id}")

        self._thread = QThread(self)
        self._worker = CameraWorker(
            self.camera_index_spin.value(),
            self.width_spin.value(),
            self.height_spin.value(),
            session_id=session_id,
        )
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.frame_ready.connect(self._on_frame_ready_from_worker, Qt.ConnectionType.QueuedConnection)
        self._worker.actual_resolution.connect(
            self._on_actual_resolution_from_worker,
            Qt.ConnectionType.QueuedConnection,
        )
        self._worker.status.connect(self._on_camera_status_from_worker, Qt.ConnectionType.QueuedConnection)
        self._worker.error.connect(self._on_camera_error_from_worker, Qt.ConnectionType.QueuedConnection)
        self._worker.finished.connect(self._on_worker_finished_from_worker, Qt.ConnectionType.QueuedConnection)
        self._worker.finished.connect(self._thread.quit)
        self._worker.finished.connect(self._worker.deleteLater)
        self._thread.finished.connect(self._thread.deleteLater)
        self._thread.start()

    def stop_camera(self) -> None:
        if self._camera_lifecycle in {"stopped", "stopping"} and self._thread is None:
            return
        stop_session_id = self._camera_session_id
        self._camera_lifecycle = "stopping"
        camera_log(f"reference capture stop requested; session={stop_session_id}")
        stopped = self._stop_worker(stop_session_id)
        if not stopped:
            self._set_feedback("Stopping camera. Waiting for camera worker to finish.", ok=True)
            self._set_state("live_preview")
            return
        self._camera_lifecycle = "stopped"
        self._captured_frame = None
        self._set_state("idle")
        self.preview_label.clear_preview()
        self._set_feedback("Camera stopped.", ok=True)

    def capture_background(self) -> None:
        if self._latest_frame is None:
            self._set_feedback("No camera frame is available to capture yet.", ok=False)
            return
        self._captured_frame = self._latest_frame.copy()
        self.preview_label.set_frame_pixmap(frame_to_pixmap(self._captured_frame))
        self._set_state("captured_preview")
        self._set_feedback("Reference frame captured.", ok=True)

    def retake(self) -> None:
        self._captured_frame = None
        if self._camera_lifecycle == "stopping":
            self._set_feedback("Camera is stopping. Start again after it has stopped.", ok=False)
            return
        self._set_state("live_preview")
        self._set_feedback("Live preview resumed.", ok=True)

    def use_as_reference(self) -> None:
        try:
            saved_path, (width, height) = save_captured_reference_frame(self._captured_frame, self.output_path_edit.text())
            self.output_path_edit.setText(str(saved_path))
            self.state.reference_image_path = saved_path
            self.state.reference_path_auto = saved_path == job_reference_image_path(self.state.inspection_job_slug)
            if self._reference_saved_callback:
                self._reference_saved_callback(saved_path)
            self._set_state("saved")
            self._set_feedback(
                f"Saved reference image to {saved_path} ({width}x{height}). Continue to Draw Zones.",
                ok=True,
            )
        except Exception as exc:
            self._set_feedback(f"Could not save reference image: {exc}", ok=False)

    def shutdown(self) -> None:
        self._camera_lifecycle = "stopping"
        self._stop_worker(self._camera_session_id)

    def _on_frame_ready(self, frame: object) -> None:
        self._on_frame_ready_for_session(self._camera_session_id, frame)

    @Slot(int, object)
    def _on_frame_ready_from_worker(self, session_id: int, frame: object) -> None:
        self._on_frame_ready_for_session(session_id, frame)

    def _on_frame_ready_for_session(self, session_id: int, frame: object) -> None:
        if not log_if_not_gui_thread("ReferenceCapturePage._on_frame_ready_for_session"):
            return
        if session_id != self._camera_session_id:
            camera_log(
                f"ignored stale reference frame; signal_session={session_id}; "
                f"current_session={self._camera_session_id}; lifecycle={self._camera_lifecycle}"
            )
            return
        if self._camera_lifecycle not in {"starting", "running"}:
            camera_log(f"ignored reference frame while lifecycle={self._camera_lifecycle}; session={session_id}")
            return
        if not isinstance(frame, np.ndarray):
            return
        if not self._first_valid_frame_seen:
            self._startup_frame_count += 1
            blank_startup = is_blank_startup_frame(frame)
            if self._startup_frame_count <= BLANK_STARTUP_FRAME_LOG_LIMIT:
                camera_log(
                    "reference startup frame stats; "
                    f"session={session_id}; lifecycle={self._camera_lifecycle}; "
                    f"frame_index={self._startup_frame_count}; {_frame_diagnostics(frame)}; "
                    f"blank_startup={blank_startup}"
                )
            if blank_startup:
                self._blank_startup_frame_count += 1
                self._latest_frame = None
                self._set_state("live_preview")
                if self._blank_startup_frame_count >= STARTUP_BLANK_FINAL_FAIL_FRAMES:
                    message = "Camera opened, but frames are blank. Check lens, exposure, lighting, or camera driver."
                    self.capture_state_banner.set_state("Blank frames", message, "blocked", "warning")
                    self._set_feedback(message, ok=False)
                    camera_log(f"reference blank startup warning; session={session_id}; blank_frames={self._blank_startup_frame_count}")
                else:
                    self.capture_state_banner.set_state(
                        "Warming up camera",
                        "Ignoring blank startup frame.",
                        "processing",
                        "info",
                    )
                    self._set_feedback("Camera warming up. Waiting for a non-blank frame.", ok=True)
                return
            self._first_valid_frame_seen = True
            camera_log(
                f"first valid reference frame; session={session_id}; "
                f"startup_frames={self._startup_frame_count}; blank_frames={self._blank_startup_frame_count}"
            )
        if not getattr(self, "_logged_first_frame_ready", False):
            camera_log(
                "first _on_frame_ready in UI thread; "
                f"session={session_id}; lifecycle={self._camera_lifecycle}; {_frame_diagnostics(frame)}"
            )
            self._logged_first_frame_ready = True
        self._camera_lifecycle = "running"
        self._latest_frame = frame
        if self._state == "live_preview":
            self._set_state(self._state)
        if self._state == "live_preview":
            self.preview_label.set_frame_pixmap(frame_to_pixmap(frame))
            self._update_capture_state_banner(self._state)

    def _on_actual_resolution(self, width: int, height: int) -> None:
        self._on_actual_resolution_for_session(self._camera_session_id, width, height)

    @Slot(int, int, int)
    def _on_actual_resolution_from_worker(self, session_id: int, width: int, height: int) -> None:
        self._on_actual_resolution_for_session(session_id, width, height)

    def _on_actual_resolution_for_session(self, session_id: int, width: int, height: int) -> None:
        if not log_if_not_gui_thread("ReferenceCapturePage._on_actual_resolution_for_session"):
            return
        if session_id != self._camera_session_id:
            camera_log(
                f"ignored stale reference resolution; signal_session={session_id}; "
                f"current_session={self._camera_session_id}"
            )
            return
        self._last_actual_resolution = (width, height)
        requested_width = self.width_spin.value()
        requested_height = self.height_spin.value()
        if requested_width and requested_height and (width != requested_width or height != requested_height):
            self._set_feedback(
                f"Camera started. Requested {requested_width}x{requested_height}; actual frame size is {width}x{height}.",
                ok=True,
            )
        else:
            self._set_feedback(f"Camera started. Actual frame size is {width}x{height}.", ok=True)

    @Slot(int, str, str)
    def _on_camera_status_from_worker(self, session_id: int, message: str, level: str) -> None:
        if not log_if_not_gui_thread("ReferenceCapturePage._on_camera_status_from_worker"):
            return
        if session_id != self._camera_session_id:
            camera_log(
                f"ignored stale reference status; signal_session={session_id}; "
                f"current_session={self._camera_session_id}; message={message}"
            )
            return
        banner_level = "warning" if level == "warning" else "info"
        title = "Reopening camera" if level == "warning" else "Warming up camera"
        state = "blocked" if level == "warning" else "processing"
        self.capture_state_banner.set_state(title, message, state, banner_level)
        self._set_feedback(message, ok=level != "error")

    def _on_camera_error(self, message: str) -> None:
        self._on_camera_error_for_session(self._camera_session_id, message)

    @Slot(int, str)
    def _on_camera_error_from_worker(self, session_id: int, message: str) -> None:
        self._on_camera_error_for_session(session_id, message)

    def _on_camera_error_for_session(self, session_id: int, message: str) -> None:
        if not log_if_not_gui_thread("ReferenceCapturePage._on_camera_error_for_session"):
            return
        if session_id != self._camera_session_id:
            camera_log(
                f"ignored stale reference error; signal_session={session_id}; "
                f"current_session={self._camera_session_id}; message={message}"
            )
            return
        camera_log(f"camera error shown to UI: {message}")
        self._camera_lifecycle = "stopped"
        self._set_feedback(message, ok=False)
        self._set_state("idle")

    def _on_worker_finished(self) -> None:
        self._on_worker_finished_for_session(self._camera_session_id)

    @Slot(int)
    def _on_worker_finished_from_worker(self, session_id: int) -> None:
        self._on_worker_finished_for_session(session_id)

    def _on_worker_finished_for_session(self, session_id: int) -> None:
        if not log_if_not_gui_thread("ReferenceCapturePage._on_worker_finished_for_session"):
            return
        if session_id != self._camera_session_id:
            camera_log(
                f"ignored stale reference finished; signal_session={session_id}; "
                f"current_session={self._camera_session_id}; lifecycle={self._camera_lifecycle}"
            )
            return
        camera_log(f"camera worker finished; session={session_id}")
        self._thread = None
        self._worker = None
        self._camera_lifecycle = "stopped"
        if self._state == "live_preview":
            self._set_state("idle")
        else:
            self._set_state(self._state)

    def _stop_worker(self, session_id: int | None = None) -> bool:
        worker = self._worker
        thread = self._thread
        if worker is not None:
            try:
                if shiboken6.isValid(worker):
                    camera_log(f"requesting camera worker stop; session={session_id}")
                    worker.stop()
            except RuntimeError:
                pass
        if thread is not None:
            try:
                if shiboken6.isValid(thread):
                    camera_log(f"requesting camera thread quit; session={session_id}")
                    thread.quit()
                    if not thread.wait(2500):
                        camera_log(f"camera thread did not stop within 2500 ms; session={session_id}")
                        return False
            except RuntimeError:
                pass
        self._thread = None
        self._worker = None
        return True

    def _set_state(self, state: str) -> None:
        self._state = state
        idle = state == "idle"
        live = state == "live_preview"
        captured = state == "captured_preview"
        saved = state == "saved"
        camera_running = self._camera_lifecycle in {"starting", "running", "stopping"} or self._thread is not None
        camera_stopping = self._camera_lifecycle == "stopping"
        self.start_button.setEnabled((idle or saved) and not camera_running)
        self.stop_button.setEnabled((live or captured or saved) and camera_running and not camera_stopping)
        self.capture_button.setEnabled(live and self._camera_lifecycle == "running")
        self.save_button.setEnabled(captured)
        self.retake_button.setEnabled(captured or (saved and camera_running))
        for widget in (self.camera_index_spin, self.width_spin, self.height_spin):
            widget.setEnabled((idle or saved) and not camera_running)
        for button in (self.start_button, self.stop_button, self.capture_button, self.save_button, self.retake_button):
            set_button_role(button, "secondary")
        if idle or saved:
            set_button_role(self.start_button, "primary")
        elif live:
            set_button_role(self.capture_button, "primary")
        elif captured:
            set_button_role(self.save_button, "primary")
        self._update_capture_state_banner(state)

    def _update_capture_state_banner(self, state: str) -> None:
        if state == "idle":
            self.capture_state_banner.set_state("Idle", "Start camera to preview the empty station.", "idle", "info")
        elif state == "live_preview":
            if self._latest_frame is None:
                self.capture_state_banner.set_state("Starting camera", "Waiting for first frame.", "processing", "info")
            else:
                self.capture_state_banner.set_state("Live preview", "Capture when the station is empty.", "live", "info")
        elif state == "captured_preview":
            self.capture_state_banner.set_state("Captured", "Review the frame, then save it as the reference.", "captured", "success")
        elif state == "saved":
            self.capture_state_banner.set_state("Saved", "Reference image is ready for Draw Zones.", "saved", "success")
        else:
            self.capture_state_banner.set_state("Error", "Capture state is unavailable.", "error", "error")

    def _set_feedback(self, message: str, ok: bool) -> None:
        self.feedback_label.set_message(message, "success" if ok else "error")
        self.state.status_message = message
        if self._status_callback:
            self._status_callback(message)
