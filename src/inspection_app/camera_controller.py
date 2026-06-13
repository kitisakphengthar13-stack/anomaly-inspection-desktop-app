from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum

from PySide6.QtCore import QObject, QThread, Qt, QTimer, Signal, Slot

from inspection_app.reference_capture_page import (
    CameraWorker,
    camera_log,
)


class CameraControllerState(StrEnum):
    RELEASED = "released"
    OPENING = "opening"
    WARMING = "warming"
    RUNNING = "running"
    SOFT_STOPPED = "soft_stopped"
    RECOVERING = "recovering"
    STOPPING = "stopping"
    FAILED = "failed"


@dataclass(frozen=True)
class CameraStartupPolicy:
    blank_max_value: int = 2
    blank_mean_max: float = 1.0
    blank_recovery_frames: int = 6
    blank_final_fail_frames: int = 6
    max_recovery_attempts: int = 1
    idle_release_timeout_ms: int = 1500
    preview_fps: float = 12.0

    @property
    def preview_interval_seconds(self) -> float:
        if self.preview_fps <= 0:
            return 1.0 / 12.0
        return 1.0 / self.preview_fps


CameraSettingsProvider = Callable[[], tuple[int, int, int]]
CameraWorkerFactory = Callable[..., CameraWorker]


class CameraController(QObject):
    state_changed = Signal(str, str, str)
    frame_ready = Signal(object)
    resolution_changed = Signal(int, int)
    error = Signal(str)
    stopped = Signal()

    def __init__(
        self,
        settings_provider: CameraSettingsProvider,
        *,
        startup_policy: CameraStartupPolicy | None = None,
        worker_factory: CameraWorkerFactory = CameraWorker,
    ) -> None:
        super().__init__()
        self._settings_provider = settings_provider
        self._policy = startup_policy or CameraStartupPolicy()
        self._worker_factory = worker_factory
        self._state = CameraControllerState.RELEASED
        self._desired_running = False
        self._session_id = 0
        self._thread: QThread | None = None
        self._worker: CameraWorker | None = None
        self._has_valid_frame = False
        self._release_after_finish = False

        self._idle_release_timer = QTimer(self)
        self._idle_release_timer.setSingleShot(True)
        self._idle_release_timer.setInterval(self._policy.idle_release_timeout_ms)
        self._idle_release_timer.timeout.connect(self._release_idle_worker)

    def start_preview(self) -> None:
        self._desired_running = True
        if self._state == CameraControllerState.SOFT_STOPPED and self._worker is not None:
            self._idle_release_timer.stop()
            next_state = CameraControllerState.RUNNING if self._has_valid_frame else CameraControllerState.WARMING
            self._set_state(next_state, "Camera preview resumed.", "success")
            return
        if self._worker is not None or self._state in {
            CameraControllerState.OPENING,
            CameraControllerState.WARMING,
            CameraControllerState.RUNNING,
            CameraControllerState.RECOVERING,
            CameraControllerState.STOPPING,
        }:
            camera_log(f"camera controller start coalesced; state={self._state}; session={self._session_id}")
            return
        self._start_worker()

    def stop_preview(self) -> None:
        self._desired_running = False
        if self._state == CameraControllerState.RUNNING and self._worker is not None:
            self._set_state(CameraControllerState.SOFT_STOPPED, "Camera stopped. Start again to resume.", "info")
            self._idle_release_timer.start()
            return
        if self._state == CameraControllerState.SOFT_STOPPED:
            self._idle_release_timer.start()
            return
        if self._state in {CameraControllerState.OPENING, CameraControllerState.WARMING, CameraControllerState.RECOVERING}:
            self._hard_stop_worker("Camera stopped before preview became ready.")
            return
        if self._state == CameraControllerState.RELEASED:
            self.stopped.emit()
            return
        if self._worker is not None:
            self._hard_stop_worker("Camera stopping.")

    def shutdown(self) -> None:
        self._desired_running = False
        self._idle_release_timer.stop()
        self._hard_stop_worker("Camera released.", wait=True)

    def is_running(self) -> bool:
        return self._state == CameraControllerState.RUNNING

    def state(self) -> str:
        return self._state.value

    def has_active_worker(self) -> bool:
        return self._worker is not None

    def session_id(self) -> int:
        return self._session_id

    def _start_worker(self) -> None:
        camera_index, width, height = self._settings_provider()
        self._session_id += 1
        session_id = self._session_id
        self._has_valid_frame = False
        self._release_after_finish = False

        self._thread = QThread(self)
        self._worker = self._worker_factory(
            camera_index,
            width,
            height,
            session_id=session_id,
            blank_recovery_frames=self._policy.blank_recovery_frames,
            blank_final_fail_frames=self._policy.blank_final_fail_frames,
            max_blank_recovery_attempts=self._policy.max_recovery_attempts,
            blank_max_value=self._policy.blank_max_value,
            blank_mean_max=self._policy.blank_mean_max,
            preview_interval_seconds=self._policy.preview_interval_seconds,
        )
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.frame_ready.connect(self._on_worker_frame_ready, Qt.ConnectionType.QueuedConnection)
        self._worker.actual_resolution.connect(self._on_worker_resolution, Qt.ConnectionType.QueuedConnection)
        self._worker.status.connect(self._on_worker_status, Qt.ConnectionType.QueuedConnection)
        self._worker.error.connect(self._on_worker_error, Qt.ConnectionType.QueuedConnection)
        self._worker.finished.connect(self._on_worker_finished, Qt.ConnectionType.QueuedConnection)
        self._worker.finished.connect(self._thread.quit)
        self._worker.finished.connect(self._worker.deleteLater)
        self._thread.finished.connect(self._thread.deleteLater)
        self._thread.start()
        self._set_state(CameraControllerState.OPENING, "Starting camera...", "info")

    def _release_idle_worker(self) -> None:
        if self._state != CameraControllerState.SOFT_STOPPED:
            return
        self._hard_stop_worker("Camera released after idle timeout.")

    def _hard_stop_worker(self, message: str, *, wait: bool = False) -> None:
        self._idle_release_timer.stop()
        worker = self._worker
        thread = self._thread
        if worker is None and thread is None:
            self._set_state(CameraControllerState.RELEASED, message, "info")
            self.stopped.emit()
            return
        self._release_after_finish = True
        self._set_state(CameraControllerState.STOPPING, message, "info")
        if worker is not None:
            try:
                worker.stop()
            except RuntimeError:
                pass
        if thread is not None:
            try:
                thread.quit()
                if wait:
                    thread.wait(2500)
            except RuntimeError:
                pass

    @Slot(int, object)
    def _on_worker_frame_ready(self, session_id: int, frame: object) -> None:
        if session_id != self._session_id:
            camera_log(f"camera controller ignored stale frame; signal_session={session_id}; current_session={self._session_id}")
            return
        self._has_valid_frame = True
        if not self._desired_running:
            return
        if self._state != CameraControllerState.RUNNING:
            self._set_state(CameraControllerState.RUNNING, "Camera live.", "success")
        self.frame_ready.emit(frame)

    @Slot(int, int, int)
    def _on_worker_resolution(self, session_id: int, width: int, height: int) -> None:
        if session_id != self._session_id:
            camera_log(f"camera controller ignored stale resolution; signal_session={session_id}; current_session={self._session_id}")
            return
        self.resolution_changed.emit(width, height)

    @Slot(int, str, str)
    def _on_worker_status(self, session_id: int, message: str, level: str) -> None:
        if session_id != self._session_id:
            camera_log(f"camera controller ignored stale status; signal_session={session_id}; current_session={self._session_id}")
            return
        if not self._desired_running:
            return
        state = CameraControllerState.RECOVERING if level == "warning" else CameraControllerState.WARMING
        self._set_state(state, message, level)

    @Slot(int, str)
    def _on_worker_error(self, session_id: int, message: str) -> None:
        if session_id != self._session_id:
            camera_log(f"camera controller ignored stale error; signal_session={session_id}; current_session={self._session_id}")
            return
        self._desired_running = False
        self._set_state(CameraControllerState.FAILED, message, "error")
        self.error.emit(message)

    @Slot(int)
    def _on_worker_finished(self, session_id: int) -> None:
        if session_id != self._session_id:
            camera_log(f"camera controller ignored stale finished; signal_session={session_id}; current_session={self._session_id}")
            return
        thread = self._thread
        self._worker = None
        self._thread = None
        if thread is not None:
            try:
                thread.quit()
                thread.wait(500)
            except RuntimeError:
                pass
        if self._state == CameraControllerState.FAILED:
            self.stopped.emit()
            return
        self._set_state(CameraControllerState.RELEASED, "Camera released.", "info")
        self.stopped.emit()

    def _set_state(self, state: CameraControllerState, message: str, level: str) -> None:
        self._state = state
        self.state_changed.emit(state.value, message, level)
