from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from threading import Lock, RLock

from anomaly_inspection.core.config import InspectionConfig, load_config
from anomaly_inspection.core.pipeline import InspectionPipeline
from anomaly_inspection.core.result_types import InspectionResult


RUNTIME_NOT_PREPARED = "not_prepared"
RUNTIME_PREPARING = "preparing"
RUNTIME_READY = "ready"
RUNTIME_NEEDS_REFRESH = "needs_refresh"
RUNTIME_ERROR = "error"


@dataclass(frozen=True)
class RuntimeFingerprint:
    config_path: Path
    mtime_ns: int
    size: int


def config_for_desktop_runtime(config: InspectionConfig) -> InspectionConfig:
    from dataclasses import replace

    return replace(config, output=replace(config.output, show_images=False))


def runtime_fingerprint(config_path: str | Path) -> RuntimeFingerprint:
    path = Path(config_path).resolve()
    stat = path.stat()
    return RuntimeFingerprint(path, stat.st_mtime_ns, stat.st_size)


class PreparedRuntimeManager:
    def __init__(self) -> None:
        self._state = RUNTIME_NOT_PREPARED
        self._error_message = ""
        self._fingerprint: RuntimeFingerprint | None = None
        self._pipeline: InspectionPipeline | None = None
        self._state_lock = RLock()
        self._inspection_lock = Lock()

    @property
    def state(self) -> str:
        with self._state_lock:
            return self._state

    @property
    def error_message(self) -> str:
        with self._state_lock:
            return self._error_message

    def mark_preparing(self) -> None:
        with self._state_lock:
            self._state = RUNTIME_PREPARING
            self._error_message = ""

    def mark_stale(self) -> None:
        with self._state_lock:
            if self._state in {RUNTIME_READY, RUNTIME_ERROR, RUNTIME_PREPARING} or self._pipeline is not None:
                self._state = RUNTIME_NEEDS_REFRESH
                self._error_message = ""

    def clear(self) -> None:
        with self._state_lock:
            self._state = RUNTIME_NOT_PREPARED
            self._error_message = ""
            self._fingerprint = None
            self._pipeline = None

    def prepare(self, config_path: str | Path) -> None:
        fingerprint = runtime_fingerprint(config_path)
        config = config_for_desktop_runtime(load_config(fingerprint.config_path))
        pipeline = InspectionPipeline(config)
        pipeline.prepare_anomaly_backend()
        with self._state_lock:
            self._fingerprint = fingerprint
            self._pipeline = pipeline
            self._state = RUNTIME_READY
            self._error_message = ""

    def set_error(self, message: str) -> None:
        with self._state_lock:
            self._state = RUNTIME_ERROR
            self._error_message = message

    def is_current(self, config_path: str | Path) -> bool:
        try:
            fingerprint = runtime_fingerprint(config_path)
        except OSError:
            return False
        with self._state_lock:
            return self._state == RUNTIME_READY and self._fingerprint == fingerprint and self._pipeline is not None

    def inspect_image(
        self,
        config_path: str | Path,
        image_path: str | Path,
        output_dir: str | Path,
        *,
        inspection_mode: str = "image",
    ) -> tuple[InspectionResult, bool]:
        pipeline = self._current_pipeline(config_path)
        if pipeline is None:
            config = config_for_desktop_runtime(load_config(config_path))
            pipeline = InspectionPipeline(config)
            return pipeline.inspect_image(image_path, output_dir, inspection_mode=inspection_mode), False
        with self._inspection_lock:
            return pipeline.inspect_image(image_path, output_dir, inspection_mode=inspection_mode), True

    def _current_pipeline(self, config_path: str | Path) -> InspectionPipeline | None:
        try:
            fingerprint = runtime_fingerprint(config_path)
        except OSError:
            return None
        with self._state_lock:
            if self._state != RUNTIME_READY or self._fingerprint != fingerprint:
                return None
            return self._pipeline
