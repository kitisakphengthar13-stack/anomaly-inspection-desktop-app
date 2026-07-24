from __future__ import annotations

from pathlib import Path
import tempfile
from typing import Any, NamedTuple, Optional
import warnings

import cv2
import numpy as np

from anomaly_inspection.core.result_types import AnomalyResult
from anomaly_inspection.core.visualization import try_save_heatmap
from anomaly_inspection.core.anomalib_model_resolver import AnomalibModelResolver


class BaseAnomalyBackend:
    backend_name = "base"

    def __init__(self, model_path: str | Path, device: str = "auto"):
        self.model_path = Path(model_path)
        self.device = resolve_device(device)

    def load(self) -> None:
        raise NotImplementedError

    def predict(self, image_bgr: np.ndarray) -> Any:
        raise NotImplementedError


class ExportedTorchBackend(BaseAnomalyBackend):
    """Backend for Anomalib exported Torch `.pt` artifacts."""

    backend_name = "exported_torch"

    def __init__(self, model_path: str | Path, device: str = "auto"):
        super().__init__(model_path, device)
        self.inferencer: Any = None

    def load(self) -> None:
        from anomalib.deploy import TorchInferencer

        self.inferencer = TorchInferencer(path=str(self.model_path), device=self.device)

    def predict(self, image_bgr: np.ndarray) -> Any:
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        try:
            return self.inferencer.predict(image=image_rgb)
        except TypeError:
            return self.inferencer.predict(image_rgb)


class LightningCheckpointBackend(BaseAnomalyBackend):
    """Backend for Anomalib Lightning `.ckpt` checkpoints.

    Verified with Anomalib PatchCore Lightning checkpoints. PyTorch 2.6+
    requires `weights_only=False` for trusted local Anomalib checkpoints that
    contain custom metadata such as enums.
    """

    backend_name = "lightning_checkpoint"

    def __init__(
        self,
        model_path: str | Path,
        device: str = "auto",
        *,
        model_class: Any | None = None,
        model_name: str | None = None,
        model_source: str | None = None,
    ):
        super().__init__(model_path, device)
        self.model: Any = None
        self.model_class = model_class
        self.model_name = model_name or "unknown"
        self.model_source = model_source or "unknown"

    def load(self) -> None:
        if self.model_class is None:
            resolved = AnomalibModelResolver().resolve_for_checkpoint(self.model_path, None)
            self.model_class = resolved.model_class
            self.model_name = resolved.name
            self.model_source = resolved.source
        try:
            self.model = self.model_class.load_from_checkpoint(str(self.model_path), weights_only=False)
        except Exception as exc:
            if self.model_source == "legacy_default":
                raise RuntimeError(
                    "Lightning checkpoint loading failed after using the legacy PatchCore fallback. "
                    "No model.anomalib_model was set, so the checkpoint was loaded as PatchCore. "
                    "If this checkpoint is not PatchCore, set model.anomalib_model explicitly "
                    "(for example, reverse_distillation). "
                    f"Original error: {exc}"
                ) from exc
            raise
        self.model.eval()
        self.model.to(self.device)

    def predict(self, image_bgr: np.ndarray) -> Any:
        import torch

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(image_rgb).float().permute(2, 0, 1).unsqueeze(0) / 255.0
        tensor = tensor.to(self.device)
        with torch.inference_mode():
            try:
                return self.model(tensor)
            except TypeError:
                return self.model({"image": tensor})


class EngineCheckpointBackend(BaseAnomalyBackend):
    """Experimental Anomalib Engine-based backend for Lightning `.ckpt` checkpoints."""

    backend_name = "engine_checkpoint"

    def __init__(
        self,
        model_path: str | Path,
        device: str = "auto",
        *,
        model_class: Any,
        model_name: str,
        model_source: str,
    ):
        super().__init__(model_path, device)
        self.model: Any = None
        self.engine: Any = None
        self._engine_output_dir: tempfile.TemporaryDirectory | None = None
        self.model_class = model_class
        self.model_name = model_name
        self.model_source = model_source

    def load(self) -> None:
        from anomalib.engine import Engine

        try:
            self.model = self.model_class.load_from_checkpoint(str(self.model_path), weights_only=False)
        except Exception as exc:
            if self.model_source == "legacy_default":
                raise RuntimeError(
                    "Engine checkpoint loading failed after using the legacy PatchCore fallback. "
                    "No model.anomalib_model was set, so the checkpoint was loaded as PatchCore. "
                    "If this checkpoint is not PatchCore, set model.anomalib_model explicitly "
                    "(for example, reverse_distillation). "
                    f"Original error: {exc}"
                ) from exc
            raise
        self.model.eval()
        self.model.to(self.device)
        self._engine_output_dir = tempfile.TemporaryDirectory(prefix="anomalib_engine_")
        self.engine = Engine(
            accelerator=engine_accelerator(self.device),
            devices=1,
            logger=False,
            default_root_dir=self._engine_output_dir.name,
            enable_progress_bar=False,
        )

    def predict(self, image_bgr: np.ndarray) -> Any:
        if self.model is None or self.engine is None:
            raise RuntimeError("Engine checkpoint backend is not loaded.")
        temp_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
                temp_path = Path(temp_file.name)
            if not cv2.imwrite(str(temp_path), image_bgr):
                raise RuntimeError("Could not create temporary image for Anomalib Engine prediction.")
            predictions = self.engine.predict(model=self.model, data_path=temp_path, return_predictions=True)
        finally:
            if temp_path is not None:
                temp_path.unlink(missing_ok=True)
        if not predictions:
            raise RuntimeError("Anomalib Engine did not return a prediction.")
        return predictions[0]


class OpenVINOBackend(BaseAnomalyBackend):
    """Backend for Anomalib OpenVINO `.xml` exports."""

    backend_name = "openvino"

    def __init__(self, model_path: str | Path, device: str = "auto"):
        self.model_path = Path(model_path)
        self.device = resolve_openvino_device(device)
        self.core: Any = None
        self.compiled_model: Any = None
        self.input_name: Optional[str] = None
        self.input_size: Optional[tuple[int, int]] = None
        self.input_shape: str = "unknown"
        self.output_names = ("pred_score", "pred_label", "anomaly_map", "pred_mask")

    def load(self) -> None:
        if self.model_path.suffix.lower() == ".xml":
            weights_path = self.model_path.with_suffix(".bin")
            if not weights_path.exists():
                raise FileNotFoundError(f"OpenVINO weights file does not exist: {weights_path}")

        try:
            from openvino import Core
        except Exception:
            from openvino.runtime import Core

        self.core = Core()
        model = self.core.read_model(str(self.model_path))
        self.compiled_model = self.core.compile_model(model, self.device)
        inputs = list(self.compiled_model.inputs)
        if not inputs:
            raise ValueError(f"OpenVINO model has no inputs: {self.model_path}")
        self._configure_input(inputs[0])

    def predict(self, image_bgr: np.ndarray) -> dict[str, Any]:
        if self.compiled_model is None or self.input_name is None or self.input_size is None:
            raise RuntimeError("OpenVINO backend is not loaded.")
        input_tensor = self.preprocess(image_bgr)
        raw_outputs = self.compiled_model({self.input_name: input_tensor})
        return {name: self._output_value(raw_outputs, name) for name in self.output_names}

    def preprocess(self, image_bgr: np.ndarray) -> np.ndarray:
        if self.input_size is None:
            raise RuntimeError("OpenVINO backend input shape has not been configured.")
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        image_rgb = cv2.resize(image_rgb, self.input_size, interpolation=cv2.INTER_AREA)
        tensor = image_rgb.astype(np.float32) / 255.0
        tensor = np.transpose(tensor, (2, 0, 1))
        return np.expand_dims(tensor, axis=0)

    def _configure_input(self, input_port: Any) -> None:
        self.input_name = input_any_name(input_port)
        shape = openvino_shape_values(input_port)
        self.input_shape = shape.representation
        self.input_size = openvino_nchw_input_size(self.model_path, shape)

    def _output_value(self, raw_outputs: Any, output_name: str) -> Any:
        if isinstance(raw_outputs, dict):
            if output_name in raw_outputs:
                return raw_outputs[output_name]
            for key, value in raw_outputs.items():
                if input_any_name(key) == output_name:
                    return value
        if self.compiled_model is not None:
            for output in self.compiled_model.outputs:
                if input_any_name(output) == output_name:
                    try:
                        return raw_outputs[output]
                    except Exception:
                        return None
        return None


class AnomalyInferencer:
    """Model-agnostic Anomalib inference facade used by the pipeline."""

    def __init__(
        self,
        model_path: str | Path,
        anomaly_threshold: float = 0.5,
        device: str = "auto",
        model_format: str = "auto",
        anomalib_model: str | None = None,
        checkpoint_inference_mode: str = "engine",
    ):
        self.model_path = Path(model_path)
        self.anomaly_threshold = float(anomaly_threshold)
        self.requested_device = device
        self.device = resolve_device(device)
        self.model_format = model_format
        self.anomalib_model = anomalib_model
        self.checkpoint_inference_mode = normalize_checkpoint_inference_mode(checkpoint_inference_mode)
        self.backend: Optional[BaseAnomalyBackend] = None
        self.backend_name: Optional[str] = None
        self._loaded = False

    def load(self) -> None:
        if self._loaded:
            return
        if not self.model_path.exists():
            raise FileNotFoundError(f"Anomaly model does not exist: {self.model_path}")
        self.backend = self._build_backend()
        self.backend.load()
        self.backend_name = self.backend.backend_name
        self._loaded = True

    def _build_backend(self) -> BaseAnomalyBackend:
        model_format = detect_model_format(self.model_path, self.model_format)
        if model_format == "torch_export":
            return ExportedTorchBackend(self.model_path, self.device)
        if model_format == "ckpt":
            resolved = AnomalibModelResolver().resolve_for_checkpoint(self.model_path, self.anomalib_model)
            if self.checkpoint_inference_mode == "direct":
                warnings.warn(
                    "model.checkpoint_inference_mode='direct' uses the legacy/debug Lightning checkpoint path. "
                    "The official default for .ckpt artifacts is Anomalib Engine inference.",
                    RuntimeWarning,
                    stacklevel=2,
                )
            backend_class = (
                LightningCheckpointBackend
                if self.checkpoint_inference_mode == "direct"
                else EngineCheckpointBackend
            )
            return backend_class(
                self.model_path,
                self.device,
                model_class=resolved.model_class,
                model_name=resolved.name,
                model_source=resolved.source,
            )
        if model_format == "openvino":
            return OpenVINOBackend(self.model_path, self.requested_device)
        raise ValueError(f"Unsupported model format '{model_format}'.")

    def predict_image_path(self, image_path: str | Path, heatmap_path: Optional[str | Path] = None) -> AnomalyResult:
        self.load()
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            return AnomalyResult(
                backend_name=self.backend_name,
                model_path=str(self.model_path),
                error_message=f"Could not read image for anomaly inference: {image_path}",
            )
        return self.predict(image, heatmap_path=heatmap_path)

    def predict(self, image_bgr: np.ndarray, heatmap_path: Optional[str | Path] = None) -> AnomalyResult:
        self.load()
        try:
            prediction = self.backend.predict(image_bgr) if self.backend else None
            score = extract_score(prediction)
            pred_label = extract_pred_label(prediction)
            heatmap = extract_heatmap(prediction)
            saved_heatmap = try_save_heatmap(image_bgr, heatmap, heatmap_path) if heatmap_path else None
            if score is None:
                return AnomalyResult(
                    pred_label=pred_label,
                    backend_name=self.backend_name,
                    model_path=str(self.model_path),
                    heatmap_path=saved_heatmap,
                    heatmap=heatmap,
                    error_message="Anomalib prediction did not expose a usable anomaly score.",
                )
            return AnomalyResult(
                anomaly_score=float(score),
                pred_label=pred_label,
                backend_name=self.backend_name,
                model_path=str(self.model_path),
                heatmap_path=saved_heatmap,
                heatmap=heatmap,
            )
        except Exception as exc:
            if self.backend_name == EngineCheckpointBackend.backend_name:
                message = f"Engine checkpoint inference failed: {exc}"
            else:
                message = f"Anomaly inference failed: {exc}"
            return AnomalyResult(
                backend_name=self.backend_name,
                model_path=str(self.model_path),
                error_message=message,
            )


def resolve_device(device: str) -> str:
    if device != "auto":
        return device
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def resolve_openvino_device(device: str) -> str:
    normalized = device.strip().lower()
    if normalized in {"", "auto", "cpu"}:
        return "CPU"
    if normalized == "cuda":
        return "GPU"
    return device.strip().upper()


def engine_accelerator(device: str) -> str:
    if device.lower().startswith("cuda"):
        return "gpu"
    if device.lower() == "cpu":
        return "cpu"
    return "auto"


def normalize_checkpoint_inference_mode(mode: str) -> str:
    normalized = str(mode or "engine").strip().lower()
    if normalized not in {"direct", "engine"}:
        raise ValueError("model.checkpoint_inference_mode must be one of: direct, engine.")
    return normalized


def detect_model_format(model_path: str | Path, requested_format: str = "auto") -> str:
    requested_format = requested_format.lower()
    aliases = {
        "pt": "torch_export",
        "torch": "torch_export",
        "torch_export": "torch_export",
        "ckpt": "ckpt",
        "openvino": "openvino",
        "xml": "openvino",
    }
    if requested_format != "auto":
        if requested_format not in aliases:
            raise ValueError("model.format must be one of: auto, ckpt, torch_export, openvino.")
        return aliases[requested_format]

    suffix = Path(model_path).suffix.lower()
    if suffix == ".ckpt":
        return "ckpt"
    if suffix == ".pt":
        return "torch_export"
    if suffix == ".xml":
        return "openvino"
    raise ValueError(f"Could not auto-detect model format from extension '{suffix}'.")


def input_any_name(value: Any) -> str:
    if isinstance(value, str):
        return value
    for attr_name in ("get_any_name", "get_node"):
        attr = getattr(value, attr_name, None)
        if attr is None:
            continue
        try:
            candidate = attr()
        except Exception:
            continue
        if isinstance(candidate, str):
            return candidate
        if candidate is not value:
            nested = input_any_name(candidate)
            if nested:
                return nested
    names = getattr(value, "names", None)
    if names:
        try:
            return next(iter(names))
        except Exception:
            pass
    return str(value)


class OpenVINOInputShape(NamedTuple):
    dimensions: tuple[Optional[int], ...]
    representation: str


def openvino_shape_values(input_port: Any) -> OpenVINOInputShape:
    shape = None
    for attr_name in ("get_partial_shape", "partial_shape", "shape"):
        attr = getattr(input_port, attr_name, None)
        if attr is None:
            continue
        try:
            shape = attr() if callable(attr) else attr
        except Exception:
            continue
        if shape is not None:
            break
    if shape is None:
        return OpenVINOInputShape((), "unknown")

    representation = str(shape)
    try:
        raw_dimensions = tuple(shape)
    except TypeError:
        raw_dimensions = tuple(getattr(shape, "dims", ()))
    return OpenVINOInputShape(tuple(openvino_dimension_value(dim) for dim in raw_dimensions), representation)


def openvino_dimension_value(dimension: Any) -> Optional[int]:
    is_dynamic = getattr(dimension, "is_dynamic", None)
    try:
        if bool(is_dynamic() if callable(is_dynamic) else is_dynamic):
            return None
    except Exception:
        pass

    for attr_name in ("get_length", "get_min_length"):
        attr = getattr(dimension, attr_name, None)
        if attr is None:
            continue
        try:
            value = int(attr())
        except Exception:
            continue
        if value >= 0:
            return value

    text = str(dimension).strip()
    if text in {"?", "-1", ""}:
        return None
    try:
        value = int(dimension)
    except Exception:
        try:
            value = int(text)
        except Exception:
            return None
    return value if value >= 0 else None


def openvino_nchw_input_size(model_path: Path, shape: OpenVINOInputShape) -> tuple[int, int]:
    dimensions = shape.dimensions
    if len(dimensions) != 4:
        raise unsupported_openvino_input_shape_error(model_path, shape.representation, "input rank is not 4")

    _, channels, height, width = dimensions
    if channels != 3:
        reason = f"channel dimension is {channels!r}, expected 3 for NCHW RGB input"
        if dimensions[-1] == 3:
            reason = "input appears to be NHWC; only fixed NCHW image input is supported"
        raise unsupported_openvino_input_shape_error(model_path, shape.representation, reason)
    if height is None or width is None:
        raise unsupported_openvino_input_shape_error(
            model_path,
            shape.representation,
            "height or width is dynamic or unavailable",
        )
    if height <= 0 or width <= 0:
        raise unsupported_openvino_input_shape_error(
            model_path,
            shape.representation,
            f"height and width must be positive, got H={height}, W={width}",
        )
    return (width, height)


def unsupported_openvino_input_shape_error(model_path: Path, detected_shape: str, reason: str) -> ValueError:
    return ValueError(
        "Unsupported OpenVINO model input shape. "
        f"Artifact: {model_path}. "
        f"Detected input shape: {detected_shape}. "
        f"Reason: {reason}. "
        "Use a fixed-shape exported OpenVINO image model with NCHW shape [N, 3, H, W]."
    )


def extract_score(prediction: Any) -> Optional[float]:
    candidates = candidate_values(prediction, ("pred_score", "anomaly_score", "score", "image_score"))
    for value in candidates:
        scalar = to_scalar(value)
        if scalar is not None:
            return scalar
    return None


def extract_pred_label(prediction: Any) -> Optional[bool]:
    candidates = candidate_values(prediction, ("pred_label", "label"))
    for value in candidates:
        scalar = to_scalar(value)
        if scalar is not None:
            return bool(scalar)
    return None


def extract_heatmap(prediction: Any) -> Optional[np.ndarray]:
    candidates = candidate_values(prediction, ("heat_map", "heatmap", "anomaly_map", "pred_mask", "prediction"))
    for value in candidates:
        arr = to_numpy(value)
        if arr is not None and arr.size > 1:
            return arr
    return None


def candidate_values(obj: Any, names: tuple[str, ...]) -> list[Any]:
    values = []
    if isinstance(obj, dict):
        values.extend(obj.get(name) for name in names if name in obj)
    for name in names:
        if hasattr(obj, name):
            values.append(getattr(obj, name))
    if isinstance(obj, (tuple, list)):
        for item in obj:
            values.extend(candidate_values(item, names))
    return values


def to_numpy(value: Any) -> Optional[np.ndarray]:
    if value is None:
        return None
    try:
        import torch

        if isinstance(value, torch.Tensor):
            return value.detach().cpu().numpy()
    except Exception:
        pass
    try:
        return np.asarray(value)
    except Exception:
        return None


def to_scalar(value: Any) -> Optional[float]:
    arr = to_numpy(value)
    if arr is None or arr.size == 0:
        return None
    try:
        return float(np.ravel(arr)[0])
    except Exception:
        return None
