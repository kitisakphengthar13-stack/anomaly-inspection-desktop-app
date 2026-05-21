from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np

from inspection.result_types import AnomalyResult
from inspection.visualization import try_save_heatmap


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

    def __init__(self, model_path: str | Path, device: str = "auto"):
        super().__init__(model_path, device)
        self.model: Any = None

    def load(self) -> None:
        model_cls = self._find_lightning_model_class()
        self.model = model_cls.load_from_checkpoint(str(self.model_path), weights_only=False)
        self.model.eval()
        self.model.to(self.device)

    @staticmethod
    def _find_lightning_model_class() -> Any:
        import anomalib.models as models

        for name in ("Patchcore", "PatchCore"):
            if hasattr(models, name):
                return getattr(models, name)
        try:
            from anomalib.models.image.patchcore import Patchcore

            return Patchcore
        except Exception:
            pass
        try:
            from anomalib.models.image.patchcore import PatchCore

            return PatchCore
        except Exception as exc:
            raise ImportError("Could not find an Anomalib PatchCore Lightning model class.") from exc

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


class AnomalyInferencer:
    """Model-agnostic Anomalib inference facade used by the pipeline."""

    def __init__(
        self,
        model_path: str | Path,
        anomaly_threshold: float = 0.5,
        device: str = "auto",
        model_format: str = "auto",
    ):
        self.model_path = Path(model_path)
        self.anomaly_threshold = float(anomaly_threshold)
        self.device = resolve_device(device)
        self.model_format = model_format
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
            return LightningCheckpointBackend(self.model_path, self.device)
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
            return AnomalyResult(
                backend_name=self.backend_name,
                model_path=str(self.model_path),
                error_message=f"Anomaly inference failed: {exc}",
            )


def resolve_device(device: str) -> str:
    if device != "auto":
        return device
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def detect_model_format(model_path: str | Path, requested_format: str = "auto") -> str:
    requested_format = requested_format.lower()
    aliases = {"pt": "torch_export", "torch": "torch_export", "torch_export": "torch_export", "ckpt": "ckpt"}
    if requested_format != "auto":
        if requested_format not in aliases:
            raise ValueError("model.format must be one of: auto, ckpt, torch_export.")
        return aliases[requested_format]

    suffix = Path(model_path).suffix.lower()
    if suffix == ".ckpt":
        return "ckpt"
    if suffix == ".pt":
        return "torch_export"
    raise ValueError(f"Could not auto-detect model format from extension '{suffix}'.")


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
