from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from anomaly_inspection.core.anomalib_model_registry import AnomalibModelRegistry, canonicalize_anomalib_model_name


LEGACY_DEFAULT_CKPT_MODEL = "patchcore"


@dataclass(frozen=True)
class ResolvedAnomalibModel:
    name: str
    model_class: Any
    source: str


class AnomalibModelResolver:
    """Resolve the Anomalib model class used to load Lightning checkpoints."""

    def __init__(self, registry: AnomalibModelRegistry | None = None) -> None:
        self.registry = registry or AnomalibModelRegistry()

    def resolve_for_checkpoint(self, model_path: str | Path, anomalib_model: str | None) -> ResolvedAnomalibModel:
        requested = (anomalib_model or "").strip()
        if not requested:
            model_class = self.registry.resolve_class(LEGACY_DEFAULT_CKPT_MODEL)
            warnings.warn(
                "model.anomalib_model is not set for a Lightning checkpoint; "
                "using legacy PatchCore fallback. If this checkpoint is not PatchCore, "
                "set model.anomalib_model explicitly.",
                UserWarning,
                stacklevel=2,
            )
            return ResolvedAnomalibModel(
                name=LEGACY_DEFAULT_CKPT_MODEL,
                model_class=model_class,
                source="legacy_default",
            )

        canonical_name = canonicalize_anomalib_model_name(requested)
        if canonical_name == "auto":
            raise NotImplementedError(
                "model.anomalib_model: auto is not implemented for Lightning checkpoints yet. "
                "Set model.anomalib_model explicitly, for example 'patchcore' or 'reverse_distillation'."
            )

        model_class = self.registry.resolve_class(canonical_name)
        return ResolvedAnomalibModel(name=canonical_name, model_class=model_class, source="explicit")
