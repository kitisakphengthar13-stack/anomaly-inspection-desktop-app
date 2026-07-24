from __future__ import annotations

import re
from typing import Any


def canonicalize_anomalib_model_name(model_name: str) -> str:
    """Normalize user-facing Anomalib model names to registry keys."""
    value = model_name.strip()
    if not value:
        return ""
    if value.lower() in {"patchcore", "patch_core"}:
        return "patchcore"
    value = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", value)
    value = re.sub(r"[^A-Za-z0-9]+", "_", value)
    return re.sub(r"_+", "_", value).strip("_").lower()


class AnomalibModelRegistry:
    """Class-level registry for installed Anomalib models.

    This intentionally resolves classes from ``anomalib.models`` without
    calling ``get_model()``, which can instantiate models and trigger cache or
    network access for some model families.
    """

    def available_model_names(self) -> set[str]:
        import anomalib.models as models

        return {canonicalize_anomalib_model_name(name) for name in models.list_models()}

    def is_supported(self, model_name: str) -> bool:
        try:
            self._resolve_canonical_name(model_name)
            return True
        except ValueError:
            return False

    def resolve_class(self, model_name: str) -> Any:
        import anomalib.models as models

        canonical_name = self._resolve_canonical_name(model_name)
        class_name = models.convert_snake_to_pascal_case(canonical_name)
        if hasattr(models, class_name):
            return getattr(models, class_name)
        raise ValueError(f"Anomalib model '{model_name}' is listed but class '{class_name}' is not available.")

    def _resolve_canonical_name(self, model_name: str) -> str:
        canonical_name = canonicalize_anomalib_model_name(model_name)
        if not canonical_name:
            raise ValueError("model.anomalib_model must not be blank.")
        available_names = self.available_model_names()
        if canonical_name in available_names:
            return canonical_name

        compact_name = canonical_name.replace("_", "")
        for available_name in available_names:
            if available_name.replace("_", "") == compact_name:
                return available_name

        available = ", ".join(sorted(available_names))
        raise ValueError(f"Unsupported Anomalib model '{model_name}'. Available models: {available}")
