import warnings

from anomaly_inspection.core.anomalib_model_registry import AnomalibModelRegistry, canonicalize_anomalib_model_name
from anomaly_inspection.core.anomalib_model_resolver import AnomalibModelResolver


class FakeRegistry:
    def __init__(self):
        self.classes = {
            "patchcore": PatchcoreModel,
            "reverse_distillation": ReverseDistillationModel,
        }

    def resolve_class(self, model_name):
        if model_name not in self.classes:
            raise ValueError(f"Unsupported Anomalib model '{model_name}'.")
        return self.classes[model_name]


class PatchcoreModel:
    pass


class ReverseDistillationModel:
    pass


def test_canonicalize_anomalib_model_name_accepts_common_forms():
    assert canonicalize_anomalib_model_name("reverse_distillation") == "reverse_distillation"
    assert canonicalize_anomalib_model_name("ReverseDistillation") == "reverse_distillation"
    assert canonicalize_anomalib_model_name("reverse-distillation") == "reverse_distillation"
    assert canonicalize_anomalib_model_name("PatchCore") == "patchcore"


def test_resolver_explicit_reverse_distillation():
    resolver = AnomalibModelResolver(registry=FakeRegistry())

    resolved = resolver.resolve_for_checkpoint("model.ckpt", "reverse_distillation")

    assert resolved.name == "reverse_distillation"
    assert resolved.model_class is ReverseDistillationModel
    assert resolved.source == "explicit"


def test_resolver_missing_model_uses_legacy_patchcore():
    resolver = AnomalibModelResolver(registry=FakeRegistry())

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        resolved = resolver.resolve_for_checkpoint("model.ckpt", None)

    assert resolved.name == "patchcore"
    assert resolved.model_class is PatchcoreModel
    assert resolved.source == "legacy_default"
    assert any("legacy PatchCore fallback" in str(item.message) for item in caught)


def test_resolver_auto_is_clear_not_implemented():
    resolver = AnomalibModelResolver(registry=FakeRegistry())

    try:
        resolver.resolve_for_checkpoint("model.ckpt", "auto")
    except NotImplementedError as exc:
        assert "model.anomalib_model: auto is not implemented" in str(exc)
    else:
        raise AssertionError("Expected auto model detection to be explicitly unsupported in Phase 1.")


def test_resolver_unknown_model_raises_clear_error():
    resolver = AnomalibModelResolver(registry=FakeRegistry())

    try:
        resolver.resolve_for_checkpoint("model.ckpt", "unknown_model")
    except ValueError as exc:
        assert "Unsupported Anomalib model" in str(exc)
    else:
        raise AssertionError("Expected unknown model to fail clearly.")


def test_registry_resolves_installed_patchcore_and_reverse_distillation_classes():
    registry = AnomalibModelRegistry()

    assert registry.resolve_class("patchcore").__name__ == "Patchcore"
    assert registry.resolve_class("reverse_distillation").__name__ == "ReverseDistillation"
