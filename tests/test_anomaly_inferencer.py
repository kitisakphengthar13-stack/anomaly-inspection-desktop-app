import sys
from types import SimpleNamespace

import numpy as np
import pytest

from inspection.anomaly_inferencer import (
    AnomalyInferencer,
    EngineCheckpointBackend,
    ExportedTorchBackend,
    LightningCheckpointBackend,
    OpenVINOBackend,
    detect_model_format,
    extract_heatmap,
    extract_pred_label,
    extract_score,
)
from inspection.result_types import AnomalyResult


class Prediction:
    def __init__(self):
        self.pred_score = np.array([[0.74]], dtype=np.float32)
        self.pred_label = np.array([True])
        self.anomaly_map = np.ones((1, 1, 8, 8), dtype=np.float32)


class FakeOpenVINOInput:
    names = {"input"}

    def __init__(self, shape):
        self._shape = shape

    def get_partial_shape(self):
        return self._shape


def test_backend_auto_detection():
    assert detect_model_format("model.ckpt", "auto") == "ckpt"
    assert detect_model_format("model.pt", "auto") == "torch_export"
    assert detect_model_format("model.xml", "auto") == "openvino"
    assert detect_model_format("model.anything", "ckpt") == "ckpt"
    assert detect_model_format("model.anything", "torch_export") == "torch_export"
    assert detect_model_format("model.anything", "openvino") == "openvino"


def test_anomaly_inferencer_ckpt_defaults_to_engine_backend(monkeypatch):
    class FakeModel:
        pass

    class FakeResolved:
        name = "reverse_distillation"
        model_class = FakeModel
        source = "explicit"

    class FakeResolver:
        def resolve_for_checkpoint(self, model_path, anomalib_model):
            assert str(model_path) == "model.ckpt"
            assert anomalib_model == "reverse_distillation"
            return FakeResolved()

    monkeypatch.setattr("inspection.anomaly_inferencer.AnomalibModelResolver", lambda: FakeResolver())
    inferencer = AnomalyInferencer(
        model_path="model.ckpt",
        anomaly_threshold=0.5,
        device="cpu",
        model_format="ckpt",
        anomalib_model="reverse_distillation",
    )

    backend = inferencer._build_backend()

    assert isinstance(backend, EngineCheckpointBackend)
    assert backend.model_class is FakeModel
    assert backend.model_name == "reverse_distillation"
    assert backend.model_source == "explicit"


def test_anomaly_inferencer_ckpt_explicit_direct_uses_legacy_backend(monkeypatch):
    class FakeModel:
        pass

    class FakeResolved:
        name = "reverse_distillation"
        model_class = FakeModel
        source = "explicit"

    class FakeResolver:
        def resolve_for_checkpoint(self, model_path, anomalib_model):
            assert str(model_path) == "model.ckpt"
            assert anomalib_model == "reverse_distillation"
            return FakeResolved()

    monkeypatch.setattr("inspection.anomaly_inferencer.AnomalibModelResolver", lambda: FakeResolver())
    inferencer = AnomalyInferencer(
        model_path="model.ckpt",
        anomaly_threshold=0.5,
        device="cpu",
        model_format="ckpt",
        anomalib_model="reverse_distillation",
        checkpoint_inference_mode="direct",
    )

    with pytest.warns(RuntimeWarning, match="legacy/debug Lightning checkpoint path"):
        backend = inferencer._build_backend()

    assert isinstance(backend, LightningCheckpointBackend)
    assert backend.model_class is FakeModel
    assert backend.model_name == "reverse_distillation"
    assert backend.model_source == "explicit"


def test_anomaly_inferencer_engine_mode_uses_engine_checkpoint_backend(monkeypatch):
    class FakeModel:
        pass

    class FakeResolved:
        name = "reverse_distillation"
        model_class = FakeModel
        source = "explicit"

    class FakeResolver:
        def resolve_for_checkpoint(self, model_path, anomalib_model):
            assert str(model_path) == "model.ckpt"
            assert anomalib_model == "reverse_distillation"
            return FakeResolved()

    monkeypatch.setattr("inspection.anomaly_inferencer.AnomalibModelResolver", lambda: FakeResolver())
    inferencer = AnomalyInferencer(
        model_path="model.ckpt",
        device="cpu",
        model_format="ckpt",
        anomalib_model="reverse_distillation",
        checkpoint_inference_mode="engine",
    )

    backend = inferencer._build_backend()

    assert isinstance(backend, EngineCheckpointBackend)
    assert backend.model_class is FakeModel
    assert backend.model_name == "reverse_distillation"
    assert backend.model_source == "explicit"


def test_invalid_checkpoint_inference_mode_fails_clearly():
    try:
        AnomalyInferencer(model_path="model.ckpt", model_format="ckpt", checkpoint_inference_mode="magic")
    except ValueError as exc:
        assert "model.checkpoint_inference_mode must be one of: direct, engine" in str(exc)
    else:
        raise AssertionError("Expected invalid checkpoint inference mode to fail.")


def test_anomaly_inferencer_pt_and_openvino_ignore_checkpoint_inference_mode():
    torch_inferencer = AnomalyInferencer(
        model_path="model.pt",
        model_format="torch_export",
        anomalib_model="reverse_distillation",
        checkpoint_inference_mode="engine",
    )
    openvino_inferencer = AnomalyInferencer(
        model_path="model.xml",
        model_format="openvino",
        anomalib_model="reverse_distillation",
        checkpoint_inference_mode="engine",
    )

    assert isinstance(torch_inferencer._build_backend(), ExportedTorchBackend)
    assert isinstance(openvino_inferencer._build_backend(), OpenVINOBackend)


def test_engine_output_conversion_uses_existing_anomaly_result_contract(monkeypatch, tmp_path):
    class FakeBackend:
        backend_name = "engine_checkpoint"

        def load(self):
            pass

        def predict(self, image):
            return Prediction()

    monkeypatch.setattr(AnomalyInferencer, "_build_backend", lambda self: FakeBackend())
    model_path = tmp_path / "model.ckpt"
    model_path.write_bytes(b"fake")
    inferencer = AnomalyInferencer(
        model_path=model_path,
        model_format="ckpt",
        checkpoint_inference_mode="engine",
    )

    result = inferencer.predict(np.zeros((16, 16, 3), dtype=np.uint8))

    assert abs(result.anomaly_score - 0.74) < 1e-6
    assert result.pred_label is True
    assert result.heatmap.shape == (1, 1, 8, 8)
    assert result.backend_name == "engine_checkpoint"


def test_engine_checkpoint_error_message_is_specific(monkeypatch, tmp_path):
    class FakeBackend:
        backend_name = "engine_checkpoint"

        def load(self):
            pass

        def predict(self, image):
            raise RuntimeError("predict boom")

    monkeypatch.setattr(AnomalyInferencer, "_build_backend", lambda self: FakeBackend())
    model_path = tmp_path / "model.ckpt"
    model_path.write_bytes(b"fake")
    inferencer = AnomalyInferencer(model_path=model_path, model_format="ckpt")

    result = inferencer.predict(np.zeros((16, 16, 3), dtype=np.uint8))

    assert result.error_message == "Engine checkpoint inference failed: predict boom"


def test_engine_checkpoint_backend_uses_temporary_default_root_dir(monkeypatch, tmp_path):
    calls = []

    class FakeEngine:
        def __init__(self, **kwargs):
            calls.append(kwargs)

    class FakeLoadedModel:
        def eval(self):
            pass

        def to(self, device):
            pass

    class FakeModelClass:
        @staticmethod
        def load_from_checkpoint(path, weights_only=False):
            return FakeLoadedModel()

    monkeypatch.setitem(sys.modules, "anomalib.engine", SimpleNamespace(Engine=FakeEngine))
    backend = EngineCheckpointBackend(
        tmp_path / "model.ckpt",
        device="cpu",
        model_class=FakeModelClass,
        model_name="patchcore",
        model_source="explicit",
    )

    backend.load()

    assert calls[0]["default_root_dir"] != "results"
    assert "anomalib_engine_" in str(calls[0]["default_root_dir"])
    assert calls[0]["accelerator"] == "cpu"


def test_lightning_checkpoint_backend_load_uses_injected_model_class(tmp_path):
    calls = []

    class FakeLoadedModel:
        def eval(self):
            calls.append("eval")

        def to(self, device):
            calls.append(("to", device))

    class FakeModelClass:
        @staticmethod
        def load_from_checkpoint(path, weights_only=False):
            calls.append((path, weights_only))
            return FakeLoadedModel()

    model_path = tmp_path / "model.ckpt"
    backend = LightningCheckpointBackend(model_path, device="cpu", model_class=FakeModelClass, model_name="fake_model")

    backend.load()

    assert calls == [(str(model_path), False), "eval", ("to", "cpu")]


def test_lightning_checkpoint_backend_legacy_fallback_failure_has_context(tmp_path):
    class FailingModelClass:
        @staticmethod
        def load_from_checkpoint(path, weights_only=False):
            raise RuntimeError("state_dict mismatch")

    backend = LightningCheckpointBackend(
        tmp_path / "model.ckpt",
        device="cpu",
        model_class=FailingModelClass,
        model_name="patchcore",
        model_source="legacy_default",
    )

    try:
        backend.load()
    except RuntimeError as exc:
        message = str(exc)
        assert "No model.anomalib_model was set" in message
        assert "legacy PatchCore fallback" in message
        assert "set model.anomalib_model explicitly" in message
        assert "state_dict mismatch" in message
    else:
        raise AssertionError("Expected legacy fallback load failure to include corrective context.")


def test_lightning_checkpoint_backend_legacy_patchcore_success_still_loads(tmp_path):
    calls = []

    class FakeLoadedModel:
        def eval(self):
            calls.append("eval")

        def to(self, device):
            calls.append(("to", device))

    class FakePatchcoreClass:
        @staticmethod
        def load_from_checkpoint(path, weights_only=False):
            calls.append((path, weights_only))
            return FakeLoadedModel()

    backend = LightningCheckpointBackend(
        tmp_path / "model.ckpt",
        device="cpu",
        model_class=FakePatchcoreClass,
        model_name="patchcore",
        model_source="legacy_default",
    )

    backend.load()

    assert calls == [(str(tmp_path / "model.ckpt"), False), "eval", ("to", "cpu")]


def test_prediction_extractors_return_plain_values():
    prediction = Prediction()

    assert abs(extract_score(prediction) - 0.74) < 1e-6
    assert extract_pred_label(prediction) is True


def test_openvino_preprocessing_shape_dtype_and_scale():
    image = np.zeros((1024, 1024, 3), dtype=np.uint8)
    image[..., 2] = 255

    backend = OpenVINOBackend("model.xml", device="cpu")
    backend._configure_input(FakeOpenVINOInput([None, 3, 256, 256]))

    tensor = backend.preprocess(image)

    assert tensor.shape == (1, 3, 256, 256)
    assert tensor.dtype == np.float32
    assert np.allclose(tensor[:, 0, :, :], 1.0)
    assert np.allclose(tensor[:, 1:, :, :], 0.0)


def test_openvino_preprocessing_uses_non_256_fixed_input_size():
    image = np.zeros((1024, 1024, 3), dtype=np.uint8)
    backend = OpenVINOBackend("model.xml", device="cpu")
    backend._configure_input(FakeOpenVINOInput([1, 3, 384, 384]))

    tensor = backend.preprocess(image)

    assert backend.input_size == (384, 384)
    assert tensor.shape == (1, 3, 384, 384)


def test_openvino_backend_rejects_dynamic_height_width():
    backend = OpenVINOBackend("model.xml", device="cpu")

    with pytest.raises(ValueError) as exc_info:
        backend._configure_input(FakeOpenVINOInput([None, 3, None, 256]))

    message = str(exc_info.value)
    assert "model.xml" in message
    assert "height or width is dynamic or unavailable" in message
    assert "fixed-shape exported OpenVINO image model" in message


def test_openvino_backend_rejects_non_nchw_layout():
    backend = OpenVINOBackend("model.xml", device="cpu")

    with pytest.raises(ValueError) as exc_info:
        backend._configure_input(FakeOpenVINOInput([None, 256, 256, 3]))

    message = str(exc_info.value)
    assert "model.xml" in message
    assert "NHWC" in message
    assert "NCHW" in message


def test_openvino_backend_rejects_non_image_like_channels():
    backend = OpenVINOBackend("model.xml", device="cpu")

    with pytest.raises(ValueError) as exc_info:
        backend._configure_input(FakeOpenVINOInput([1, 1, 256, 256]))

    message = str(exc_info.value)
    assert "channel dimension is 1" in message
    assert "expected 3" in message


def test_openvino_backend_output_dict_matches_extractors(tmp_path):
    class FakeCompiledModel:
        inputs = ["input"]
        outputs = ["pred_score", "pred_label", "anomaly_map", "pred_mask"]

        def __call__(self, inputs):
            assert inputs["input"].shape == (1, 3, 256, 256)
            return {
                "pred_score": np.array([0.82], dtype=np.float32),
                "pred_label": np.array([True]),
                "anomaly_map": np.ones((1, 1, 256, 256), dtype=np.float32),
                "pred_mask": np.ones((1, 1, 256, 256), dtype=bool),
            }

    backend = OpenVINOBackend(tmp_path / "model.xml", device="cpu")
    backend.compiled_model = FakeCompiledModel()
    backend._configure_input(FakeOpenVINOInput([None, 3, 256, 256]))

    prediction = backend.predict(np.zeros((512, 512, 3), dtype=np.uint8))

    assert abs(extract_score(prediction) - 0.82) < 1e-6
    assert extract_pred_label(prediction) is True
    assert extract_heatmap(prediction).shape == (1, 1, 256, 256)


def test_openvino_backend_output_dict_uses_discovered_384_input_size(tmp_path):
    class FakeCompiledModel:
        inputs = ["input"]
        outputs = ["pred_score", "pred_label", "anomaly_map", "pred_mask"]

        def __call__(self, inputs):
            assert inputs["input"].shape == (1, 3, 384, 384)
            return {
                "pred_score": np.array([0.82], dtype=np.float32),
                "pred_label": np.array([True]),
                "anomaly_map": np.ones((1, 1, 384, 384), dtype=np.float32),
                "pred_mask": np.ones((1, 1, 384, 384), dtype=bool),
            }

    backend = OpenVINOBackend(tmp_path / "model.xml", device="cpu")
    backend.compiled_model = FakeCompiledModel()
    backend._configure_input(FakeOpenVINOInput([None, 3, 384, 384]))

    prediction = backend.predict(np.zeros((512, 512, 3), dtype=np.uint8))

    assert abs(extract_score(prediction) - 0.82) < 1e-6
    assert extract_heatmap(prediction).shape == (1, 1, 384, 384)


def test_openvino_backend_reports_missing_adjacent_bin(tmp_path):
    model_path = tmp_path / "model.xml"
    model_path.write_text("<xml />", encoding="utf-8")
    backend = OpenVINOBackend(model_path, device="cpu")

    try:
        backend.load()
    except FileNotFoundError as exc:
        assert "model.bin" in str(exc)
    else:
        raise AssertionError("Expected missing OpenVINO .bin file to fail before runtime loading.")


def test_unified_anomaly_result_contract():
    result = AnomalyResult(
        anomaly_score=0.25,
        pred_label=False,
        backend_name="exported_torch",
        model_path="model.pt",
        heatmap_path="heatmap.png",
    )

    assert result.anomaly_score == 0.25
    assert result.pred_label is False
    assert result.backend_name == "exported_torch"
    assert result.model_path == "model.pt"
