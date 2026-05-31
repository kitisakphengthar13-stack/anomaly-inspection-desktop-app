import numpy as np

from inspection.anomaly_inferencer import (
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


def test_backend_auto_detection():
    assert detect_model_format("model.ckpt", "auto") == "ckpt"
    assert detect_model_format("model.pt", "auto") == "torch_export"
    assert detect_model_format("model.xml", "auto") == "openvino"
    assert detect_model_format("model.anything", "ckpt") == "ckpt"
    assert detect_model_format("model.anything", "torch_export") == "torch_export"
    assert detect_model_format("model.anything", "openvino") == "openvino"


def test_prediction_extractors_return_plain_values():
    prediction = Prediction()

    assert abs(extract_score(prediction) - 0.74) < 1e-6
    assert extract_pred_label(prediction) is True


def test_openvino_preprocessing_shape_dtype_and_scale():
    image = np.zeros((1024, 1024, 3), dtype=np.uint8)
    image[..., 2] = 255

    tensor = OpenVINOBackend.preprocess(image)

    assert tensor.shape == (1, 3, 256, 256)
    assert tensor.dtype == np.float32
    assert np.allclose(tensor[:, 0, :, :], 1.0)
    assert np.allclose(tensor[:, 1:, :, :], 0.0)


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
    backend.input_name = "input"

    prediction = backend.predict(np.zeros((512, 512, 3), dtype=np.uint8))

    assert abs(extract_score(prediction) - 0.82) < 1e-6
    assert extract_pred_label(prediction) is True
    assert extract_heatmap(prediction).shape == (1, 1, 256, 256)


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
