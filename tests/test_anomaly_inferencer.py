import numpy as np

from inspection.anomaly_inferencer import detect_model_format, extract_pred_label, extract_score
from inspection.result_types import AnomalyResult


class Prediction:
    def __init__(self):
        self.pred_score = np.array([[0.74]], dtype=np.float32)
        self.pred_label = np.array([True])


def test_backend_auto_detection():
    assert detect_model_format("model.ckpt", "auto") == "ckpt"
    assert detect_model_format("model.pt", "auto") == "torch_export"
    assert detect_model_format("model.anything", "ckpt") == "ckpt"
    assert detect_model_format("model.anything", "torch_export") == "torch_export"


def test_prediction_extractors_return_plain_values():
    prediction = Prediction()

    assert abs(extract_score(prediction) - 0.74) < 1e-6
    assert extract_pred_label(prediction) is True


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
