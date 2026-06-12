from inspection.anomaly_validation import ValidationRow, compute_metrics, iter_labeled_images, run_anomaly_validation
from inspection.config import InspectionConfig, ModelConfig, OutputConfig, PresenceConfig, ProjectConfig


def test_folder_mapping_good_bad(tmp_path):
    (tmp_path / "good").mkdir()
    (tmp_path / "bad").mkdir()
    (tmp_path / "good" / "a.png").write_bytes(b"x")
    (tmp_path / "bad" / "b.png").write_bytes(b"x")

    items = list(iter_labeled_images(tmp_path, ["good"], ["bad"]))

    labels = {(path.parent.name, label) for path, _, label in items}
    assert labels == {("good", "normal"), ("bad", "abnormal")}


def test_folder_mapping_splicing_style(tmp_path):
    for folder in ("good", "logical_anomalies", "structural_anomalies"):
        (tmp_path / folder).mkdir()
        (tmp_path / folder / f"{folder}.png").write_bytes(b"x")

    items = list(iter_labeled_images(tmp_path, ["good"], ["logical_anomalies", "structural_anomalies"]))

    labels = {(category, label) for _, category, label in items}
    assert labels == {
        ("good", "normal"),
        ("logical_anomalies", "abnormal"),
        ("structural_anomalies", "abnormal"),
    }


def test_compute_metrics_with_pred_label():
    rows = [
        ValidationRow("good.png", "good", "normal", 0.1, False, True),
        ValidationRow("fp.png", "good", "normal", 0.9, True, False),
        ValidationRow("bad.png", "bad", "abnormal", 0.8, True, True),
        ValidationRow("fn.png", "bad", "abnormal", 0.2, False, False),
    ]

    metrics = compute_metrics(rows)

    assert metrics["TP"] == 1
    assert metrics["TN"] == 1
    assert metrics["FP"] == 1
    assert metrics["FN"] == 1


def test_run_anomaly_validation_passes_explicit_anomalib_model(tmp_path, monkeypatch):
    captured = {}

    class FakeInferencer:
        backend_name = "lightning_checkpoint"

        def __init__(self, **kwargs):
            captured.update(kwargs)

        def load(self):
            pass

    monkeypatch.setattr("inspection.anomaly_validation.AnomalyInferencer", FakeInferencer)
    (tmp_path / "good").mkdir()
    config = InspectionConfig(
        project=ProjectConfig(name="RD"),
        model=ModelConfig(
            path=tmp_path / "model.ckpt",
            anomaly_threshold=0.5,
            device="cpu",
            format="ckpt",
            anomalib_model="reverse_distillation",
        ),
        presence=PresenceConfig(reference_image_path=tmp_path / "ref.png", zones_path=tmp_path / "zones.json"),
        output=OutputConfig(),
        config_path=tmp_path / "inspection.yaml",
    )

    run_anomaly_validation(
        config=config,
        test_root=tmp_path,
        normal_folders=["good"],
        abnormal_folders=[],
        output_dir=tmp_path / "out",
    )

    assert captured["model_path"] == tmp_path / "model.ckpt"
    assert captured["model_format"] == "ckpt"
    assert captured["anomalib_model"] == "reverse_distillation"
