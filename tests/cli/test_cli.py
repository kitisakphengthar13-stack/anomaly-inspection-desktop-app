from anomaly_inspection.cli import build_parser


def test_cli_exposes_the_inspection_commands():
    parser = build_parser()

    validate = parser.parse_args(["validate-config", "--config", "configs/local_inspection.yaml"])
    inspect = parser.parse_args(
        [
            "inspect-image",
            "--image",
            "data/samples/test_001.png",
            "--config",
            "configs/local_inspection.yaml",
            "--output",
            "outputs/single_test",
        ]
    )

    assert callable(validate.func)
    assert callable(inspect.func)
