from __future__ import annotations

import argparse
from pathlib import Path

from inspection.config import load_config
from inspection.anomaly_validation import run_anomaly_validation
from inspection.pipeline import InspectionPipeline
from inspection.presence_checker import PresenceChecker
from inspection.reference_capture import run_reference_capture
from inspection.result_types import FinalResult
from inspection.zone_editor import run_zone_editor
from inspection.zone_io import load_zones


def _print_result(result) -> None:
    print(f"Image: {result.image_path}")
    print(f"Final result: {result.final_result.value}")
    print(f"Presence: {result.presence_status.value if result.presence_status else ''}")
    if result.foreground_ratio is not None:
        print(f"Foreground ratio: {result.foreground_ratio:.6f}")
    if result.mean_diff is not None:
        print(f"Mean diff: {result.mean_diff:.6f}")
    if result.largest_blob_area is not None:
        print(f"Largest blob area: {result.largest_blob_area:.2f}")
    if result.anomaly_score is not None:
        print(f"Anomaly score: {result.anomaly_score:.6f}")
        print(f"Anomaly threshold: {result.anomaly_threshold:.6f}")
    if result.anomaly_pred_label is not None:
        print(f"Anomaly pred_label: {result.anomaly_pred_label}")
    if result.anomaly_backend:
        print(f"Anomaly backend: {result.anomaly_backend}")
    if result.annotated_image_path:
        print(f"Annotated image: {result.annotated_image_path}")
    if result.heatmap_path:
        print(f"Heatmap: {result.heatmap_path}")
    if result.presence_mask_path:
        print(f"Presence mask: {result.presence_mask_path}")
    if result.error_message:
        print(f"Error: {result.error_message}")


def setup_zones(args: argparse.Namespace) -> int:
    run_zone_editor(args.image, args.zones)
    return 0


def capture_reference(args: argparse.Namespace) -> int:
    run_reference_capture(args.output, camera_index=args.camera_index, width=args.width, height=args.height)
    return 0


def inspect_image(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    pipeline = InspectionPipeline(config)
    result = pipeline.inspect_image(args.image, args.output)
    _print_result(result)
    if config.output.save_csv_log:
        print(f"Inspection CSV log: {Path(args.output) / 'inspection_log.csv'}")
    return 1 if result.final_result == FinalResult.ERROR else 0


def inspect_folder(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    pipeline = InspectionPipeline(config)
    results = pipeline.inspect_folder(args.input, args.output)
    error_count = sum(1 for result in results if result.final_result == FinalResult.ERROR)
    print(f"Processed images: {len(results)}")
    print(f"Errors: {error_count}")
    if pipeline.last_folder_output_dir is not None:
        print(f"Folder inspection run directory: {pipeline.last_folder_output_dir}")
    if config.output.save_csv_log:
        summary_path = (
            pipeline.last_folder_output_dir / "summary.csv"
            if pipeline.last_folder_output_dir is not None
            else Path(args.output) / "summary.csv"
        )
        print(f"Folder summary CSV: {summary_path}")
    return 1 if error_count else 0


def test_presence(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    zones = load_zones(config.presence.zones_path)
    checker = PresenceChecker(config.presence, zones)
    output_dir = Path(args.output)
    result = checker.check_image_path(args.image, debug_mask_path=output_dir / "presence_mask.png")
    print(f"Presence: {result.status.value}")
    print(f"Foreground ratio: {result.foreground_ratio:.6f}")
    print(f"Mean diff: {result.mean_diff:.6f}")
    print(f"Largest blob area: {result.largest_blob_area:.2f}")
    print(f"Changed pixels: {result.changed_pixel_count}")
    print(f"Zone pixels: {result.zone_pixel_count}")
    if result.presence_mask_path:
        print(f"Presence mask: {result.presence_mask_path}")
    return 0


def validate_config(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    zones = load_zones(config.presence.zones_path)
    PresenceChecker(config.presence, zones)
    print("Config is valid.")
    print(f"Reference image: {config.presence.reference_image_path}")
    print(f"Zones: {config.presence.zones_path}")
    print(f"Model path: {config.model.path}")
    print(f"Model format: {config.model.format}")
    return 0


def validate_anomaly_model(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    report = run_anomaly_validation(
        config=config,
        test_root=args.test_root,
        normal_folders=args.normal_folders,
        abnormal_folders=args.abnormal_folders,
        output_dir=args.output,
    )
    print(f"Backend: {report['backend_name']}")
    print(f"Total images: {report['total_images']}")
    print(f"Predictions CSV: {report['prediction_csv']}")
    print(f"Threshold sweep CSV: {report['threshold_sweep_csv']}")
    print(f"Overall: {report['overall']}")
    print(f"Best F1: {report['best_f1']}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Presence-gated anomaly inspection CLI.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    setup = subparsers.add_parser("setup-zones", help="Draw polygon zones on a reference image.")
    setup.add_argument("--image", required=True, help="Reference image path.")
    setup.add_argument("--zones", required=True, help="Output zone JSON path.")
    setup.set_defaults(func=setup_zones)

    capture = subparsers.add_parser("capture-reference", help="Capture an empty reference image from a webcam.")
    capture.add_argument("--output", required=True, help="Output reference image path.")
    capture.add_argument("--camera-index", type=int, default=0, help="OpenCV camera index. Default: 0.")
    capture.add_argument("--width", type=int, default=None, help="Optional requested capture width.")
    capture.add_argument("--height", type=int, default=None, help="Optional requested capture height.")
    capture.set_defaults(func=capture_reference)

    single = subparsers.add_parser("inspect-image", help="Inspect one image.")
    single.add_argument("--image", required=True, help="Input image path.")
    single.add_argument("--config", required=True, help="Inspection YAML config path.")
    single.add_argument("--output", required=True, help="Output directory.")
    single.set_defaults(func=inspect_image)

    folder = subparsers.add_parser("inspect-folder", help="Inspect a folder of images.")
    folder.add_argument("--input", required=True, help="Input folder.")
    folder.add_argument("--config", required=True, help="Inspection YAML config path.")
    folder.add_argument("--output", required=True, help="Output directory.")
    folder.set_defaults(func=inspect_folder)

    presence = subparsers.add_parser("test-presence", help="Run only the presence checker.")
    presence.add_argument("--image", required=True, help="Input image path.")
    presence.add_argument("--config", required=True, help="Inspection YAML config path.")
    presence.add_argument("--output", required=True, help="Output directory.")
    presence.set_defaults(func=test_presence)

    validate = subparsers.add_parser("validate-config", help="Validate config, zones, and reference image.")
    validate.add_argument("--config", required=True, help="Inspection YAML config path.")
    validate.set_defaults(func=validate_config)

    validate_model = subparsers.add_parser("validate-anomaly-model", help="Validate anomaly model on labeled folders.")
    validate_model.add_argument("--test-root", required=True, help="Root containing normal/abnormal category folders.")
    validate_model.add_argument("--normal-folders", nargs="+", required=True, help="Folder names treated as normal.")
    validate_model.add_argument("--abnormal-folders", nargs="+", required=True, help="Folder names treated as abnormal.")
    validate_model.add_argument("--config", required=True, help="Inspection YAML config path.")
    validate_model.add_argument("--output", required=True, help="Output directory.")
    validate_model.set_defaults(func=validate_anomaly_model)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
