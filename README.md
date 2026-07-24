# Anomaly Inspection Desktop App

A local PySide6 application for fixed-camera anomaly inspection. It checks that a part is present before running an Anomalib-compatible model, so an empty station is reported as `NO_PART` rather than a false anomaly result.

![Inspection result](src/anomaly_inspection/resources/images/sample_inspection.png)

## How it works

```text
image or camera frame
  -> polygon-zone presence check
  -> NO_PART, or anomaly inference
  -> OK / NG / ERROR
```

- The presence check compares an image with an empty-station reference inside configured polygon zones.
- When a model provides `pred_label`, it determines `OK` or `NG`. The configured threshold is used only when no label is returned.
- The app saves inspection artifacts and CSV logs locally. It does not require a server or database.

## Install

Python 3.9 or later is required.

```bash
python -m pip install -e ".[gui,anomaly,test]"
```

For the desktop app without model runtime dependencies:

```bash
python -m pip install -e ".[gui]"
```

## Run the desktop app

```bash
anomaly-inspection
```

Use the desktop workflow in this order:

1. **Project Setup** — save a local YAML config and validate its paths.
2. **Capture Reference** — save an empty-station reference image.
3. **Draw Zones** — draw the presence-check polygons on that reference image.
4. **Inspect Image** or **Inspect Camera** — review the result, annotated image, heatmap, and presence mask.
5. **Logs** — review existing CSV history and saved artifacts.

The reference image, zones JSON, and inspection image must have the same resolution.

## Configuration

Start from the public samples:

```powershell
Copy-Item configs\inspection.sample.yaml configs\local_inspection.yaml
Copy-Item configs\zones.sample.json configs\zones.json
```

Edit `configs/local_inspection.yaml` with local model, reference-image, and zones paths. Local configs, models, images, and outputs are ignored by Git.

Important fields:

```yaml
model:
  path: "path/to/anomaly_model.xml"
  format: "auto" # ckpt, torch_export, openvino, or auto
  anomaly_threshold: 0.5

presence:
  reference_image_path: "path/to/reference_image.png"
  zones_path: "path/to/zones.json"
```

Supported model artifacts:

- Lightning checkpoint: `.ckpt`
- Exported Torch artifact: `.pt`
- OpenVINO model: `.xml` with its adjacent `.bin` file

Only load model artifacts from trusted sources.

## CLI

The CLI is optional and uses the same core pipeline as the desktop app.

```bash
# Validate local setup
anomaly-inspection-cli validate-config --config configs/local_inspection.yaml

# Capture an empty reference image
anomaly-inspection-cli capture-reference --output data/reference/empty_reference.png --camera-index 0

# Create polygon zones
anomaly-inspection-cli setup-zones --image data/reference/empty_reference.png --zones configs/zones.json

# Inspect one image
anomaly-inspection-cli inspect-image --image data/samples/test_001.png --config configs/local_inspection.yaml --output outputs/single_test

# Inspect a folder
anomaly-inspection-cli inspect-folder --input data/samples/batch --config configs/local_inspection.yaml --output outputs/batch
```

`inspection-app` and `inspection-cli` remain available as compatibility aliases.

## Outputs

Single-image and camera inspections append to `inspection_log.csv`. Folder inspections create a separate run directory.

```text
outputs/<job>/folder/run_<timestamp>/
  summary.csv
  annotated/
  heatmaps/
  presence_masks/
  OK/
  NG/
  NO_PART/
  ERROR/
```

CSV records include the final result, presence metrics, model score/label, artifact paths, timings, and any error message.

## Project layout

```text
src/anomaly_inspection/
  core/          inspection rules, pipeline, and model adapters
  desktop/       PySide6 application and workflow pages
  cli.py         command-line entry point
  cli_support/   OpenCV helpers for CLI setup commands
  resources/     packaged UI assets
```

## Tests

```bash
python -m pytest
```

## Notes

- Presence detection assumes a stable camera, background, and lighting. Recalibrate its thresholds when those conditions change.
- Score scales can differ between model artifacts; validate a model with representative normal and abnormal images before setting a fallback threshold.
- The sample preview uses the MVTec Anomaly Detection Dataset, Transistor category, for demonstration only.
