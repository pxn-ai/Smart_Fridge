# Date Region YOLO Guide

This project now supports an optional YOLO detector to crop the MFD/EXP print area before OCR.

## Why this helps

Tesseract struggles when the full label contains logos, graphics, and curved text.
Detecting only the date-print patch first improves OCR precision.

## Integration in this repo

- Script: `camera_project/bottle_ocr_test.py`
- Model path expected: `camera_project/models/date_region_yolo.pt`
- If model is missing, script falls back to center ROI automatically.

## 1) Collect training images

Capture many bottle photos in real conditions:

- different brands and bottle sizes
- different lighting / glare / blur levels
- different distances and orientations
- include difficult samples

Start with at least 150-300 labeled images.

## 2) Label only one class

Class name: `date_region`

Draw one box tightly around the printed block containing:

- `MFD`, `MFG`, `EXP`, `Best Before`, etc.
- nearby date strings

Do not label logos or full label area.

## 3) YOLO dataset layout

Use standard YOLO format:

```text
dataset/
  images/
    train/
    val/
  labels/
    train/
    val/
  data.yaml
```

`data.yaml` example:

```yaml
path: /absolute/path/to/dataset
train: images/train
val: images/val
names:
  0: date_region
```

## 4) Train (on laptop/desktop, not Pi)

Install Ultralytics and train a small model:

```bash
pip install ultralytics
yolo detect train data=/path/to/data.yaml model=yolov8n.pt imgsz=640 epochs=100 batch=16
```

Recommended start:

- model: `yolov8n.pt` or `yolov8s.pt`
- imgsz: `640`
- augmentations enabled (default)

## 5) Export model to Pi

Copy best checkpoint to:

`camera_project/models/date_region_yolo.pt`

Then install runtime on Pi:

```bash
pip install ultralytics
```

## 6) Run OCR pipeline

```bash
cd /home/pasan/smartf
.venv/bin/python camera_project/bottle_ocr_test.py
```

You should see:

- `ROI source: YOLO date ROI (...)` when detection works
- fallback message when model is missing or no box is detected

## Notes

- Detector quality dominates OCR quality in this pipeline.
- Keep date box labels tight and consistent.
- Add hard negatives (boxes that look like date regions but are not) to reduce false positives.
