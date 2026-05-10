#!/usr/bin/env python3
"""
Bottle Label OCR — Headless Test
Captures a single frame from Camera V2, runs OCR, prints results,
and saves the annotated image.  No display/GUI needed.

Uses Tesseract OCR (PaddlePaddle 3.2.2 segfaults on aarch64/RPi5).
Includes blur detection + adaptive sharpening from bottle_ocr.py.
"""

import cv2
import numpy as np
import time
import os
import pytesseract

# ── Configuration ─────────────────────────────────────────────
CAMERA_WIDTH  = 1920
CAMERA_HEIGHT = 1080
CONFIDENCE_MIN = 60        # Tesseract confidence is 0-100
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "test_output")
# ──────────────────────────────────────────────────────────────


def blur_score(gray: np.ndarray) -> float:
    """
    Laplacian variance — higher = sharper.
    Rough guide: <30 very blurry, 30-80 soft, >80 acceptable.
    """
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def sharpen(gray: np.ndarray, strength: float = 1.5) -> np.ndarray:
    """
    Unsharp mask — the standard fix for close-up/macro softness.
    Subtracts a blurred version of itself to amplify edges.
    strength: 0.8 = subtle, 1.5 = moderate, 2.5 = aggressive.
    """
    blurred = cv2.GaussianBlur(gray, (0, 0), sigmaX=2)
    return cv2.addWeighted(gray, 1 + strength, blurred, -strength, 0)


def preprocess(frame: np.ndarray) -> np.ndarray:
    """
    Improve OCR accuracy for close-up, glossy bottle labels:
      1. Grayscale
      2. Blur detection — reports score so you can judge distance/focus
      3. Unsharp mask sharpening — strength scales with how blurry it is
      4. CLAHE — local contrast boost for curved/shiny surfaces
      5. Denoise
    Returns grayscale image (Tesseract works best with grayscale, not thresholded).
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # ── Blur detection & report ───────────────────────────────
    score = blur_score(gray)
    if score < 30:
        print(f"  WARNING: Very blurry (score: {score:.1f}) — try moving slightly further back.")
    elif score < 80:
        print(f"  NOTE: Soft image (score: {score:.1f}) — sharpening applied.")
    else:
        print(f"  OK: Sharpness good (score: {score:.1f})")

    # ── Unsharp mask sharpening ───────────────────────────────
    strength = max(0.8, min(2.5, 120.0 / (score + 1)))
    gray = sharpen(gray, strength=strength)

    # ── CLAHE ─────────────────────────────────────────────────
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    # ── Denoise ───────────────────────────────────────────────
    gray = cv2.fastNlMeansDenoising(gray, h=10)

    return gray


def run_ocr(image: np.ndarray) -> list:
    """
    Run Tesseract OCR and return list of (box, text, confidence).
    box is [[x1,y1],[x2,y2],[x3,y3],[x4,y4]] (clockwise from top-left).
    Uses PSM 11 (sparse text) which works best on product labels.
    """
    data = pytesseract.image_to_data(
        image,
        config='--psm 11',
        output_type=pytesseract.Output.DICT,
    )

    results = []
    n = len(data['text'])
    for i in range(n):
        text = data['text'][i].strip()
        conf = int(data['conf'][i])

        if not text or conf < 0:
            continue

        x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
        box = [
            [x, y],
            [x + w, y],
            [x + w, y + h],
            [x, y + h],
        ]
        results.append((box, text, conf))

    return results


def draw_results(frame: np.ndarray, results: list) -> np.ndarray:
    """Draw bounding boxes and text on the frame."""
    overlay = frame.copy()

    for box, text, confidence in results:
        if confidence < CONFIDENCE_MIN:
            continue

        pts = np.array(box, dtype=np.int32)
        cv2.polylines(overlay, [pts], isClosed=True,
                      color=(0, 255, 0), thickness=2)

        x, y = box[0][0], box[0][1] - 8
        label = f"{text}  ({confidence}%)"
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(overlay, (x, y - h - 4), (x + w + 4, y + 4),
                      (0, 255, 0), -1)
        cv2.putText(overlay, label, (x + 2, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)

    return overlay


def print_results(results: list) -> None:
    """Print detected text to terminal."""
    print("\n" + "=" * 50)
    print("  DETECTED TEXT")
    print("=" * 50)

    filtered = [(t, c) for _, t, c in results if c >= CONFIDENCE_MIN]

    if not filtered:
        print("  No text detected.")
    else:
        for text, confidence in filtered:
            print(f"  {text:<35} ({confidence}%)")

    print("=" * 50 + "\n")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── 1. Verify Tesseract ──────────────────────────────────
    version = pytesseract.get_tesseract_version()
    print(f"Tesseract OCR v{version} ready.")
    print("(Fallback for PaddleOCR — PaddlePaddle 3.2.2 segfaults on aarch64)\n")

    # ── 2. Start camera ─────────────────────────────────────
    print("Starting Camera V2 (IMX219)...")
    from picamera2 import Picamera2
    cam = Picamera2()
    config = cam.create_preview_configuration(
        main={"size": (CAMERA_WIDTH, CAMERA_HEIGHT), "format": "BGR888"}
    )
    cam.configure(config)
    cam.start()
    time.sleep(2)  # let auto-exposure / AWB settle
    print("Camera ready.\n")

    # ── 3. Capture frame ─────────────────────────────────────
    print("Capturing frame...")
    frame = cam.capture_array()
    cam.stop()
    cam.close()

    # IMX219 reports Rotation=180 — correct it
    frame = cv2.rotate(frame, cv2.ROTATE_180)

    ts = time.strftime("%Y%m%d_%H%M%S")
    raw_path = os.path.join(OUTPUT_DIR, f"raw_{ts}.jpg")
    cv2.imwrite(raw_path, frame)
    print(f"  Raw frame saved  → {raw_path}")

    # ── 4. Preprocess ────────────────────────────────────────
    print("Preprocessing (blur check → sharpen → CLAHE → denoise)...")
    processed = preprocess(frame)

    proc_path = os.path.join(OUTPUT_DIR, f"processed_{ts}.jpg")
    cv2.imwrite(proc_path, processed)
    print(f"  Processed frame  → {proc_path}")

    # ── 5. Run OCR ───────────────────────────────────────────
    print("Running Tesseract OCR (PSM 11 — sparse text)...")
    t0 = time.time()
    results = run_ocr(processed)
    elapsed = time.time() - t0
    print(f"OCR completed in {elapsed:.2f}s")

    # ── 6. Print & save annotated output ─────────────────────
    print_results(results)

    overlay = draw_results(frame, results)
    overlay_path = os.path.join(OUTPUT_DIR, f"annotated_{ts}.jpg")
    cv2.imwrite(overlay_path, overlay)
    print(f"  Annotated frame  → {overlay_path}")

    # Also dump full OCR text for debugging
    full_text = pytesseract.image_to_string(processed, config='--psm 11').strip()
    print("\n── Full OCR text (unfiltered) ──")
    print(full_text if full_text else "(empty)")
    print("── End ──\n")

    print("Done! Check the output images in:")
    print(f"  {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
