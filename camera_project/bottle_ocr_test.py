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
import re
import pytesseract
from typing import List, Tuple

# ── Configuration ─────────────────────────────────────────────
CAMERA_WIDTH  = 1920
CAMERA_HEIGHT = 1080
CONFIDENCE_MIN = 60        # Tesseract confidence is 0-100
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "test_output")
ROTATE_180 = True
UPSCALE_FACTOR = 2.0
SMALL_TEXT_UPSCALE_FACTOR = 3.2
MIN_TEXT_LEN = 3
BURST_COUNT = 6
BURST_INTERVAL_SEC = 0.20
ROTATION_CANDIDATES = (0, 180)  # Keep lightweight for Pi
# Central ROI where bottle labels usually appear.
ROI_X0 = 0.18
ROI_X1 = 0.82
ROI_Y0 = 0.18
ROI_Y1 = 0.88
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


def extract_label_roi(frame: np.ndarray) -> Tuple[np.ndarray, tuple]:
    """Crop central region to focus OCR on bottle label area."""
    h, w = frame.shape[:2]
    x0, x1 = int(w * ROI_X0), int(w * ROI_X1)
    y0, y1 = int(h * ROI_Y0), int(h * ROI_Y1)
    roi = frame[y0:y1, x0:x1]
    return roi, (x0, y0)


def capture_best_burst_frame(cam) -> np.ndarray:
    """
    Capture a short burst and pick the sharpest frame by ROI blur score.
    This improves OCR stability when focus fluctuates frame-to-frame.
    """
    best_frame = None
    best_score = -1.0
    scores = []

    for i in range(BURST_COUNT):
        frame = cam.capture_array()
        if ROTATE_180:
            frame = cv2.rotate(frame, cv2.ROTATE_180)

        roi, _ = extract_label_roi(frame)
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        score = blur_score(gray_roi)
        scores.append(score)

        if score > best_score:
            best_score = score
            best_frame = frame

        if i < BURST_COUNT - 1:
            time.sleep(BURST_INTERVAL_SEC)

    scores_str = ", ".join(f"{s:.1f}" for s in scores)
    print(f"  Burst sharpness scores: [{scores_str}]")
    print(f"  Selected sharpest frame: {best_score:.1f}")
    return best_frame


def build_preprocess_variants(frame: np.ndarray) -> List[Tuple[str, np.ndarray]]:
    """
    Build multiple variants for label OCR.
    Soft drink labels vary (glossy, curved, embossed), so multi-pass helps.
    """
    base = preprocess(frame)
    up = cv2.resize(
        base,
        None,
        fx=UPSCALE_FACTOR,
        fy=UPSCALE_FACTOR,
        interpolation=cv2.INTER_CUBIC,
    )

    variants = [("gray_upscaled", up)]
    up_small = cv2.resize(
        base,
        None,
        fx=SMALL_TEXT_UPSCALE_FACTOR,
        fy=SMALL_TEXT_UPSCALE_FACTOR,
        interpolation=cv2.INTER_CUBIC,
    )

    variants.append(("gray_smalltext_upscaled", up_small))

    # Variant 2: adaptive threshold for high-contrast printed text.
    th_adapt = cv2.adaptiveThreshold(
        up, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 8
    )
    variants.append(("adaptive_thresh", th_adapt))

    # Variant 3: Otsu threshold after slight blur to reduce shiny noise.
    blur = cv2.GaussianBlur(up, (3, 3), 0)
    _, th_otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    variants.append(("otsu_thresh", th_otsu))

    # Variant 4: small text on glossy labels.
    # Top-hat highlights bright fine strokes; close connects broken characters.
    kernel_tophat = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    fine = cv2.morphologyEx(up_small, cv2.MORPH_TOPHAT, kernel_tophat)
    th_small = cv2.adaptiveThreshold(
        fine, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 25, 4
    )
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    th_small = cv2.morphologyEx(th_small, cv2.MORPH_CLOSE, kernel_close, iterations=1)
    variants.append(("smalltext_tophat", th_small))

    return variants


def rotate_image(image: np.ndarray, angle: int) -> np.ndarray:
    """Rotate image for OCR orientation search."""
    if angle == 0:
        return image
    if angle == 90:
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    if angle == 180:
        return cv2.rotate(image, cv2.ROTATE_180)
    if angle == 270:
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return image


def is_date_like_token(text: str) -> bool:
    """Heuristic for expiry/manufacture-like tokens."""
    t = text.strip().lower()
    if not t:
        return False
    patterns = (
        r"(mfg|mfd|exp|bb|best|before|use\s*by|packed)",
        r"\b\d{1,2}[\/\-.]\d{1,2}[\/\-.]\d{2,4}\b",
        r"\b\d{4}[\/\-.]\d{1,2}[\/\-.]\d{1,2}\b",
    )
    return any(re.search(p, t) for p in patterns)


def run_ocr(image: np.ndarray, x_offset: int = 0, y_offset: int = 0) -> list:
    """
    Run Tesseract OCR and return list of (box, text, confidence).
    box is [[x1,y1],[x2,y2],[x3,y3],[x4,y4]] (clockwise from top-left).
    Uses PSM 11 (sparse text) which works best on product labels.
    """
    # Mix sparse-text and block-text assumptions; include single-line mode.
    psm_modes = (11, 6, 7)
    best_global_score = -1.0
    best_variant_preview = None
    best_results = []
    best_label = ""

    for angle in ROTATION_CANDIDATES:
        rot = rotate_image(image, angle)
        variants = build_preprocess_variants(rot)
        all_results = {}

        for name, variant in variants:
            variant_score = 0.0
            for psm in psm_modes:
                data = pytesseract.image_to_data(
                    variant,
                    config=(
                        f'--oem 1 --psm {psm} '
                        '-c preserve_interword_spaces=1 '
                        '-c user_defined_dpi=300'
                    ),
                    output_type=pytesseract.Output.DICT,
                )

                n = len(data["text"])
                for i in range(n):
                    text = data["text"][i].strip()
                    try:
                        conf = int(float(data["conf"][i]))
                    except ValueError:
                        continue

                    if not text or conf < 0:
                        continue

                    # Weight likely date/expiry tokens higher.
                    token_weight = 2.5 if is_date_like_token(text) else 1.0
                    if conf >= CONFIDENCE_MIN:
                        variant_score += conf * token_weight

                    scale = SMALL_TEXT_UPSCALE_FACTOR if "smalltext" in name else UPSCALE_FACTOR
                    x = int(data["left"][i] / scale)
                    y = int(data["top"][i] / scale)
                    w = int(data["width"][i] / scale)
                    h = int(data["height"][i] / scale)
                    box = [[x, y], [x + w, y], [x + w, y + h], [x, y + h]]

                    key = text.lower()
                    prev = all_results.get(key)
                    if prev is None or conf > prev[2]:
                        all_results[key] = (box, text, conf)

            if variant_score > best_global_score:
                best_global_score = variant_score
                best_variant_preview = variant
                best_label = f"{name} @ {angle}deg"
                best_results = list(all_results.values())

    # Convert best token boxes back to absolute frame coordinates.
    mapped_results = []
    h, w = image.shape[:2]
    selected_angle = int(best_label.split("@")[-1].replace("deg", "").strip()) if best_label else 0
    for box, text, conf in best_results:
        if selected_angle == 0:
            mapped = [[pt[0] + x_offset, pt[1] + y_offset] for pt in box]
        elif selected_angle == 180:
            mapped = [[(w - pt[0]) + x_offset, (h - pt[1]) + y_offset] for pt in box]
        else:
            mapped = [[pt[0] + x_offset, pt[1] + y_offset] for pt in box]
        mapped_results.append((mapped, text, conf))

    print(f"  OCR best variant: {best_label} (score={best_global_score:.1f})")
    return mapped_results, best_variant_preview


def is_plausible_label_text(text: str) -> bool:
    """
    Filter out OCR noise such as standalone punctuation and tiny fragments.
    Keeps words/numbers typically seen on beverage labels.
    """
    t = text.strip()
    if len(t) < MIN_TEXT_LEN:
        return False

    alnum = sum(ch.isalnum() for ch in t)
    if alnum == 0:
        return False

    # Require at least half the token to be letters/digits.
    if (alnum / len(t)) < 0.5:
        return False

    return True


def draw_results(frame: np.ndarray, results: list) -> np.ndarray:
    """Draw bounding boxes and text on the frame."""
    overlay = frame.copy()

    for box, text, confidence in results:
        if confidence < CONFIDENCE_MIN or not is_plausible_label_text(text):
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

    filtered = [
        (t, c) for _, t, c in results
        if c >= CONFIDENCE_MIN and is_plausible_label_text(t)
    ]

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
    print(f"Capturing burst ({BURST_COUNT} frames @ {BURST_INTERVAL_SEC:.2f}s)...")
    frame = capture_best_burst_frame(cam)
    cam.stop()
    cam.close()

    ts = time.strftime("%Y%m%d_%H%M%S")
    raw_path = os.path.join(OUTPUT_DIR, f"raw_{ts}.jpg")
    cv2.imwrite(raw_path, frame)
    print(f"  Raw frame saved  → {raw_path}")

    # ── 4. Preprocess ────────────────────────────────────────
    roi, (x0, y0) = extract_label_roi(frame)
    print("Preprocessing ROI (blur check → sharpen → CLAHE → denoise + variants)...")
    processed = preprocess(roi)

    proc_path = os.path.join(OUTPUT_DIR, f"processed_{ts}.jpg")
    cv2.imwrite(proc_path, processed)
    print(f"  Processed frame  → {proc_path}")

    # ── 5. Run OCR ───────────────────────────────────────────
    print("Running Tesseract OCR (PSM 11 — sparse text)...")
    t0 = time.time()
    results, preview_variant = run_ocr(roi, x_offset=x0, y_offset=y0)
    elapsed = time.time() - t0
    print(f"OCR completed in {elapsed:.2f}s")

    # ── 6. Print & save annotated output ─────────────────────
    print_results(results)

    # Draw ROI box to help camera framing.
    roi_h, roi_w = roi.shape[:2]
    cv2.rectangle(frame, (x0, y0), (x0 + roi_w, y0 + roi_h), (255, 200, 0), 2)
    overlay = draw_results(frame, results)
    overlay_path = os.path.join(OUTPUT_DIR, f"annotated_{ts}.jpg")
    cv2.imwrite(overlay_path, overlay)
    print(f"  Annotated frame  → {overlay_path}")

    # Also dump full OCR text for debugging
    full_text = pytesseract.image_to_string(
        preview_variant if preview_variant is not None else processed,
        config='--psm 6'
    ).strip()
    print("\n── Full OCR text (unfiltered) ──")
    print(full_text if full_text else "(empty)")
    print("── End ──\n")

    print("Done! Check the output images in:")
    print(f"  {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
