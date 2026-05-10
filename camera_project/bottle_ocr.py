#!/usr/bin/env python3
"""
Bottle Label OCR Scanner
Raspberry Pi 5 + Camera Module V2 + PaddleOCR
"""

import cv2
import numpy as np
import time
import platform
from picamera2 import Picamera2
from paddleocr import PaddleOCR
import pytesseract

# ── Configuration ─────────────────────────────────────────────
CAMERA_WIDTH    = 1920
CAMERA_HEIGHT   = 1080
PREVIEW_WIDTH   = 960   # downscaled for display window
PREVIEW_HEIGHT  = 540
CAPTURE_KEY     = ord('c')   # press C to capture & scan
QUIT_KEY        = ord('q')   # press Q to quit
CONFIDENCE_MIN  = 0.6        # ignore detections below this score
BLUR_WARN_SCORE = 30.0
BLUR_SOFT_SCORE = 80.0
MAX_SHARPEN_STRENGTH = 2.5
MIN_SHARPEN_STRENGTH = 0.8
SHARPEN_FACTOR = 120.0
CAMERA_WARMUP_SEC = 1.5
SAVE_DEBUG_IMAGES = False
OCR_ENGINE = "auto"          # "auto", "paddle", "tesseract"
TESSERACT_PSM = 11
ROTATE_180 = True            # IMX219 often reports 180-degree orientation
# ──────────────────────────────────────────────────────────────

# Reuse CLAHE object to avoid rebuilding on every scan.
CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


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
      6. Adaptive threshold -> BGR for PaddleOCR
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # ── Blur detection & report ───────────────────────────────
    score = blur_score(gray)
    if score < BLUR_WARN_SCORE:
        print(f"  WARNING: Very blurry (score: {score:.1f}) — try moving slightly further back.")
    elif score < BLUR_SOFT_SCORE:
        print(f"  NOTE: Soft image (score: {score:.1f}) — sharpening applied.")
    else:
        print(f"  OK: Sharpness good (score: {score:.1f})")

    # ── Unsharp mask sharpening ───────────────────────────────
    # Blurrier images get stronger sharpening, up to a sensible ceiling.
    # Formula: low score -> high strength, capped between 0.8 and 2.5.
    strength = max(MIN_SHARPEN_STRENGTH, min(MAX_SHARPEN_STRENGTH, SHARPEN_FACTOR / (score + 1)))
    gray = sharpen(gray, strength=strength)

    # ── CLAHE ─────────────────────────────────────────────────
    gray = CLAHE.apply(gray)

    # ── Denoise ───────────────────────────────────────────────
    gray = cv2.fastNlMeansDenoising(gray, h=10)

    # ── Adaptive threshold ────────────────────────────────────
    thresh = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=11,
        C=2
    )

    return cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)


def preprocess_tesseract(frame: np.ndarray) -> np.ndarray:
    """
    Tesseract-friendly preprocessing:
    grayscale -> adaptive sharpen -> CLAHE -> denoise (no hard thresholding).
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    score = blur_score(gray)
    if score < BLUR_WARN_SCORE:
        print(f"  WARNING: Very blurry (score: {score:.1f}) — try moving slightly further back.")
    elif score < BLUR_SOFT_SCORE:
        print(f"  NOTE: Soft image (score: {score:.1f}) — sharpening applied.")
    else:
        print(f"  OK: Sharpness good (score: {score:.1f})")

    strength = max(MIN_SHARPEN_STRENGTH, min(MAX_SHARPEN_STRENGTH, SHARPEN_FACTOR / (score + 1)))
    gray = sharpen(gray, strength=strength)
    gray = CLAHE.apply(gray)
    gray = cv2.fastNlMeansDenoising(gray, h=10)
    return gray


def parse_results(results) -> list:
    """
    Normalise PaddleOCR output into a flat list of (box, text, confidence).
    Handles both the old list-of-lists format and the new Result object format.
    """
    lines = []
    if not results:
        return lines

    first = results[0]
    if hasattr(first, "boxes"):
        # New API: results is a list of Result objects with a .boxes attribute.
        for res in results:
            for box_obj in res.boxes:
                box = box_obj.coordinate   # [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
                text = box_obj.rec_text
                conf = float(box_obj.rec_score)
                lines.append((box, text, conf))
        return lines

    # Old API: results[0] is a list of [box, (text, score)]
    if first:
        for line in first:
            box, (text, conf) = line
            lines.append((box, text, float(conf)))
    return lines


def run_tesseract_ocr(image: np.ndarray) -> list:
    """Run Tesseract OCR and return normalized (box, text, confidence[0..1])."""
    data = pytesseract.image_to_data(
        image,
        config=f"--psm {TESSERACT_PSM}",
        output_type=pytesseract.Output.DICT,
    )

    lines = []
    total = len(data["text"])
    for i in range(total):
        text = data["text"][i].strip()
        try:
            conf_raw = float(data["conf"][i])
        except ValueError:
            continue
        if not text or conf_raw < 0:
            continue

        x, y = data["left"][i], data["top"][i]
        w, h = data["width"][i], data["height"][i]
        box = [[x, y], [x + w, y], [x + w, y + h], [x, y + h]]
        lines.append((box, text, conf_raw / 100.0))
    return lines


def run_ocr(engine_name: str, ocr: PaddleOCR | None, image: np.ndarray) -> list:
    """
    Run OCR with cross-version compatibility.
    Older APIs accept cls=True on ocr(); newer 3.x APIs may reject it.
    """
    if engine_name == "tesseract":
        return run_tesseract_ocr(image)

    if ocr is None:
        return []

    try:
        results = ocr.ocr(image, cls=True)
    except TypeError:
        results = ocr.ocr(image)
    return parse_results(results)


def init_ocr_engine():
    """
    Pick and initialize OCR backend.
    On aarch64 Raspberry Pi, Paddle may segfault at inference; auto mode
    defaults to Tesseract for stability.
    """
    machine = platform.machine().lower()
    requested = OCR_ENGINE.lower().strip()
    use_tesseract = requested == "tesseract" or (requested == "auto" and "aarch64" in machine)

    if use_tesseract:
        version = pytesseract.get_tesseract_version()
        print(f"Tesseract OCR v{version} ready (engine=tesseract).\n")
        return "tesseract", None

    print("Initialising PaddleOCR (first run downloads models ~60MB)...")
    ocr_kwargs = {
        "use_textline_orientation": True,
        "lang": "en",
    }
    # PaddleOCR versions differ in accepted constructor args.
    # Keep startup compatible across 2.x/3.x releases.
    try:
        ocr = PaddleOCR(show_log=False, **ocr_kwargs)
    except (TypeError, ValueError):
        ocr = PaddleOCR(**ocr_kwargs)
    print("PaddleOCR ready (engine=paddle).\n")
    return "paddle", ocr


def draw_results(frame: np.ndarray, lines: list) -> np.ndarray:
    """Draw bounding boxes and text on the frame."""
    overlay = frame.copy()

    for box, text, confidence in lines:
        if confidence < CONFIDENCE_MIN:
            continue

        pts = np.array(box, dtype=np.int32)
        cv2.polylines(overlay, [pts], isClosed=True,
                      color=(0, 255, 0), thickness=2)

        x, y = int(box[0][0]), int(box[0][1]) - 8
        label = f"{text}  ({confidence:.0%})"
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(overlay, (x, y - h - 4), (x + w + 4, y + 4),
                      (0, 255, 0), -1)
        cv2.putText(overlay, label, (x + 2, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)

    return overlay


def print_results(lines: list) -> None:
    """Print detected text to terminal."""
    print("\n" + "=" * 50)
    print("  DETECTED TEXT")
    print("=" * 50)

    filtered = [(t, c) for _, t, c in lines if c >= CONFIDENCE_MIN]

    if not filtered:
        print("  No text detected.")
    else:
        for text, confidence in filtered:
            print(f"  {text:<35} ({confidence:.0%})")

    print("=" * 50 + "\n")


def main():
    engine_name, ocr = init_ocr_engine()

    print("Starting camera...")
    cam = Picamera2()
    config = cam.create_preview_configuration(
        main={"size": (CAMERA_WIDTH, CAMERA_HEIGHT), "format": "BGR888"}
    )
    cam.configure(config)
    cam.start()
    time.sleep(CAMERA_WARMUP_SEC)  # let exposure settle
    print("Camera ready.\n")

    print("Controls:")
    print("  [C]  -- Capture frame and run OCR")
    print("  [Q]  -- Quit\n")

    last_overlay = None

    while True:
        frame = cam.capture_array()
        if ROTATE_180:
            frame = cv2.rotate(frame, cv2.ROTATE_180)

        if last_overlay is not None:
            show = cv2.resize(last_overlay, (PREVIEW_WIDTH, PREVIEW_HEIGHT))
        else:
            show = cv2.resize(frame, (PREVIEW_WIDTH, PREVIEW_HEIGHT))
            cv2.putText(show, "Press C to scan", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 220, 255), 2)

        cv2.imshow("Bottle Label OCR  |  C=Scan  Q=Quit", show)

        key = cv2.waitKey(1) & 0xFF

        if key == QUIT_KEY:
            break

        elif key == CAPTURE_KEY:
            print("Scanning...")
            t0 = time.time()

            if engine_name == "tesseract":
                processed = preprocess_tesseract(frame)
            else:
                processed = preprocess(frame)
            lines = run_ocr(engine_name, ocr, processed)

            elapsed = time.time() - t0
            print(f"Done in {elapsed:.2f}s ({len(lines)} candidates)")

            if SAVE_DEBUG_IMAGES:
                ts = int(time.time())
                cv2.imwrite(f"ocr_raw_{ts}.jpg", frame)
                cv2.imwrite(f"ocr_processed_{ts}.jpg", processed)

            last_overlay = draw_results(frame, lines)
            print_results(lines)

    cam.stop()
    cv2.destroyAllWindows()
    print("Bye!")


if __name__ == "__main__":
    main()