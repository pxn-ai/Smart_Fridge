#!/usr/bin/env python3
"""
Bottle Label OCR Scanner
Raspberry Pi 5 + Camera Module V2 + PaddleOCR
"""

import cv2
import numpy as np
import time
from picamera2 import Picamera2
from paddleocr import PaddleOCR

# ── Configuration ─────────────────────────────────────────────
CAMERA_WIDTH    = 1920
CAMERA_HEIGHT   = 1080
PREVIEW_WIDTH   = 960   # downscaled for display window
PREVIEW_HEIGHT  = 540
CAPTURE_KEY     = ord('c')   # press C to capture & scan
QUIT_KEY        = ord('q')   # press Q to quit
CONFIDENCE_MIN  = 0.6        # ignore detections below this score
# ──────────────────────────────────────────────────────────────


def preprocess(frame: np.ndarray) -> np.ndarray:
    """
    Improve OCR accuracy for glossy bottle labels:
      1. Convert to grayscale
      2. CLAHE — boosts local contrast (helps with curved/shiny labels)
      3. Mild denoise
      4. Adaptive threshold → back to BGR for PaddleOCR
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Contrast Limited Adaptive Histogram Equalization
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    # Remove noise while preserving edges
    gray = cv2.fastNlMeansDenoising(gray, h=10)

    # Adaptive threshold — handles uneven lighting on curved surfaces
    thresh = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=11,
        C=2
    )

    # PaddleOCR expects a BGR image
    return cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)


def parse_results(results) -> list:
    """
    Normalise PaddleOCR output into a flat list of (box, text, confidence).
    Handles both the old list-of-lists format and the new Result object format.
    """
    lines = []
    try:
        # New API: results is a list of Result objects with a .boxes attribute
        for res in results:
            for box_obj in res.boxes:
                box  = box_obj.coordinate   # [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
                text = box_obj.rec_text
                conf = box_obj.rec_score
                lines.append((box, text, conf))
    except AttributeError:
        # Old API: results[0] is a list of [box, (text, score)]
        if results and results[0]:
            for line in results[0]:
                box, (text, conf) = line
                lines.append((box, text, conf))
    return lines


def draw_results(frame: np.ndarray, results) -> np.ndarray:
    """Draw bounding boxes and text on the frame."""
    overlay = frame.copy()

    for box, text, confidence in parse_results(results):
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


def print_results(results) -> None:
    """Print detected text to terminal."""
    print("\n" + "═" * 50)
    print("  DETECTED TEXT")
    print("═" * 50)

    filtered = [(t, c) for _, t, c in parse_results(results) if c >= CONFIDENCE_MIN]

    if not filtered:
        print("  No text detected.")
    else:
        for text, confidence in filtered:
            print(f"  {text:<35} ({confidence:.0%})")

    print("═" * 50 + "\n")


def main():
    print("Initialising PaddleOCR (first run downloads models ~60MB)...")
    ocr = PaddleOCR(
        use_textline_orientation=True,  # replaces deprecated use_angle_cls
        lang='en',
        show_log=False,
    )
    print("PaddleOCR ready.\n")

    print("Starting camera...")
    cam = Picamera2()
    config = cam.create_preview_configuration(
        main={"size": (CAMERA_WIDTH, CAMERA_HEIGHT), "format": "BGR888"}
    )
    cam.configure(config)
    cam.start()
    time.sleep(1.5)  # let exposure settle
    print("Camera ready.\n")

    print("Controls:")
    print(f"  [C]  — Capture frame and run OCR")
    print(f"  [Q]  — Quit\n")

    last_overlay = None

    while True:
        # Grab live frame
        frame = cam.capture_array()
        display = cv2.resize(frame, (PREVIEW_WIDTH, PREVIEW_HEIGHT))

        # Show last OCR result overlaid on preview
        if last_overlay is not None:
            show = cv2.resize(last_overlay, (PREVIEW_WIDTH, PREVIEW_HEIGHT))
        else:
            show = display.copy()
            cv2.putText(show, "Press C to scan", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 220, 255), 2)

        cv2.imshow("Bottle Label OCR  |  C=Scan  Q=Quit", show)

        key = cv2.waitKey(1) & 0xFF

        if key == QUIT_KEY:
            break

        elif key == CAPTURE_KEY:
            print("Scanning...")
            t0 = time.time()

            processed = preprocess(frame)
            results = ocr.ocr(processed, cls=True)

            elapsed = time.time() - t0
            print(f"Done in {elapsed:.2f}s")

            last_overlay = draw_results(frame, results)
            print_results(results)

    cam.stop()
    cv2.destroyAllWindows()
    print("Bye!")


if __name__ == "__main__":
    main()