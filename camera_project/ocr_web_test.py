"""
SmartFridge OCR Web Test

Standalone Flask app for live camera preview plus OCR / expiration detection
results in the browser.

Run:
    python camera_project/ocr_web_test.py

Optional:
    python camera_project/ocr_web_test.py --port 5001 --interval 2.5
"""

import argparse
import json
import logging
import sys
import tempfile
import threading
import time
from datetime import datetime
from io import BytesIO
from pathlib import Path

from flask import Flask, Response, jsonify, render_template_string, stream_with_context


PROJECT_ROOT = Path(__file__).resolve().parent.parent
CAMERA_PROJECT = PROJECT_ROOT / "camera_project"
SRC_DIR = CAMERA_PROJECT / "src"

if str(CAMERA_PROJECT) not in sys.path:
    sys.path.insert(0, str(CAMERA_PROJECT))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

try:
    import config.config as cam_config
    from src.camera_capture import CameraCapture
    from camera_project.backup.ocr_engine import OCREngine
    from src.expiration_detector import ExpirationDetector
    PIPELINE_AVAILABLE = True
except Exception as exc:
    PIPELINE_AVAILABLE = False
    PIPELINE_IMPORT_ERROR = str(exc)

try:
    from PIL import Image
except Exception:
    Image = None


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("smartfridge-ocr-web-test")

app = Flask(__name__)

camera = None
camera_lock = threading.Lock()
ocr_lock = threading.Lock()
ocr_state_lock = threading.Lock()
latest_result = {
    "status": "starting",
    "text": "",
    "expiry": None,
    "detected_dates": [],
    "primary_date": None,
    "confidence": 0,
    "updated_at": None,
    "error": None,
}


HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>SmartFridge OCR Web Test</title>
  <style>
    :root {
      --bg: #0b1020;
      --panel: #11182d;
      --panel-2: #0f1527;
      --line: rgba(255,255,255,.08);
      --text: #e8ecf6;
      --muted: #98a2b3;
      --accent: #00cec9;
      --accent-2: #8b5cf6;
      --warn: #ffb020;
      --error: #ff6b6b;
      --radius: 18px;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      min-height: 100vh;
      font-family: Inter, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background:
        radial-gradient(circle at top left, rgba(0, 206, 201, .10), transparent 30%),
        radial-gradient(circle at top right, rgba(139, 92, 246, .14), transparent 28%),
        linear-gradient(180deg, #070b16, var(--bg));
      color: var(--text);
    }
    .wrap {
      max-width: 1400px;
      margin: 0 auto;
      padding: 24px;
    }
    .hero {
      display: flex;
      align-items: flex-end;
      justify-content: space-between;
      gap: 16px;
      margin-bottom: 20px;
    }
    .title h1 {
      margin: 0 0 6px;
      font-size: clamp(1.6rem, 2vw, 2.4rem);
    }
    .title p {
      margin: 0;
      color: var(--muted);
    }
    .pill {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      padding: 8px 12px;
      border-radius: 999px;
      background: rgba(0, 206, 201, .12);
      border: 1px solid rgba(0, 206, 201, .22);
      color: var(--accent);
      font-weight: 600;
      font-size: .88rem;
    }
    .grid {
      display: grid;
      grid-template-columns: 1.2fr .8fr;
      gap: 18px;
    }
    .card {
      background: linear-gradient(180deg, rgba(255,255,255,.03), rgba(255,255,255,0)), var(--panel);
      border: 1px solid var(--line);
      border-radius: var(--radius);
      box-shadow: 0 18px 50px rgba(0,0,0,.32);
      overflow: hidden;
    }
    .card-header {
      padding: 16px 18px;
      border-bottom: 1px solid var(--line);
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      background: rgba(255,255,255,.02);
    }
    .card-header h2 {
      margin: 0;
      font-size: 1rem;
    }
    .card-header span {
      color: var(--muted);
      font-size: .86rem;
    }
    .preview {
      aspect-ratio: 4 / 3;
      background: #02040a;
    }
    .preview img {
      width: 100%;
      height: 100%;
      object-fit: contain;
      display: block;
    }
    .body {
      padding: 18px;
    }
    .meta {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 12px;
      margin-bottom: 16px;
    }
    .meta .box {
      background: var(--panel-2);
      border: 1px solid var(--line);
      border-radius: 14px;
      padding: 12px;
    }
    .meta .label {
      display: block;
      color: var(--muted);
      font-size: .76rem;
      text-transform: uppercase;
      letter-spacing: .08em;
      margin-bottom: 8px;
    }
    .meta .value {
      font-size: .95rem;
      line-height: 1.45;
      white-space: pre-wrap;
      word-break: break-word;
    }
    pre {
      margin: 0;
      padding: 16px;
      background: #070b14;
      border: 1px solid var(--line);
      border-radius: 14px;
      color: #d6dbe8;
      white-space: pre-wrap;
      word-break: break-word;
      min-height: 340px;
      max-height: 520px;
      overflow: auto;
    }
    .footer-note {
      margin-top: 14px;
      color: var(--muted);
      font-size: .88rem;
    }
    .status-ok { color: var(--accent); }
    .status-warn { color: var(--warn); }
    .status-error { color: var(--error); }
    @media (max-width: 980px) {
      .grid { grid-template-columns: 1fr; }
      .hero { flex-direction: column; align-items: flex-start; }
    }
  </style>
</head>
<body>
  <div class="wrap">
    <div class="hero">
      <div class="title">
        <h1>SmartFridge OCR Web Test</h1>
        <p>Live camera feed on the left, detected text and expiry parsing on the right.</p>
      </div>
      <div class="pill" id="runtime-pill">Starting...</div>
    </div>

    <div class="grid">
      <section class="card">
        <div class="card-header">
          <h2>Live Feed</h2>
          <span id="camera-state">Connecting...</span>
        </div>
        <div class="preview">
          <img src="/stream" alt="Live camera feed">
        </div>
      </section>

      <section class="card">
        <div class="card-header">
          <h2>OCR Result</h2>
          <span id="ocr-updated">Waiting...</span>
        </div>
        <div class="body">
          <div class="meta">
            <div class="box">
              <span class="label">Detected text</span>
              <div class="value" id="ocr-text">Loading...</div>
            </div>
            <div class="box">
              <span class="label">Expiry detection</span>
              <div class="value" id="ocr-expiry">Loading...</div>
            </div>
          </div>
          <pre id="ocr-json">Loading...</pre>
          <div class="footer-note">
            OCR refreshes every few seconds from the live camera stream.
          </div>
        </div>
      </section>
    </div>
  </div>

  <script>
    const ocrTextEl = document.getElementById("ocr-text");
    const ocrExpiryEl = document.getElementById("ocr-expiry");
    const ocrJsonEl = document.getElementById("ocr-json");
    const updatedEl = document.getElementById("ocr-updated");
    const runtimePill = document.getElementById("runtime-pill");
    const cameraStateEl = document.getElementById("camera-state");

    async function refreshStatus() {
      try {
        const res = await fetch("/api/status");
        const data = await res.json();
        const ready = data.camera_ready;
        runtimePill.textContent = ready ? "Camera Ready" : "Camera Unavailable";
        runtimePill.className = "pill " + (ready ? "status-ok" : "status-warn");
        cameraStateEl.textContent = ready ? "Camera is live" : "Using fallback preview";
      } catch (error) {
        runtimePill.textContent = "Status error";
        runtimePill.className = "pill status-error";
      }
    }

    async function refreshOcr() {
      try {
        const res = await fetch("/api/result");
        const data = await res.json();
        const text = (data.text || "").trim();
        const expiry = data.primary_date ? JSON.stringify(data.primary_date) : "No expiry date detected";

        ocrTextEl.textContent = text || "[No text detected]";
        ocrExpiryEl.textContent = expiry;
        ocrJsonEl.textContent = JSON.stringify(data, null, 2);
        updatedEl.textContent = data.updated_at ? `Updated ${new Date(data.updated_at).toLocaleTimeString()}` : "Waiting...";
      } catch (error) {
        ocrTextEl.textContent = "OCR request failed";
        ocrExpiryEl.textContent = error.message || "Unknown error";
      }
    }

    refreshStatus();
    refreshOcr();
    setInterval(refreshStatus, 3000);
    setInterval(refreshOcr, 2500);
  </script>
</body>
</html>
"""


def get_camera():
    """Lazy-init the shared camera instance."""
    global camera
    if camera is None and PIPELINE_AVAILABLE:
        camera = CameraCapture(cam_config)
    return camera


def _frame_to_jpeg_bytes(frame):
    """Convert a numpy frame into JPEG bytes for the browser stream."""
    if frame is None or Image is None:
        return None

    if len(frame.shape) == 3 and frame.shape[2] == 3:
        image = Image.fromarray(frame[:, :, ::-1])
    else:
        image = Image.fromarray(frame)

    image.thumbnail((960, 720))
    output = BytesIO()
    image.save(output, format="JPEG", quality=80)
    return output.getvalue()


def _save_frame_for_ocr(frame):
    """Write a frame to a temporary file for OCR processing."""
    if frame is None or Image is None:
        return None

    if len(frame.shape) == 3 and frame.shape[2] == 3:
        image = Image.fromarray(frame[:, :, ::-1])
    else:
        image = Image.fromarray(frame)

    temp_dir = Path(tempfile.gettempdir())
    temp_path = temp_dir / "smartfridge_ocr_live_test.jpg"
    image.save(temp_path, format="JPEG", quality=90)
    return str(temp_path)


def _ocr_loop(interval_seconds):
    """Refresh OCR results in the background."""
    if not PIPELINE_AVAILABLE:
        with ocr_state_lock:
            latest_result.update({
                "status": "error",
                "error": f"Pipeline unavailable: {PIPELINE_IMPORT_ERROR}",
                "updated_at": datetime.now().isoformat(),
            })
        return

    ocr_engine = OCREngine(cam_config)
    detector = ExpirationDetector(cam_config)

    while True:
        try:
            cam = get_camera()
            if cam is None:
                with ocr_state_lock:
                    latest_result.update({
                        "status": "error",
                        "error": "Camera is not available.",
                        "updated_at": datetime.now().isoformat(),
                    })
                time.sleep(interval_seconds)
                continue

            with camera_lock:
                frame = cam.get_preview_frame() if hasattr(cam, "get_preview_frame") else cam.get_frame()

            temp_image_path = _save_frame_for_ocr(frame)
            if not temp_image_path:
                with ocr_state_lock:
                    latest_result.update({
                        "status": "error",
                        "error": "Could not convert camera frame for OCR.",
                        "updated_at": datetime.now().isoformat(),
                    })
                time.sleep(interval_seconds)
                continue

            with ocr_lock:
                ocr_result = ocr_engine.extract_text(temp_image_path)

            if not ocr_result or ocr_result.get("status") != "success":
                with ocr_state_lock:
                    latest_result.update({
                        "status": "error",
                        "error": (ocr_result or {}).get("error", "OCR failed"),
                        "updated_at": datetime.now().isoformat(),
                    })
                time.sleep(interval_seconds)
                continue

            text = ocr_result.get("full_text", "")
            detection = detector.detect_expiration(text)

            with ocr_state_lock:
                latest_result.update({
                    "status": "ok",
                    "text": text,
                    "words": ocr_result.get("words", []),
                    "confidence_values": ocr_result.get("confidence", []),
                    "expiry_lines": detection.get("expiry_lines", []),
                    "detected_dates": detection.get("detected_dates", []),
                    "primary_date": detection.get("primary_date"),
                    "confidence": detection.get("confidence", 0),
                    "raw": detection,
                    "updated_at": datetime.now().isoformat(),
                    "error": None,
                })

        except Exception as exc:
            logger.exception("OCR loop failed")
            with ocr_state_lock:
                latest_result.update({
                    "status": "error",
                    "error": str(exc),
                    "updated_at": datetime.now().isoformat(),
                })

        time.sleep(interval_seconds)


@app.route("/")
def index():
    return render_template_string(HTML)


@app.route("/stream")
def stream():
    def generate_frames():
        while True:
            cam = get_camera()
            if cam is None:
                if Image is not None:
                    placeholder = Image.new("RGB", (960, 720), color=(12, 16, 28))
                    out = BytesIO()
                    placeholder.save(out, format="JPEG", quality=85)
                    frame_bytes = out.getvalue()
                else:
                    frame_bytes = b""
            else:
                with camera_lock:
                    frame = cam.get_preview_frame() if hasattr(cam, "get_preview_frame") else cam.get_frame()
                frame_bytes = _frame_to_jpeg_bytes(frame)

            if not frame_bytes:
                time.sleep(0.1)
                continue

            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
            )
            time.sleep(0.08)

    return Response(
        stream_with_context(generate_frames()),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/api/result")
def api_result():
    with ocr_state_lock:
        return jsonify(latest_result)


@app.route("/api/status")
def api_status():
    cam = get_camera() if PIPELINE_AVAILABLE else None
    return jsonify({
        "pipeline_available": PIPELINE_AVAILABLE,
        "camera_ready": bool(cam and getattr(cam, "camera", None)),
        "camera_initialized": cam is not None,
        "ocr_running": True,
    })


def _start_background_workers(interval_seconds):
    thread = threading.Thread(target=_ocr_loop, args=(interval_seconds,), daemon=True)
    thread.start()


def main():
    parser = argparse.ArgumentParser(description="Live camera + OCR web test for SmartFridge")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=5001, help="Port to bind to")
    parser.add_argument("--interval", type=float, default=2.5, help="Seconds between OCR refreshes")
    args = parser.parse_args()

    _start_background_workers(args.interval)

    print("SmartFridge OCR web test starting...")
    print(f"Open: http://127.0.0.1:{args.port}")
    if not PIPELINE_AVAILABLE:
        print(f"Pipeline unavailable: {PIPELINE_IMPORT_ERROR}")

    app.run(host=args.host, port=args.port, debug=False, threaded=True)


if __name__ == "__main__":
    main()