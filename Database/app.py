"""
Fridge Inventory — Flask Web Server
Provides a REST API and web interface for managing fridge items.
Integrates with the camera_project modules for Pi Camera capture + OCR + expiry detection.
"""

import os
import sys
import sqlite3
import shutil
import logging
import time
from threading import Lock
from io import BytesIO
from datetime import datetime, date
from pathlib import Path
from flask import Flask, jsonify, request, render_template, send_from_directory, abort, Response, stream_with_context

# ── Path setup: make camera_project importable ─────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent  # /home/pasan/smartf
CAMERA_PROJECT = PROJECT_ROOT / "camera_project"
sys.path.insert(0, str(CAMERA_PROJECT))
sys.path.insert(0, str(CAMERA_PROJECT / "src"))

# ── Camera / OCR imports (graceful fallback) ───────────────────────────────
try:
    import config.config as cam_config
    from src.camera_capture import CameraCapture
    from src.ocr_engine import OCREngine
    from src.expiration_detector import ExpirationDetector
    CAMERA_PIPELINE_AVAILABLE = True
except ImportError as e:
    CAMERA_PIPELINE_AVAILABLE = False
    print(f"[WARN] Camera pipeline not fully available: {e}")

# Ensure Flask knows where to find templates and static files when the module
# is imported dynamically by the top-level launcher. Set explicit paths to the
# Database/ templates and static folders so Jinja can find index.html.
APP_DIR = Path(__file__).resolve().parent
app = Flask(__name__, template_folder=str(APP_DIR / "templates"), static_folder=str(APP_DIR / "static"))
logger = logging.getLogger(__name__)

# Directory where item images are served from
UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ── SQLite helpers ─────────────────────────────────────────────────────────
DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fridge.db")


def get_connection():
    """Get a database connection with row factory enabled."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init_db():
    """Create the items table if it doesn't exist."""
    conn = get_connection()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS items (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            image_file  TEXT    NOT NULL,
            expiry_date TEXT    NOT NULL,
            name        TEXT    DEFAULT '',
            created_at  TEXT    DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()
    print(f"[DB] Database initialized at {DB_PATH}")


def add_item(image_file, expiry_date, name=""):
    """Insert a new item into the database."""
    conn = get_connection()
    cursor = conn.execute(
        "INSERT INTO items (image_file, expiry_date, name) VALUES (?, ?, ?)",
        (image_file, expiry_date, name),
    )
    item_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return item_id


def get_all_items():
    """Fetch all items, sorted by expiry date (soonest first)."""
    conn = get_connection()
    rows = conn.execute("SELECT * FROM items ORDER BY expiry_date ASC").fetchall()
    conn.close()
    return [dict(row) for row in rows]


def get_item(item_id):
    """Fetch a single item by its ID."""
    conn = get_connection()
    row = conn.execute("SELECT * FROM items WHERE id = ?", (item_id,)).fetchone()
    conn.close()
    return dict(row) if row else None


def update_item(item_id, image_file=None, expiry_date=None, name=None):
    """Update an existing item. Only provided fields are updated."""
    conn = get_connection()
    item = conn.execute("SELECT * FROM items WHERE id = ?", (item_id,)).fetchone()
    if not item:
        conn.close()
        return False

    new_image = image_file if image_file is not None else item["image_file"]
    new_expiry = expiry_date if expiry_date is not None else item["expiry_date"]
    new_name = name if name is not None else item["name"]

    conn.execute(
        "UPDATE items SET image_file = ?, expiry_date = ?, name = ? WHERE id = ?",
        (new_image, new_expiry, new_name, item_id),
    )
    conn.commit()
    conn.close()
    return True


def delete_item(item_id):
    """Delete an item by its ID and return its image filename."""
    conn = get_connection()
    row = conn.execute("SELECT image_file FROM items WHERE id = ?", (item_id,)).fetchone()

    if not row:
        conn.close()
        return None

    image_file = row["image_file"]
    conn.execute("DELETE FROM items WHERE id = ?", (item_id,))
    conn.commit()
    conn.close()
    return image_file


def get_item_count():
    """Get total number of items in the database."""
    conn = get_connection()
    count = conn.execute("SELECT COUNT(*) FROM items").fetchone()[0]
    conn.close()
    return count

# ── Lazy-initialized camera singleton ──────────────────────────────────────
_camera = None
_preview_lock = Lock()

def get_camera():
    """Lazy-init camera so it's only opened when first capture is requested."""
    global _camera
    if _camera is None and CAMERA_PIPELINE_AVAILABLE:
        _camera = CameraCapture(cam_config)
    return _camera


# ── Web UI ──────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    """Serve the main web interface."""
    return render_template("index.html")


# ── Image serving ──────────────────────────────────────────────────────────

@app.route("/uploads/<path:filename>")
def serve_image(filename):
    """Serve item images from the uploads directory."""
    return send_from_directory(UPLOAD_FOLDER, filename)


def _build_preview_placeholder(title, subtitle):
    """Create a lightweight JPEG placeholder for the preview panel."""
    from PIL import Image, ImageDraw, ImageFont

    image = Image.new("RGB", (960, 540), color=(26, 29, 39))
    draw = ImageDraw.Draw(image)

    # Add a subtle frame so the preview panel still feels intentional when
    # the camera is unavailable or not ready yet.
    draw.rounded_rectangle((20, 20, 940, 520), radius=24, outline=(43, 46, 61), width=3)
    draw.rounded_rectangle((60, 60, 900, 480), radius=18, fill=(19, 22, 31), outline=(43, 46, 61), width=2)

    try:
        title_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 42)
        body_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 24)
    except Exception:
        title_font = ImageFont.load_default()
        body_font = ImageFont.load_default()

    draw.text((100, 170), title, fill=(228, 230, 240), font=title_font)
    draw.text((100, 240), subtitle, fill=(139, 143, 163), font=body_font)
    draw.text((100, 340), "The live feed appears here once the camera is ready.", fill=(0, 206, 201), font=body_font)

    output = BytesIO()
    image.save(output, format="JPEG", quality=90)
    return output.getvalue()


def _frame_to_jpeg_bytes(frame):
    """Convert a camera frame array into JPEG bytes for the preview panel."""
    from PIL import Image

    if frame is None:
        return None

    if len(frame.shape) == 3 and frame.shape[2] == 3:
        image = Image.fromarray(frame[:, :, ::-1])
    else:
        image = Image.fromarray(frame)

    image.thumbnail((640, 360))
    output = BytesIO()
    image.save(output, format="JPEG", quality=72, optimize=True)
    return output.getvalue()


def _preview_frame_bytes():
    """Get a JPEG-encoded preview frame with lightweight fallback handling."""
    if not CAMERA_PIPELINE_AVAILABLE:
        return _build_preview_placeholder(
            "Camera unavailable",
            "Install or reconnect the camera to see a live feed."
        )

    camera = get_camera()
    if camera is None or camera.camera is None:
        return _build_preview_placeholder(
            "Camera not ready",
            "The preview panel will switch to the live feed automatically."
        )

    with _preview_lock:
        frame = camera.get_preview_frame() if hasattr(camera, "get_preview_frame") else camera.get_frame()
        preview_bytes = _frame_to_jpeg_bytes(frame)

    if preview_bytes is None:
        return _build_preview_placeholder(
            "Preview unavailable",
            "The camera will keep retrying in the background."
        )

    return preview_bytes


# ── REST API — Items ───────────────────────────────────────────────────────

@app.route("/api/items", methods=["GET"])
def api_get_items():
    """
    GET /api/items
    Returns all items sorted by expiry date.
    Each item includes an 'expiry_status' field: 'expired', 'warning', or 'fresh'.
    """
    items = get_all_items()
    today = date.today()

    for item in items:
        try:
            exp = datetime.strptime(item["expiry_date"], "%Y-%m-%d").date()
            delta = (exp - today).days
            if delta < 0:
                item["expiry_status"] = "expired"
            elif delta <= 3:
                item["expiry_status"] = "warning"
            else:
                item["expiry_status"] = "fresh"
            item["days_left"] = delta
        except (ValueError, TypeError):
            item["expiry_status"] = "unknown"
            item["days_left"] = None

    return jsonify({
        "items": items,
        "total": len(items)
    })


@app.route("/api/items", methods=["POST"])
def api_add_item():
    """
    POST /api/items
    Add a new item to the inventory.

    Accepts JSON body:
    {
        "image_file":  "IMG_001.jpg",       (required)
        "expiry_date": "2025-06-15",        (required, YYYY-MM-DD)
        "name":        "Milk"               (optional)
    }
    """
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "Request body must be JSON"}), 400

    image_file = data.get("image_file", "").strip()
    expiry_date = data.get("expiry_date", "").strip()
    name = data.get("name", "").strip()

    if not image_file:
        return jsonify({"error": "image_file is required"}), 400
    if not expiry_date:
        return jsonify({"error": "expiry_date is required"}), 400

    # Validate date format
    try:
        datetime.strptime(expiry_date, "%Y-%m-%d")
    except ValueError:
        return jsonify({"error": "expiry_date must be in YYYY-MM-DD format"}), 400

    item_id = add_item(image_file, expiry_date, name)

    return jsonify({
        "message": "Item added successfully",
        "id": item_id
    }), 201


@app.route("/api/items/<int:item_id>", methods=["GET"])
def api_get_item(item_id):
    """GET /api/items/<id> — Fetch a single item."""
    item = get_item(item_id)
    if not item:
        return jsonify({"error": "Item not found"}), 404
    return jsonify(item)


@app.route("/api/items/<int:item_id>", methods=["PUT"])
def api_update_item(item_id):
    """
    PUT /api/items/<id>
    Update an item's fields. Only include fields you want to change.

    Accepts JSON body with any of:
    {
        "image_file":  "new_image.jpg",
        "expiry_date": "2025-07-20",
        "name":        "Updated Name"
    }
    """
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "Request body must be JSON"}), 400

    # Validate date format if provided
    if "expiry_date" in data and data["expiry_date"]:
        try:
            datetime.strptime(data["expiry_date"], "%Y-%m-%d")
        except ValueError:
            return jsonify({"error": "expiry_date must be in YYYY-MM-DD format"}), 400

    success = update_item(
        item_id,
        image_file=data.get("image_file"),
        expiry_date=data.get("expiry_date"),
        name=data.get("name"),
    )

    if not success:
        return jsonify({"error": "Item not found"}), 404

    return jsonify({"message": "Item updated successfully"})


@app.route("/api/items/<int:item_id>", methods=["DELETE"])
def api_delete_item(item_id):
    """
    DELETE /api/items/<id>
    Delete an item and optionally remove its image file.
    """
    image_file = delete_item(item_id)
    if image_file is None:
        return jsonify({"error": "Item not found"}), 404

    # Try to remove the image file
    image_path = os.path.join(UPLOAD_FOLDER, image_file)
    if os.path.exists(image_path):
        try:
            os.remove(image_path)
        except OSError:
            pass  # Non-critical if file removal fails

    return jsonify({"message": "Item deleted successfully"})


@app.route("/api/stats", methods=["GET"])
def api_stats():
    """GET /api/stats — Dashboard statistics."""
    items = get_all_items()
    today = date.today()
    expired = 0
    expiring_soon = 0
    fresh = 0

    for item in items:
        try:
            exp = datetime.strptime(item["expiry_date"], "%Y-%m-%d").date()
            delta = (exp - today).days
            if delta < 0:
                expired += 1
            elif delta <= 3:
                expiring_soon += 1
            else:
                fresh += 1
        except (ValueError, TypeError):
            pass

    return jsonify({
        "total": len(items),
        "expired": expired,
        "expiring_soon": expiring_soon,
        "fresh": fresh,
    })


# ── REST API — Camera Capture ──────────────────────────────────────────────

@app.route("/api/capture", methods=["POST"])
def api_capture():
    """
    POST /api/capture
    Trigger the Pi Camera to:
      1. Capture a photo
      2. Run OCR to extract text
      3. Detect expiration date
      4. Copy image to uploads/ and save item to database

    Optional JSON body:
    {
        "name": "Milk"    (optional label for the item)
    }

    Returns:
    {
        "message": "...",
        "item_id": 5,
        "image_file": "photo_20260506_160000_0001.jpg",
        "expiry_date": "2026-08-15",
        "ocr_text": "...",
        "confidence": 0.9
    }
    """
    # Allow forcing a simulated capture from the client by sending {"simulate": true}
    # This bypasses the CAMERA_PIPELINE_AVAILABLE gate so the UI can still test
    # the capture/OCR/detection flow when hardware or native libs are missing.
    request_data = request.get_json(silent=True) or {}
    simulate = bool(request_data.get("simulate", False))

    if not CAMERA_PIPELINE_AVAILABLE and not simulate:
        return jsonify({
            "error": "Camera pipeline not available. Check that picamera2, pytesseract, and opencv are installed."
        }), 503

    data = request.get_json(silent=True) or {}
    item_name = data.get("name", "").strip()

    try:
        # 1. Capture photo
        camera = get_camera()

        # If client requested simulation or camera couldn't be initialized,
        # CameraCapture.capture_photo() will use the built-in simulation path.
        image_path = None
        if simulate:
            # ensure we have a CameraCapture instance for simulation utilities
            if _camera is None:
                try:
                    _ = CameraCapture(cam_config)
                except Exception:
                    pass
            image_path = CameraCapture(cam_config)._simulate_capture() if 'CameraCapture' in globals() else None
        else:
            if camera is None:
                return jsonify({"error": "Failed to initialize camera"}), 500
            image_path = camera.capture_photo()
        if not image_path:
            return jsonify({"error": "Camera capture failed"}), 500

        image_filename = os.path.basename(image_path)
        logger.info(f"[CAPTURE] Photo saved: {image_filename}")

        # 2. OCR — extract text from the captured image
        ocr = OCREngine(cam_config)
        ocr_result = ocr.extract_text(image_path)

        extracted_text = ""
        ocr_meta = {}
        if ocr_result and ocr_result.get("status") == "success":
            extracted_text = ocr_result.get("full_text", "")
            ocr_meta = ocr_result.get("_meta", {})
            logger.info(
                "[CAPTURE] OCR text: %s",
                extracted_text.replace('\n', ' ').strip()[:200] or "(empty)",
            )
            logger.info(
                "[CAPTURE] OCR strategy=%s blur=%.1f roi=%s",
                ocr_meta.get("strategy", "?"),
                ocr_meta.get("blur_score", -1),
                ocr_meta.get("roi", {}).get("source", "?"),
            )
        else:
            logger.warning("[CAPTURE] OCR failed or returned no text: %s",
                           (ocr_result or {}).get("error", "unknown"))

        # 3. Detect expiration date
        detector = ExpirationDetector(cam_config)
        detection = detector.detect_expiration(extracted_text)

        expiry_date = None
        confidence = 0
        if detection and detection.get("primary_date"):
            expiry_date = detection["primary_date"].get("formatted")
            confidence = detection.get("confidence", 0)
            logger.info(f"[CAPTURE] Expiry detected: {expiry_date} ({confidence:.0%})")
        else:
            logger.warning("[CAPTURE] No expiration date detected from text: %s",
                           extracted_text.replace('\n', ' ').strip()[:100] or "(empty)")

        # 4. Copy image to uploads/ so the web UI can serve it
        dest_path = os.path.join(UPLOAD_FOLDER, image_filename)
        shutil.copy2(image_path, dest_path)

        # 5. Save to database
        # Use today's date + 30 days as fallback if no expiry detected
        if not expiry_date:
            from datetime import timedelta
            expiry_date = (date.today() + timedelta(days=30)).isoformat()
            confidence = 0

        item_id = add_item(image_filename, expiry_date, item_name)
        logger.info(f"[CAPTURE] Item #{item_id} added to database")

        return jsonify({
            "message": "Capture successful",
            "item_id": item_id,
            "image_file": image_filename,
            "expiry_date": expiry_date,
            "ocr_text": extracted_text[:500],  # Truncate for response
            "confidence": confidence,
            "expiry_detected": confidence > 0,
        }), 201

    except Exception as e:
        logger.exception("[CAPTURE] Error during capture pipeline")
        return jsonify({"error": f"Capture failed: {str(e)}"}), 500


@app.route("/api/capture/status", methods=["GET"])
def api_capture_status():
    """GET /api/capture/status — Check if camera pipeline is available."""
    return jsonify({
        "camera_available": CAMERA_PIPELINE_AVAILABLE,
        "camera_initialized": _camera is not None,
    })


@app.route("/api/capture/preview", methods=["GET"])
def api_capture_preview():
    """GET /api/capture/preview — Return a live camera frame for the dashboard."""
    try:
        preview_bytes = _preview_frame_bytes()
        response = Response(preview_bytes, mimetype="image/jpeg")
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
        return response
    except Exception as e:
        logger.warning(f"Preview unavailable: {e}")
        preview_bytes = _build_preview_placeholder(
            "Preview unavailable",
            "The camera will keep retrying in the background."
        )
        response = Response(preview_bytes, mimetype="image/jpeg")
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
        return response


@app.route("/api/capture/stream", methods=["GET"])
def api_capture_stream():
    """GET /api/capture/stream — Return a continuous MJPEG preview stream."""
    def generate_frames():
        try:
            while True:
                frame_bytes = _preview_frame_bytes()
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
                )
                time.sleep(0.08)
        except GeneratorExit:
            return

    response = Response(
        stream_with_context(generate_frames()),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response


# ── Main ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    init_db()
    print("=" * 50)
    print("  🧊 Fridge Inventory Server")
    print(f"  Camera pipeline: {'✓ Available' if CAMERA_PIPELINE_AVAILABLE else '✗ Not available'}")
    print("  Access from any device on your network:")
    print("  http://<raspberry-pi-ip>:5000")
    print("=" * 50)
    app.run(host="0.0.0.0", port=5000, debug=False)
