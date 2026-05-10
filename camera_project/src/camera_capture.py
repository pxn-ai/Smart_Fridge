"""
Camera Capture Module
Captures photos triggered by button press with live preview
"""

import logging
import os
import threading
import time
from datetime import datetime
from pathlib import Path

try:
    from picamera2 import Picamera2
    from libcamera import controls
    PICAMERA_AVAILABLE = True
except Exception as e:
    PICAMERA_AVAILABLE = False
    print(f"Warning: picamera2 not available: {e}")

try:
    import cv2
    import numpy as np
    OPENCV_AVAILABLE = True
except Exception as e:
    OPENCV_AVAILABLE = False
    print(f"Warning: OpenCV not available for preview: {e}")

logger = logging.getLogger(__name__)


class CameraCapture:
    """Handles camera photo capture with live preview"""

    def __init__(self, config):
        """
        Initialize camera

        Args:
            config: Configuration module
        """
        self.config = config
        self.camera = None
        self.photo_count = 0
        self.last_frame = None
        self.preview_running = False
        self._camera_lock = threading.Lock()

        if PICAMERA_AVAILABLE:
            self._init_camera()
        else:
            logger.warning("Picamera2 not available - simulation mode")

    def _init_camera(self):
        """Initialize the Pi Camera with preview"""
        try:
            self.camera = Picamera2()
            config = self.camera.create_still_configuration(
                main={"size": self.config.CAMERA_RESOLUTION},
                lores={"size": (640, 480), "format": "BGR888"}
            )
            self.camera.configure(config)
            self.camera.start()
            self._apply_camera_tuning()
            logger.info("Camera initialized successfully")
            logger.info(f"Resolution: {self.config.CAMERA_RESOLUTION}")

        except Exception as e:
            logger.error(f"Error initializing camera: {e}")
            self.camera = None

    def start_preview(self):
        """Start live camera preview display"""
        if not OPENCV_AVAILABLE:
            logger.warning("OpenCV not available - preview disabled")
            return False

        try:
            cv2.namedWindow("Camera Preview", cv2.WINDOW_AUTOSIZE)
            logger.info("Preview window opened")
            self.preview_running = True
            return True

        except Exception as e:
            logger.error(f"Error starting preview: {e}")
            return False

    def get_frame(self):
        """
        Get current frame from camera

        Returns:
            Frame as numpy array or None
        """
        if not self.camera:
            return None

        try:
            with self._camera_lock:
                array = self.camera.capture_array("main")
                array = self._apply_orientation(array)
                self.last_frame = array
                return array

        except Exception as e:
            logger.error(f"Error getting frame: {e}")
            return None

    def get_preview_frame(self):
        """Get a lighter-weight frame for the web preview."""
        if not self.camera:
            return None

        try:
            with self._camera_lock:
                try:
                    return self._apply_orientation(self.camera.capture_array("lores"))
                except Exception:
                    return self._apply_orientation(self.camera.capture_array("main"))
        except Exception as e:
            logger.error(f"Error getting preview frame: {e}")
            return None

    def display_frame(self, frame, title="Camera Preview"):
        """Display frame in OpenCV window"""
        if not OPENCV_AVAILABLE or not self.preview_running:
            return

        try:
            # Resize for display if too large
            display_frame = frame.copy()
            if display_frame.shape[0] > 720:
                scale = 720 / display_frame.shape[0]
                new_size = (int(display_frame.shape[1] * scale), 720)
                display_frame = cv2.resize(display_frame, new_size)

            # Convert BGR to RGB if needed
            if len(display_frame.shape) == 3 and display_frame.shape[2] == 3:
                display_frame = cv2.cvtColor(display_frame, cv2.COLOR_RGB2BGR)

            cv2.imshow(title, display_frame)

        except Exception as e:
            logger.debug(f"Error displaying frame: {e}")

    def capture_photo(self):
        """
        Capture a single photo from current frame

        Returns:
            Path to captured image or None
        """
        try:
            if not self.camera or not PICAMERA_AVAILABLE:
                logger.warning("No frame available")
                return self._simulate_capture()

            with self._camera_lock:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                self.photo_count += 1
                filename = f"photo_{timestamp}_{self.photo_count:04d}.jpg"
                filepath = os.path.join(self.config.IMAGES_DIR, filename)

                # Save directly from the camera when OpenCV is not installed.
                if OPENCV_AVAILABLE:
                    frame = self._capture_sharpest_frame()
                    frame = self._apply_orientation(frame)
                    cv2.imwrite(filepath, frame)
                else:
                    self.camera.capture_file(filepath)

            logger.info(f"✓ Photo saved: {filepath}")
            return filepath

        except Exception as e:
            logger.error(f"Error capturing photo: {e}")
            return None

    def _capture_sharpest_frame(self, attempts=3, delay=0.05):
        """Capture a short burst and return the sharpest frame."""
        best_frame = None
        best_score = -1.0

        for _ in range(max(1, attempts)):
            frame = self.camera.capture_array("main")
            score = self._sharpness_score(frame)
            if score > best_score:
                best_score = score
                best_frame = frame
            if delay:
                time.sleep(delay)

        if best_frame is not None:
            logger.info("Selected sharpest capture frame (score=%.1f)", best_score)
            self.last_frame = best_frame
            return best_frame

        return self.camera.capture_array("main")

    def _apply_orientation(self, frame):
        """Rotate and flip frames according to the configured camera mount."""
        if frame is None:
            return None

        rotated = frame
        rotation = int(getattr(self.config, "CAMERA_ROTATION", 0) or 0)
        if rotation == 90:
            rotated = cv2.rotate(rotated, cv2.ROTATE_90_CLOCKWISE)
        elif rotation == 180:
            rotated = cv2.rotate(rotated, cv2.ROTATE_180)
        elif rotation == 270:
            rotated = cv2.rotate(rotated, cv2.ROTATE_90_COUNTERCLOCKWISE)

        if bool(getattr(self.config, "CAMERA_FLIP_HORIZONTAL", False)):
            rotated = cv2.flip(rotated, 1)
        if bool(getattr(self.config, "CAMERA_FLIP_VERTICAL", False)):
            rotated = cv2.flip(rotated, 0)

        return rotated

    def _apply_camera_tuning(self):
        """Bias the sensor toward short, sharp captures for OCR."""
        if not self.camera:
            return

        controls_to_set = {
            "AeEnable": bool(getattr(self.config, "CAMERA_AE_ENABLE", True)),
            "AeMeteringMode": int(getattr(self.config, "CAMERA_AE_METERING_MODE", 2)),
            "AeExposureMode": int(getattr(self.config, "CAMERA_AE_EXPOSURE_MODE", 0)),
            "AwbEnable": bool(getattr(self.config, "CAMERA_AWB_ENABLE", True)),
            "ExposureTime": int(getattr(self.config, "CAMERA_EXPOSURE_TIME", 10000)),
            "AnalogueGain": float(getattr(self.config, "CAMERA_ANALOGUE_GAIN", 1.8)),
            "FrameDurationLimits": (
                int(getattr(self.config, "CAMERA_FRAME_DURATION_MIN", 10000)),
                int(getattr(self.config, "CAMERA_FRAME_DURATION_MAX", 16666)),
            ),
        }

        try:
            self.camera.set_controls(controls_to_set)
            logger.info("Applied OCR-oriented camera tuning")
        except Exception as exc:
            logger.warning("Could not apply camera tuning: %s", exc)

    def _sharpness_score(self, frame):
        """Estimate focus quality with Laplacian variance when OpenCV is available."""
        if not OPENCV_AVAILABLE:
            return 0.0

        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            return float(cv2.Laplacian(gray, cv2.CV_64F).var())
        except Exception:
            return 0.0

    def _simulate_capture(self):
        """Simulate a photo capture (for testing without camera)"""
        try:
            from PIL import Image, ImageDraw, ImageFont

            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            self.photo_count += 1
            filename = f"photo_{timestamp}_{self.photo_count:04d}.jpg"
            filepath = os.path.join(self.config.IMAGES_DIR, filename)

            # Create a test image
            width, height = 400, 300
            image = Image.new('RGB', (width, height), color='white')
            draw = ImageDraw.Draw(image)

            try:
                font = ImageFont.truetype(
                    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18
                )
            except:
                font = ImageFont.load_default()

            text_lines = [
                "PRODUCT SAMPLE",
                f"Captured: {timestamp}",
                "",
                "Best Before:",
                "06/15/2025",
                "",
                "Keep Refrigerated"
            ]

            y_offset = 30
            for line in text_lines:
                draw.text((20, y_offset), line, fill='black', font=font)
                y_offset += 35

            image.save(filepath, 'JPEG')
            logger.info(f"✓ Simulation photo saved: {filepath}")
            return filepath

        except Exception as e:
            logger.error(f"Error simulating capture: {e}")
            return None

    def stop_preview(self):
        """Stop preview display"""
        if OPENCV_AVAILABLE:
            try:
                cv2.destroyAllWindows()
                self.preview_running = False
                logger.info("Preview stopped")
            except Exception as e:
                logger.error(f"Error stopping preview: {e}")

    def cleanup(self):
        """Cleanup camera resources"""
        self.stop_preview()
        if self.camera:
            try:
                self.camera.stop()
                self.camera.close()
                logger.info("Camera cleanup complete")
            except Exception as e:
                logger.error(f"Error cleaning up camera: {e}")
