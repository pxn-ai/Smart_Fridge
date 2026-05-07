"""
Camera Capture Module
Captures photos triggered by button press with live preview
"""

import logging
import os
import threading
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
                    return self.camera.capture_array("lores")
                except Exception:
                    return self.camera.capture_array("main")
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
                    if self.last_frame is not None:
                        frame = self.last_frame
                    else:
                        frame = self.camera.capture_array("main")

                    cv2.imwrite(filepath, frame)
                else:
                    self.camera.capture_file(filepath)

            logger.info(f"✓ Photo saved: {filepath}")
            return filepath

        except Exception as e:
            logger.error(f"Error capturing photo: {e}")
            return None

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
