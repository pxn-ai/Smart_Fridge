# Configuration settings for Raspberry Pi Camera Project

import os
from pathlib import Path

# Project directories
BASE_DIR = Path(__file__).resolve().parent.parent
IMAGES_DIR = os.path.join(BASE_DIR, 'images')
LOGS_DIR = os.path.join(BASE_DIR, 'logs')
DATA_DIR = os.path.join(BASE_DIR, 'data')
METADATA_DIR = os.path.join(DATA_DIR, 'metadata')

# Camera settings
CAMERA_RESOLUTION = (1920, 1080)  # (width, height)
CAMERA_FRAMERATE = 30
CAMERA_ROTATION = 180  # 0, 90, 180, 270
CAMERA_FLIP_HORIZONTAL = False
CAMERA_FLIP_VERTICAL = False
CAMERA_AE_ENABLE = False
CAMERA_AE_METERING_MODE = 2  # MeteringMatrix
CAMERA_AE_EXPOSURE_MODE = 0   # ExposureNormal
CAMERA_AWB_ENABLE = True
CAMERA_EXPOSURE_TIME = 30000  # microseconds
CAMERA_ANALOGUE_GAIN = 4.0
CAMERA_FRAME_DURATION_MIN = 30000
CAMERA_FRAME_DURATION_MAX = 33333

# OCR guide overlay / crop settings
OCR_GUIDE_BOX_WIDTH_RATIO = 0.72
OCR_GUIDE_BOX_HEIGHT_RATIO = 0.36
OCR_GUIDE_BOX_CENTER_X = 0.50
OCR_GUIDE_BOX_CENTER_Y = 0.50

# Image capture settings
CAPTURE_INTERVAL = 5  # seconds between captures
ENABLE_PREVIEW = False  # Set to True to show camera preview
IMAGE_FORMAT = 'jpeg'  # 'jpeg' or 'rgb'
JPEG_QUALITY = 85

# Image scanning settings
SCAN_INTERVAL = 10  # seconds between scans
MAX_IMAGES_TO_KEEP = 100  # Maximum images stored before cleanup
IMAGE_THRESHOLD = 50  # For motion/change detection (0-255)

# OCR settings
OCR_LANGUAGE = 'eng'  # Tesseract language code
TESSERACT_PATH = '/usr/bin/tesseract'  # Path to tesseract binary
YOLO_DATE_MODEL_PATH = os.path.join(BASE_DIR, 'models', 'yolov26n.pt')
YOLO_DATE_DETECT_CONFIDENCE = 0.30
YOLO_DATE_DETECT_PADDING = 0.08

# GPIO / Button settings (set USE_GPIO=False for keyboard mode)
USE_GPIO = False
BUTTON_GPIO = 17  # BCM pin number
BUTTON_DEBOUNCE_TIME = 0.3  # seconds

# Data storage settings
EXPIRATION_LOG_FILE = os.path.join(DATA_DIR, 'expiration_log.json')

# Logging settings
LOG_LEVEL = 'INFO'  # 'DEBUG', 'INFO', 'WARNING', 'ERROR'
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

# Create directories if they don't exist
os.makedirs(IMAGES_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(METADATA_DIR, exist_ok=True)
