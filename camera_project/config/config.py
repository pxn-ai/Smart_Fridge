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
CAMERA_ROTATION = 0  # 0, 90, 180, 270
CAMERA_FLIP_HORIZONTAL = False
CAMERA_FLIP_VERTICAL = False

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
