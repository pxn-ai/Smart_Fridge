#!/usr/bin/env python3
"""
Quick OCR test on existing captured images.
Runs the updated OCR engine on images in camera_project/images/ and Database/uploads/.
"""

import sys
import os
from pathlib import Path

# Set up paths
PROJECT_ROOT = Path(__file__).resolve().parent
CAMERA_PROJECT = PROJECT_ROOT / "camera_project"
sys.path.insert(0, str(CAMERA_PROJECT))
sys.path.insert(0, str(CAMERA_PROJECT / "src"))

import config.config as cam_config
from src.ocr_engine import OCREngine
from src.expiration_detector import ExpirationDetector

import logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

def test_image(ocr, detector, image_path):
    print(f"\n{'='*60}")
    print(f"  Testing: {os.path.basename(image_path)}")
    print(f"{'='*60}")
    
    result = ocr.extract_text(image_path)
    
    if not result:
        print("  ❌ OCR returned None")
        return False
    
    if result.get("status") != "success":
        print(f"  ❌ OCR failed: {result.get('error', 'unknown')}")
        return False
    
    text = result.get("full_text", "")
    meta = result.get("_meta", {})
    
    print(f"\n  Strategy: {meta.get('strategy', '?')}")
    print(f"  PSM mode: {meta.get('psm', '?')}")
    print(f"  Blur score: {meta.get('blur_score', -1):.1f}")
    print(f"  ROI source: {meta.get('roi', {}).get('source', '?')}")
    print(f"  Mean conf: {meta.get('mean_conf', 0):.1f}%")
    
    # Show extracted text
    clean_text = text.strip()
    if clean_text:
        print(f"\n  📝 Extracted text ({len(clean_text)} chars):")
        for line in clean_text.split('\n'):
            line = line.strip()
            if line:
                print(f"     {line}")
    else:
        print("\n  ❌ No text extracted")
        return False
    
    # Run expiration detection
    detection = detector.detect_expiration(text)
    dates = detection.get("detected_dates", [])
    primary = detection.get("primary_date")
    confidence = detection.get("confidence", 0)
    
    print(f"\n  📅 Dates found: {len(dates)}")
    for d in dates:
        print(f"     {d['raw']} → {d['formatted']}")
    
    if primary:
        print(f"  ✅ Primary expiry: {primary['formatted']} (confidence: {confidence:.0%})")
        return True
    else:
        print("  ⚠️  No expiry date detected from text")
        return False


def main():
    print("Initializing OCR engine...")
    ocr = OCREngine(cam_config)
    detector = ExpirationDetector(cam_config)
    
    # Test on the most recent images
    images_dir = CAMERA_PROJECT / "images"
    image_files = sorted(images_dir.glob("*.jpg"), key=os.path.getmtime, reverse=True)
    
    if not image_files:
        print("No images found in camera_project/images/")
        return
    
    # Test the 3 most recent images
    test_images = image_files[:3]
    
    successes = 0
    total = len(test_images)
    
    for img_path in test_images:
        if test_image(ocr, detector, str(img_path)):
            successes += 1
    
    print(f"\n{'='*60}")
    print(f"  Results: {successes}/{total} images had dates detected")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
