"""
OCR Engine Module
Handles text extraction from images using Tesseract OCR
"""

import logging
import os

try:
    import pytesseract
    from PIL import Image, ImageEnhance, ImageFilter, ImageOps
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False
    print("Warning: pytesseract or PIL not available. OCR disabled.")

logger = logging.getLogger(__name__)


class OCREngine:
    """Extracts text from images using Tesseract OCR"""

    def __init__(self, config):
        self.config = config
        if OCR_AVAILABLE and hasattr(config, 'TESSERACT_PATH'):
            pytesseract.pytesseract.pytesseract_cmd = config.TESSERACT_PATH

    def extract_text(self, image_path):
        """
        Extract all text from image

        Args:
            image_path: Path to image file

        Returns:
            Dictionary with OCR results
        """
        if not OCR_AVAILABLE:
            logger.warning("OCR not available")
            return None

        try:
            if not os.path.exists(image_path):
                logger.error(f"Image not found: {image_path}")
                return None

            image = Image.open(image_path)

            def run_ocr(candidate_image, page_segmentation_mode):
                config = f"--oem 3 --psm {page_segmentation_mode}"
                text_result = pytesseract.image_to_string(
                    candidate_image,
                    lang=self.config.OCR_LANGUAGE,
                    config=config,
                )
                data_result = pytesseract.image_to_data(
                    candidate_image,
                    lang=self.config.OCR_LANGUAGE,
                    config=config,
                    output_type='dict',
                )
                return text_result, data_result

            def preprocess_for_ocr(source_image):
                working_image = source_image.convert("RGB")
                working_image = ImageOps.grayscale(working_image)
                working_image = ImageOps.autocontrast(working_image)
                working_image = working_image.resize(
                    (working_image.width * 2, working_image.height * 2)
                )
                working_image = ImageEnhance.Sharpness(working_image).enhance(2.0)
                working_image = ImageEnhance.Contrast(working_image).enhance(1.8)
                working_image = working_image.filter(ImageFilter.MedianFilter(size=3))
                return working_image

            processed_image = preprocess_for_ocr(image)
            text, data = run_ocr(processed_image, 6)

            if not text or not text.strip():
                text, data = run_ocr(image, 11)

            logger.debug(f"Extracted text from {os.path.basename(image_path)}")

            return {
                'full_text': text,
                'confidence': data.get('conf', []),
                'words': data.get('text', []),
                'status': 'success'
            }

        except Exception as e:
            logger.error(f"Error extracting text from {image_path}: {e}")
            return {'status': 'error', 'error': str(e)}

    def extract_regions(self, image_path, regions=None):
        """
        Extract text from specific image regions

        Args:
            image_path: Path to image
            regions: List of tuples (x, y, width, height) for regions

        Returns:
            Dictionary with extracted region text
        """
        if not OCR_AVAILABLE:
            return None

        try:
            image = Image.open(image_path)
            results = {}

            if regions:
                for i, (x, y, w, h) in enumerate(regions):
                    cropped = image.crop((x, y, x + w, y + h))
                    text = pytesseract.image_to_string(
                        cropped,
                        lang=self.config.OCR_LANGUAGE
                    )
                    results[f'region_{i}'] = text.strip()

            logger.debug(f"Extracted {len(results)} regions from image")
            return results

        except Exception as e:
            logger.error(f"Error extracting regions: {e}")
            return {}
