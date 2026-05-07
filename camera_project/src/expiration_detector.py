"""
Expiration Date Detector Module
Detects and parses expiration dates from extracted text
"""

import argparse
import json
import logging
import re
import sys
from datetime import datetime
from pathlib import Path
from dateutil import parser as dateparser

logger = logging.getLogger(__name__)


class ExpirationDetector:
    """Detects expiration dates in text"""

    EXPIRY_KEYWORDS = [
        r'best\s+by',
        r'use\s+by',
        r'best\s+before',
        r'expir(y|ation|es)?',
        r'exp[\s\-]?',
        r'due[\s\-]?',
        r'consume\s+by',
        r'sell[\s\-]?by',
    ]

    DATE_PATTERNS = [
        r'\d{1,2}/\d{1,2}/\d{2,4}',
        r'\d{1,2}-\d{1,2}-\d{2,4}',
        r'\d{4}-\d{1,2}-\d{1,2}',
        r'(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s*\d{2,4}',
        r'\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{2,4}',
    ]

    def __init__(self, config=None):
        self.config = config

    def find_expiry_lines(self, text):
        """
        Find lines containing expiration indicators

        Args:
            text: Extracted text from OCR

        Returns:
            List of relevant text lines
        """
        try:
            lines = text.split('\n')
            relevant_lines = []

            keyword_pattern = '|'.join(self.EXPIRY_KEYWORDS)

            for line in lines:
                if re.search(keyword_pattern, line, re.IGNORECASE):
                    relevant_lines.append(line.strip())

            logger.debug(f"Found {len(relevant_lines)} expiry lines")
            return relevant_lines

        except Exception as e:
            logger.error(f"Error finding expiry lines: {e}")
            return []

    def extract_dates(self, text):
        """
        Extract all potential dates from text

        Args:
            text: Text to search

        Returns:
            List of date strings found
        """
        dates = []

        for pattern in self.DATE_PATTERNS:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                dates.append(match.group(0))

        return list(set(dates))  # Remove duplicates

    def parse_date(self, date_string):
        """
        Parse date string to standardized format

        Args:
            date_string: Date string to parse

        Returns:
            Datetime object or None
        """
        try:
            parsed_date = dateparser.parse(date_string, dayfirst=False)
            return parsed_date

        except Exception as e:
            logger.debug(f"Could not parse date '{date_string}': {e}")
            return None

    def detect_expiration(self, text):
        """
        Main method to detect expiration date from text

        Args:
            text: Extracted OCR text

        Returns:
            Dictionary with detection results
        """
        try:
            results = {
                'raw_text': text,
                'expiry_lines': [],
                'detected_dates': [],
                'primary_date': None,
                'confidence': 0
            }

            # Find lines with expiry keywords
            expiry_lines = self.find_expiry_lines(text)
            results['expiry_lines'] = expiry_lines

            # Extract all dates
            dates = self.extract_dates(text)

            # Parse and validate dates
            parsed_dates = []
            for date_str in dates:
                parsed = self.parse_date(date_str)
                if parsed and parsed > datetime.now():
                    parsed_dates.append({
                        'raw': date_str,
                        'parsed': parsed.isoformat(),
                        'formatted': parsed.strftime('%Y-%m-%d')
                    })

            results['detected_dates'] = parsed_dates

            # Set primary date (earliest future date or first found)
            if parsed_dates:
                results['primary_date'] = min(
                    parsed_dates,
                    key=lambda x: x['parsed']
                )
                results['confidence'] = 0.9 if expiry_lines else 0.7

            logger.info(f"Detected expiration dates: {parsed_dates}")
            return results

        except Exception as e:
            logger.error(f"Error in expiration detection: {e}")
            return {'status': 'error', 'error': str(e)}


def _format_detection_result(result):
    """Format detection results for CLI output."""
    output = {
        "expiry_lines": result.get("expiry_lines", []),
        "detected_dates": result.get("detected_dates", []),
        "primary_date": result.get("primary_date"),
        "confidence": result.get("confidence", 0),
    }
    return json.dumps(output, indent=2, ensure_ascii=False)


def _main():
    """Run a quick OCR and expiration detection test from the command line."""
    project_root = Path(__file__).resolve().parent.parent
    src_dir = project_root / "src"

    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))

    parser = argparse.ArgumentParser(
        description="Test OCR text extraction and expiration detection on an image."
    )
    parser.add_argument(
        "image",
        help="Path to the image file you want to test.",
    )
    parser.add_argument(
        "--text",
        help="Optional raw text input. If provided, OCR is skipped and this text is analyzed directly.",
    )
    args = parser.parse_args()

    detector = ExpirationDetector()

    if args.text:
        print("=== INPUT TEXT ===")
        print(args.text)
        print()
        result = detector.detect_expiration(args.text)
        print("=== EXPIRATION RESULT ===")
        print(_format_detection_result(result))
        return 0

    try:
        import config.config as cam_config
        from camera_project.backup.ocr_engine import OCREngine
    except Exception as exc:
        print(f"Failed to import OCR dependencies: {exc}")
        return 1

    ocr = OCREngine(cam_config)
    ocr_result = ocr.extract_text(args.image)

    if not ocr_result or ocr_result.get("status") != "success":
        print("OCR failed.")
        print(json.dumps(ocr_result or {"status": "error"}, indent=2, ensure_ascii=False))
        return 1

    text = ocr_result.get("full_text", "")

    print("=== OCR TEXT ===")
    print(text.strip() if text.strip() else "[No text detected]")
    print()

    result = detector.detect_expiration(text)
    print("=== EXPIRATION RESULT ===")
    print(_format_detection_result(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
