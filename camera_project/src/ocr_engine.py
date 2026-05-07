"""
OCR Engine Module
Handles text extraction from images using Tesseract OCR.

Improvements over v1:
- Blur detection via Laplacian variance (skips/warns on shaky frames)
- Three independent preprocessing strategies run in parallel;
  the one with the highest Tesseract confidence wins
- Adaptive local-mean thresholding for uneven lighting / shadows
- Proper unsharp-mask kernel instead of amplifying motion smear
- Expanded PSM mode sweep (6, 4, 7, 11) per strategy
"""

import logging
import os
import statistics

try:
    import pytesseract
    from PIL import Image, ImageEnhance, ImageFilter, ImageOps, ImageChops
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False
    print("Warning: pytesseract or PIL not available. OCR disabled.")

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

logger = logging.getLogger(__name__)

# ── Tuning knobs ──────────────────────────────────────────────────────────────
BLUR_THRESHOLD   = 80.0   # Laplacian variance below this → warn & try harder
UPSCALE_FACTOR   = 2      # Applied in strategies A and B
UPSCALE_FACTOR_C = 3      # Strategy C uses a harder upscale for faded labels
PSM_MODES        = [6, 4, 7, 11]  # Page-segmentation modes to try per strategy


class OCREngine:
    """Extracts text from images using Tesseract OCR."""

    def __init__(self, config):
        self.config = config
        if OCR_AVAILABLE and hasattr(config, "TESSERACT_PATH"):
            pytesseract.pytesseract.pytesseract_cmd = config.TESSERACT_PATH

    # ── Public API ────────────────────────────────────────────────────────────

    def extract_text(self, image_path):
        """
        Extract all text from image.

        Args:
            image_path: Path to image file.

        Returns:
            dict with keys: full_text, confidence, words, status
        """
        if not OCR_AVAILABLE:
            logger.warning("OCR not available")
            return None

        if not os.path.exists(image_path):
            logger.error("Image not found: %s", image_path)
            return None

        try:
            image = Image.open(image_path)
        except Exception as exc:
            logger.error("Cannot open image %s: %s", image_path, exc)
            return {"status": "error", "error": str(exc)}

        try:
            gray = image.convert("L")
            blur_score = self._blur_score(gray)
            is_blurry = blur_score < BLUR_THRESHOLD
            if is_blurry:
                logger.warning(
                    "Image looks blurry (Laplacian var=%.1f < %.1f) — "
                    "trying all deblur strategies",
                    blur_score, BLUR_THRESHOLD,
                )
            else:
                logger.debug("Blur score OK: %.1f", blur_score)

            candidates = self._build_candidates(image, gray, is_blurry)
            best = self._best_result(candidates)

            if best is None:
                logger.warning("All OCR strategies returned empty text")
                return {"status": "error", "error": "No text detected"}

            logger.info(
                "OCR done — strategy=%s psm=%s conf=%.0f%% chars=%d",
                best["strategy"], best["psm"],
                best["mean_conf"], len(best["text"]),
            )
            return {
                "full_text": best["text"],
                "confidence": best["raw_conf"],
                "words": best["words"],
                "status": "success",
                "_meta": {
                    "strategy": best["strategy"],
                    "psm": best["psm"],
                    "mean_conf": best["mean_conf"],
                    "blur_score": blur_score,
                },
            }

        except Exception as exc:
            logger.error("Error extracting text from %s: %s", image_path, exc)
            return {"status": "error", "error": str(exc)}

    def extract_regions(self, image_path, regions=None):
        """
        Extract text from specific image regions.

        Args:
            image_path: Path to image.
            regions: List of (x, y, width, height) tuples.

        Returns:
            dict mapping region_N to extracted text.
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
                        cropped, lang=self.config.OCR_LANGUAGE
                    )
                    results[f"region_{i}"] = text.strip()
            logger.debug("Extracted %d regions", len(results))
            return results
        except Exception as exc:
            logger.error("Error extracting regions: %s", exc)
            return {}

    # ── Blur detection ────────────────────────────────────────────────────────

    def _blur_score(self, gray_image):
        """
        Estimate sharpness via Laplacian variance.
        Higher = sharper.  Works without OpenCV.
        """
        if NUMPY_AVAILABLE:
            arr = np.array(gray_image, dtype=np.float32)
            # Discrete 3×3 Laplacian
            kernel = np.array([[0, 1, 0],
                                [1, -4, 1],
                                [0, 1, 0]], dtype=np.float32)
            from numpy.lib.stride_tricks import as_strided
            # Pad and convolve manually (scipy not guaranteed)
            padded = np.pad(arr, 1, mode="reflect")
            h, w = arr.shape
            lap = (
                padded[0:h,   1:w+1] +
                padded[2:h+2, 1:w+1] +
                padded[1:h+1, 0:w]   +
                padded[1:h+1, 2:w+2] -
                4 * padded[1:h+1, 1:w+1]
            )
            return float(lap.var())
        else:
            # PIL-only fallback: compare image to heavily blurred version
            blurred = gray_image.filter(ImageFilter.GaussianBlur(radius=3))
            diff = ImageChops.difference(gray_image, blurred)
            histogram = diff.histogram()
            total = sum(histogram)
            if total == 0:
                return 0.0
            mean = sum(i * v for i, v in enumerate(histogram)) / total
            variance = sum((i - mean) ** 2 * v for i, v in enumerate(histogram)) / total
            return float(variance)

    # ── Preprocessing strategies ──────────────────────────────────────────────

    def _build_candidates(self, image, gray, is_blurry):
        """
        Build all (preprocessed_image, strategy_name) pairs to try.
        Extra deblur strategies are added when the image is blurry.
        """
        candidates = [
            ("A_contrast",  self._strategy_contrast(gray)),
            ("B_adaptive",  self._strategy_adaptive(gray)),
            ("C_binarize",  self._strategy_binarize(gray)),
        ]
        if is_blurry:
            candidates.append(("D_deblur", self._strategy_deblur(gray)))
        return candidates

    def _strategy_contrast(self, gray):
        """
        Strategy A — improved version of the original pipeline.
        autocontrast → upscale → unsharp mask (NOT raw sharpen) → mild contrast.
        """
        img = ImageOps.autocontrast(gray, cutoff=1)
        img = img.resize(
            (img.width * UPSCALE_FACTOR, img.height * UPSCALE_FACTOR),
            Image.LANCZOS,
        )
        # Unsharp mask: (radius, percent, threshold)
        # Much safer than Sharpness.enhance on blurry input.
        img = img.filter(ImageFilter.UnsharpMask(radius=1, percent=120, threshold=2))
        img = ImageEnhance.Contrast(img).enhance(1.5)
        img = img.filter(ImageFilter.MedianFilter(size=3))
        return img

    def _strategy_adaptive(self, gray):
        """
        Strategy B — adaptive local-mean thresholding.
        Handles uneven lighting, shadows, and curved labels well.
        """
        img = gray.resize(
            (gray.width * UPSCALE_FACTOR, gray.height * UPSCALE_FACTOR),
            Image.LANCZOS,
        )
        if NUMPY_AVAILABLE:
            arr = np.array(img, dtype=np.float32)
            # Local mean via a box blur (block size ~31px after upscale)
            pil_blur = img.filter(ImageFilter.BoxBlur(15))
            local_mean = np.array(pil_blur, dtype=np.float32)
            # Threshold: pixel > local_mean - C  →  white, else black
            C = 8
            binary = ((arr > local_mean - C) * 255).astype(np.uint8)
            img = Image.fromarray(binary, mode="L")
        else:
            # Fallback: autocontrast + hard binarize
            img = ImageOps.autocontrast(img, cutoff=2)
            img = img.point(lambda p: 255 if p > 127 else 0)
        return img

    def _strategy_binarize(self, gray):
        """
        Strategy C — Otsu-like global binarization with aggressive upscale.
        Best for printed labels with consistent lighting.
        """
        img = ImageOps.autocontrast(gray, cutoff=2)
        img = img.resize(
            (gray.width * UPSCALE_FACTOR_C, gray.height * UPSCALE_FACTOR_C),
            Image.LANCZOS,
        )
        img = ImageEnhance.Contrast(img).enhance(2.0)
        # Otsu threshold approximation via histogram
        img = self._otsu_binarize(img)
        # Light denoise after binarization
        img = img.filter(ImageFilter.MedianFilter(size=3))
        return img

    def _strategy_deblur(self, gray):
        """
        Strategy D — specifically for motion blur / camera shake.
        Applies a strong high-frequency boost before upscaling.
        """
        # Gentle Gaussian blur to estimate the low-freq component
        low_freq = gray.filter(ImageFilter.GaussianBlur(radius=1))
        # High-boost: original + k*(original - low_freq)
        if NUMPY_AVAILABLE:
            orig = np.array(gray, dtype=np.float32)
            lf   = np.array(low_freq, dtype=np.float32)
            k = 2.5
            boosted = np.clip(orig + k * (orig - lf), 0, 255).astype(np.uint8)
            img = Image.fromarray(boosted, mode="L")
        else:
            img = ImageEnhance.Sharpness(gray).enhance(3.0)

        img = ImageOps.autocontrast(img, cutoff=1)
        img = img.resize(
            (gray.width * UPSCALE_FACTOR, gray.height * UPSCALE_FACTOR),
            Image.LANCZOS,
        )
        img = img.filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3))
        img = ImageEnhance.Contrast(img).enhance(1.6)
        return img

    def _otsu_binarize(self, gray_img):
        """Approximate Otsu threshold using the image histogram."""
        histogram = gray_img.histogram()
        total = sum(histogram)
        if total == 0:
            return gray_img
        sum_all = sum(i * v for i, v in enumerate(histogram))
        sum_bg, weight_bg, best_thresh, best_var = 0.0, 0, 0, 0.0
        for t in range(256):
            weight_bg += histogram[t]
            if weight_bg == 0:
                continue
            weight_fg = total - weight_bg
            if weight_fg == 0:
                break
            sum_bg += t * histogram[t]
            mean_bg = sum_bg / weight_bg
            mean_fg = (sum_all - sum_bg) / weight_fg
            var_between = weight_bg * weight_fg * (mean_bg - mean_fg) ** 2
            if var_between > best_var:
                best_var = var_between
                best_thresh = t
        return gray_img.point(lambda p: 255 if p >= best_thresh else 0)

    # ── OCR runner ────────────────────────────────────────────────────────────

    def _run_tesseract(self, img, psm):
        """Run Tesseract on a preprocessed image with the given PSM mode."""
        cfg = f"--oem 3 --psm {psm}"
        lang = self.config.OCR_LANGUAGE

        text = pytesseract.image_to_string(img, lang=lang, config=cfg)
        data = pytesseract.image_to_data(
            img, lang=lang, config=cfg, output_type="dict"
        )
        confs = [c for c in data.get("conf", []) if isinstance(c, (int, float)) and c >= 0]
        mean_conf = statistics.mean(confs) if confs else 0.0
        words = [w for w in data.get("text", []) if w and w.strip()]

        return {
            "text": text,
            "raw_conf": data.get("conf", []),
            "words": words,
            "mean_conf": mean_conf,
            "psm": psm,
        }

    def _best_result(self, candidates):
        """
        Run all strategy × PSM combinations.
        Return the result with the highest mean Tesseract confidence
        that also contains non-empty text.
        """
        best = None
        for strategy_name, preprocessed in candidates:
            for psm in PSM_MODES:
                try:
                    result = self._run_tesseract(preprocessed, psm)
                    result["strategy"] = strategy_name
                    if not result["text"].strip():
                        continue
                    if best is None or result["mean_conf"] > best["mean_conf"]:
                        best = result
                        logger.debug(
                            "New best: strategy=%s psm=%d conf=%.1f",
                            strategy_name, psm, result["mean_conf"],
                        )
                except Exception as exc:
                    logger.debug(
                        "strategy=%s psm=%d failed: %s", strategy_name, psm, exc
                    )
        return best
