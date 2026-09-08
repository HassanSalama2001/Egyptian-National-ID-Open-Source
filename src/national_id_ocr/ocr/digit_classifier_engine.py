"""
Lightweight digit recognizer for numeric fields (national_id, serial_number,
birth_date), replacing EasyOCR for these fields entirely.

Rationale (see scripts/training/ for how this was chosen): Arabic-Indic
digits (unlike Arabic letters) don't connect/reshape based on neighbors, so
they can be segmented per-character via classical CV and classified
independently - a well-scoped 10-class problem for a single fixed printed
font. A HOG+SVM classifier trained on synthetic renders of the actual
on-card font beat a small CNN and RandomForest under an out-of-distribution
stress test (85.7% vs 80.8%/84.9% per-digit accuracy) while being smaller
(8.8MB) and needing no GPU/deep-learning framework at inference time.
Combined with Mod-11 checksum validation/repair, this is both far faster
and more accurate than running EasyOCR's general-purpose neural OCR on
digit fields.
"""
import logging
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import joblib
import numpy as np
from skimage.feature import hog

logger = logging.getLogger(__name__)

MODEL_PATH = Path(__file__).parent / "models" / "digit_classifier_svm.joblib"
CANVAS = 64  # must match scripts/training/generate_digit_crops.py

WESTERN_TO_ARABIC_INDIC = str.maketrans("0123456789", "٠١٢٣٤٥٦٧٨٩")
ARABIC_INDIC_TO_WESTERN = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")


class DigitClassifierEngine:
    """Segments a numeric-field crop into individual digits and classifies
    each with the trained HOG+SVM model. extract_digits() returns Western
    digit characters (0-9) - the pipeline's downstream checksum/parsing
    logic already expects Western digits."""

    def __init__(self, model_path: Path = MODEL_PATH):
        if not model_path.exists():
            raise FileNotFoundError(
                f"Digit classifier model not found at {model_path}. "
                f"Train it first: python scripts/training/train_digit_classifier.py"
            )
        self.clf = joblib.load(model_path)

    def _segment_digits(self, gray: np.ndarray) -> List[Tuple[int, np.ndarray]]:
        """Binarizes and finds individual digit blobs via contours,
        returning (x_position, digit_crop) pairs sorted left-to-right."""
        # Upscale small crops for more reliable contour detection
        h, w = gray.shape
        if h < 80:
            scale = 80 / h
            gray = cv2.resize(gray, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)

        denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
        _, thresh = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes = [cv2.boundingRect(c) for c in contours]
        if not boxes:
            return []

        # Field crop boxes carry generous padding (they're sized to fit
        # varying-length text), so glyph height is often a small, unknown
        # fraction of crop height - a fixed threshold like "crop_h * 0.35"
        # rejected every real digit crop while passing small noise specks
        # in initial testing. Digits in one field share the same font size,
        # so their heights cluster tightly - use the median height of
        # plausible candidates (an absolute floor of 12px removes obvious
        # noise/diacritic dots) as an adaptive reference instead.
        plausible = [h for (_, _, w, h) in boxes if h >= 12 and w >= 3]
        if not plausible:
            return []
        median_h = float(np.median(plausible))

        candidates = []
        for (x, y, w, h) in boxes:
            # '0' (a small diamond in this font) legitimately renders at
            # roughly half the height of other digits - a lower bound of
            # 0.5x median sits right at that boundary and single-pixel
            # rendering noise was dropping real '0's. 0.35x keeps real
            # noise (observed ~0.3x median or smaller) filtered while
            # giving '0' enough margin.
            if not (0.35 * median_h <= h <= 1.5 * median_h):
                continue
            if w < 3 or w > h * 1.8:
                continue
            candidates.append((x, thresh[y:y + h, x:x + w]))

        candidates.sort(key=lambda c: c[0])
        return candidates

    def _prepare_crop(self, digit_img: np.ndarray) -> np.ndarray:
        """Pads and resizes a segmented digit to match the classifier's
        training canvas (CANVASxCANVAS, white background, black glyph)."""
        h, w = digit_img.shape
        scale = (CANVAS * 0.7) / max(h, w)
        new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
        resized = cv2.resize(digit_img, (new_w, new_h), interpolation=cv2.INTER_AREA)

        canvas = np.zeros((CANVAS, CANVAS), dtype=np.uint8)
        x_off = (CANVAS - new_w) // 2
        y_off = (CANVAS - new_h) // 2
        canvas[y_off:y_off + new_h, x_off:x_off + new_w] = resized

        # Training data is white background / black glyph; segmented crops
        # here are white glyph / black background (THRESH_BINARY_INV) -
        # invert to match.
        return 255 - canvas

    def extract_digits(self, image: np.ndarray) -> str:
        """image: a BGR or grayscale crop containing a horizontal run of
        digits (e.g. the national_id or serial_number field crop)."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image

        segments = self._segment_digits(gray)
        if not segments:
            return ""

        digits = []
        for _, seg in segments:
            prepared = self._prepare_crop(seg)
            feat = hog(prepared, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2))
            pred = self.clf.predict(feat.reshape(1, -1))[0]
            digits.append(str(pred))

        return "".join(digits)
