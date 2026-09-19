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

Training font is Tahoma Bold. A persistent, hard-to-fix real-card '٣' ->
'٢' misread led to trying Simplified Arabic Bold instead, after visually
confirming its calligraphic glyph shapes are a closer match to a real
card than Tahoma's geometric ones - it did fix that specific confusion,
but real-card retesting then showed a NET WORSE result: 3 simultaneous
wrong digits instead of 1 (too many for the checksum-repair's ambiguity-
safe logic to fix at all). That whole family of calligraphic Arabic
fonts (also checked Sakkal Majalla and Traditional Arabic - same
underlying digit shapes) has '٢'/'٧' as similar angular hooks and '٥'/
'٠' as similar round blobs - thinner, more delicate distinguishing
strokes than Tahoma's, and more fragile under real-world blur/
compression. Reverted to Tahoma as the net-better, known-quantity
baseline; properly fixing the remaining '٣'/'٢' confusion needs a
bigger, more carefully-curated training effort than a same-session font
swap (see memory/conversation history for the full investigation).
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

    # Binarization methods tried by extract_digits, cheapest/most
    # conservative first. No single one wins across real-world capture
    # conditions - confirmed by scanning the same physical card four ways
    # (normal, "enhanced", greyscale and "eco"): the hard black-and-white
    # "eco" scan was the ONLY one whose national_id segmented into the
    # correct 14 digits, while the colour and "enhanced" versions came
    # back with 11 and 12. The classifier wants hard binary contrast, so
    # rather than picking one preprocessing path and hoping, try several
    # and let the known field length decide which one actually worked.
    BINARIZATION_METHODS = ("otsu", "clahe_otsu", "sauvola", "adaptive", "raw_binary")

    # Upscale factors tried alongside the methods above. Upscaling before
    # thresholding gives thin strokes more pixels to survive denoising and
    # can separate glyphs that merge into one blob at native resolution.
    UPSCALE_FACTORS = (None, 2, 3)

    def _binarize(self, gray: np.ndarray, denoised: np.ndarray, method: str) -> np.ndarray:
        """Produces a white-glyph-on-black binary image by `method` - see
        BINARIZATION_METHODS. `denoised` is passed in already computed
        because fastNlMeansDenoising is by far the most expensive step
        here and is identical across methods at a given scale."""
        if method == "clahe_otsu":
            # Low-contrast/embossed content (e.g. front's birth_date
            # security-stamp watermark, much fainter than solid-ink
            # printed fields like national_id) can wash out under plain
            # Otsu thresholding - boost local contrast first so faint
            # strokes separate from the background instead of blending
            # into noise or nothing surviving the threshold at all. Not
            # the default: it amplifies faint background texture into
            # spurious digit-shaped blobs on fields that already
            # threshold cleanly.
            boosted = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(gray)
            boosted = cv2.fastNlMeansDenoising(boosted, None, 10, 7, 21)
            _, thresh = cv2.threshold(boosted, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            return thresh

        if method == "sauvola":
            # Local thresholding designed for document images: computes a
            # cutoff per neighbourhood from local mean and standard
            # deviation, so it survives a gradient across the crop (one
            # side shadowed, glare on the other) that a single global
            # Otsu cutoff cannot. This is effectively what a scanner
            # app's "eco"/black-and-white filter does - the mode that
            # empirically produced the only correct 14-digit read on a
            # real card.
            from skimage.filters import threshold_sauvola
            local_thresh = threshold_sauvola(denoised, window_size=25, k=0.2)
            return ((denoised < local_thresh) * 255).astype(np.uint8)

        # A dilating "thickened" variant was tried here to reconnect glyphs
        # that a harsh black-and-white filter erodes into two blobs. A/B'd
        # over 40 seeded cards it changed nothing on any condition, so it
        # was removed rather than kept as cost without benefit.

        if method == "raw_binary":
            # For a scan that is ALREADY hard black-and-white (a scanner
            # app's "eco" filter), the usual denoise-then-Otsu path is
            # actively harmful: denoising a binary image introduces grey
            # halos that the second threshold then re-cuts in the wrong
            # place, eroding or thickening strokes and turning a '٥' into
            # a '٠'. Thresholding the raw pixels at a fixed midpoint
            # instead preserves the glyph shapes exactly as scanned.
            #
            # Safe to add for every input, not just binary ones: a crude
            # fixed cutoff on a normal greyscale photo rarely produces the
            # field's expected glyph count, so the selection logic in
            # extract_digits_detailed simply discards it. It can only win
            # where it is genuinely better.
            _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
            return thresh

        if method == "adaptive":
            # A single global cutoff assumes roughly uniform contrast
            # across the crop - a real card's security-hologram print can
            # fail that badly enough that nothing usable survives. The
            # morphological close bridges a glyph split into fragments
            # where its own stroke crosses a local contrast dip, without
            # reaching far enough to merge genuinely separate digits.
            thresh = cv2.adaptiveThreshold(
                denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 8
            )
            return cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, np.ones((5, 9), np.uint8))

        _, thresh = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        return thresh

    def _rescale(self, gray: np.ndarray, upscale: Optional[float]) -> np.ndarray:
        h, w = gray.shape
        if upscale:
            return cv2.resize(
                gray, (int(w * upscale), int(h * upscale)), interpolation=cv2.INTER_CUBIC
            )
        if h < 80:
            # Upscale small crops for more reliable contour detection
            scale = 80 / h
            return cv2.resize(gray, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)
        return gray

    def _segment_digits(
        self,
        gray: np.ndarray,
        method: str = "otsu",
        upscale: Optional[float] = None,
    ) -> List[Tuple[int, np.ndarray]]:
        """Binarizes and finds individual digit blobs via contours,
        returning (x_position, digit_crop) pairs sorted left-to-right."""
        scaled = self._rescale(gray, upscale)
        denoised = cv2.fastNlMeansDenoising(scaled, None, 10, 7, 21)
        return self._find_digit_boxes(self._binarize(scaled, denoised, method))

    def _find_digit_boxes(
        self, thresh: np.ndarray, target_count: Optional[int] = None
    ) -> List[Tuple[int, np.ndarray]]:
        """Filters contours in a binarized crop down to plausible digit
        blobs, returning (x_position, digit_crop) sorted left-to-right.

        target_count, when known, lets an over-wide blob be split rather
        than discarded - see _split_merged_box."""
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

        accepted: List[Tuple[int, int, int, int]] = []
        too_wide: List[Tuple[int, int, int, int]] = []
        for (x, y, w, h) in boxes:
            # '0' (a small diamond in this font) legitimately renders at
            # roughly half the height of other digits - a lower bound of
            # 0.5x median sits right at that boundary and single-pixel
            # rendering noise was dropping real '0's. 0.35x keeps real
            # noise (observed ~0.3x median or smaller) filtered while
            # giving '0' enough margin.
            if not (0.35 * median_h <= h <= 1.5 * median_h):
                continue
            if w < 3:
                continue
            # (Tried also rejecting w >= h here, on the theory that real
            # digits/separators are always taller than wide - reverted:
            # this font's '0' can render as a near-square diamond
            # (confirmed on synthetic data: a legitimate '0' at w=8,h=8),
            # too close to the w=14,h=13 stray speck that motivated the
            # idea to separate safely by aspect ratio alone.)
            if w > h * 1.8:
                # Card-height but far too wide for one glyph: almost
                # always neighbouring digits fused by a harsh threshold.
                # These used to be DISCARDED, losing two or more real
                # digits at once - which is why an eco-filtered back
                # national_id found the right digit count only 41.7% of
                # the time. Keep them aside; if the count comes up short,
                # they are the obvious place to recover digits from.
                too_wide.append((x, y, w, h))
                continue
            accepted.append((x, y, w, h))

        if target_count is not None and len(accepted) < target_count and too_wide:
            typical_w = float(np.median([w for (_, _, w, _) in accepted])) if accepted else median_h * 0.6
            for box in too_wide:
                accepted.extend(self._split_merged_box(thresh, box, typical_w))

        accepted.sort(key=lambda b: b[0])
        return [(x, thresh[y:y + h, x:x + w]) for (x, y, w, h) in accepted]

    @staticmethod
    def _split_merged_box(
        thresh: np.ndarray,
        box: Tuple[int, int, int, int],
        typical_width: float,
    ) -> List[Tuple[int, int, int, int]]:
        """Cuts a fused multi-digit blob back into single digits.

        How many digits it holds is estimated from the blob's width
        against the width of the glyphs that did segment cleanly. The cuts
        are then placed at the narrowest points of the blob's vertical ink
        profile near each evenly-spaced boundary - where two digits touch
        there is still a thin waist, even when a threshold has bridged it.
        """
        x, y, w, h = box
        if typical_width <= 0:
            return []
        n_digits = int(round(w / typical_width))
        if n_digits < 2:
            return []

        column_ink = (thresh[y:y + h, x:x + w] > 0).sum(axis=0)
        cuts = [0]
        search_radius = max(1, int(typical_width * 0.25))
        for i in range(1, n_digits):
            ideal = int(round(i * w / n_digits))
            low = max(cuts[-1] + 1, ideal - search_radius)
            high = min(w - 1, ideal + search_radius)
            if low >= high:
                cuts.append(ideal)
                continue
            # Narrowest column in the window = where the glyphs touch.
            cuts.append(low + int(np.argmin(column_ink[low:high])))
        cuts.append(w)

        parts = []
        for start, end in zip(cuts, cuts[1:]):
            if end - start >= 3:
                parts.append((x + start, y, end - start, h))
        return parts

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

    def extract_digits(
        self,
        image: np.ndarray,
        expected_length: Optional[int] = None,
        separator_positions: Optional[List[int]] = None,
    ) -> str:
        """image: a BGR or grayscale crop containing a horizontal run of
        digits (e.g. the national_id or serial_number field crop).

        expected_length: for fields with a fixed digit count (e.g. a 14-
        digit national_id, or "YYYY/MM"'s 6 digits), pass it so a "/"
        separator segmented alongside the digits can be dropped rather
        than forced into some digit prediction (this classifier has no
        "not a digit" class).

        separator_positions: for fields with a known, fixed date format
        (e.g. "YYYY/MM" or "YYYY/MM/DD"), the 0-indexed position(s) each
        "/" falls at within the raw digits+separators sequence (e.g. [4]
        for YYYY/MM: Y,Y,Y,Y,/,M,M; [4, 7] for YYYY/MM/DD). When the
        segmented count exactly matches expected_length + len(separator_
        positions), those exact positions are dropped directly instead of
        guessed at. This replaces a fill-ratio guess (lowest relative
        ink-coverage in the crop) that assumed a "/" always fills its
        bounding box less than a real digit does - confirmed false on a
        real card: a "/" segmented as a tall, narrow blob scored a HIGHER
        fill ratio than a genuinely small "0", so the guess dropped the
        wrong (real) digit instead. Falls back to the fill-ratio guess
        when the count doesn't match exactly (extra noise, a merged
        glyph, etc. - anything not accounted for by the known format)."""
        return self.extract_digits_detailed(image, expected_length, separator_positions)[0]

    def _drop_separators(
        self,
        segments: List[Tuple[int, np.ndarray]],
        expected_length: Optional[int],
        separator_positions: Optional[List[int]],
    ) -> List[Tuple[int, np.ndarray]]:
        if (
            separator_positions
            and expected_length is not None
            and len(segments) == expected_length + len(separator_positions)
        ):
            drop_idx = set(separator_positions)
            return [s for i, s in enumerate(segments) if i not in drop_idx]
        if expected_length is not None and len(segments) > expected_length:
            n_drop = len(segments) - expected_length
            fill_ratios = [(seg > 0).sum() / seg.size for _, seg in segments]
            drop_idx = set(sorted(range(len(segments)), key=lambda i: fill_ratios[i])[:n_drop])
            return [s for i, s in enumerate(segments) if i not in drop_idx]
        return segments

    def _classify_segments(self, segments: List[Tuple[int, np.ndarray]]) -> str:
        digits = []
        for _, seg in segments:
            prepared = self._prepare_crop(seg)
            feat = hog(prepared, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2))
            digits.append(str(self.clf.predict(feat.reshape(1, -1))[0]))
        return "".join(digits)

    # Confidence returned alongside the digits (see extract_digits_detailed).
    # These are not calibrated probabilities - they encode how much
    # independent support the reading has, which is what downstream
    # confidence scoring actually needs to stop overstating a bad read.
    CONFIDENCE_METHODS_AGREE = 0.95
    CONFIDENCE_SINGLE_METHOD = 0.6
    CONFIDENCE_WRONG_LENGTH = 0.25

    def extract_digits_detailed(
        self,
        image: np.ndarray,
        expected_length: Optional[int] = None,
        separator_positions: Optional[List[int]] = None,
    ) -> Tuple[str, float]:
        """
        Reads the digits and reports how much independent support that
        reading has, as (digits, confidence).

        The old code reported a flat 0.9 for any non-empty result, which
        made downstream confidence meaningless: a garbled read scored the
        same as a clean one. Two signals are available here for free and
        both were measured to matter far more than the classifier's own
        decision-function margin (which barely separates correct from
        incorrect predictions - means of 1.14 vs 0.98, overlapping enough
        that the best threshold rejects only ~28% of errors):

        1. Did any configuration segment the crop into exactly the number
           of glyphs this field is known to have? When one does, the
           resulting read was correct ~98% of the time in testing; the
           failures are overwhelmingly the cases where none does.
        2. Do two INDEPENDENT binarization methods produce the identical
           digit string? Agreement between methods that fail in different
           ways is strong evidence, and it doubles as a free accuracy
           improvement: on disagreement the most-supported reading wins
           rather than whichever method happened to run first.

        Stops as soon as two configurations agree, so the common case
        costs barely more than before.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image

        target_count = None
        if expected_length is not None:
            target_count = expected_length + len(separator_positions or [])

        reads: List[str] = []
        closest: List[Tuple[int, np.ndarray]] = []

        for upscale in self.UPSCALE_FACTORS:
            # Denoising dominates the cost here and is identical for every
            # binarization method at a given scale, so it's computed once
            # per scale rather than once per (scale, method) pair.
            scaled = self._rescale(gray, upscale)
            denoised = cv2.fastNlMeansDenoising(scaled, None, 10, 7, 21)
            for method in self.BINARIZATION_METHODS:
                try:
                    segments = self._find_digit_boxes(
                        self._binarize(scaled, denoised, method), target_count
                    )
                except Exception as e:
                    logger.debug(f"segmentation failed (method={method}, upscale={upscale}): {e}")
                    continue
                if not segments:
                    continue

                if target_count is None:
                    # No known length to aim for - nothing to cross-check
                    # against, so the first method that finds anything is
                    # all we can justify choosing.
                    return self._classify_segments(segments), self.CONFIDENCE_SINGLE_METHOD

                if len(segments) == target_count:
                    digits = self._classify_segments(
                        self._drop_separators(segments, expected_length, separator_positions)
                    )
                    reads.append(digits)
                    if reads.count(digits) >= 2:
                        return digits, self.CONFIDENCE_METHODS_AGREE
                elif not closest or abs(len(segments) - target_count) < abs(len(closest) - target_count):
                    closest = segments

        if reads:
            # No two methods agreed outright; go with the most supported
            # reading and say so in the confidence.
            from collections import Counter
            digits, support = Counter(reads).most_common(1)[0]
            if support >= 2:
                return digits, self.CONFIDENCE_METHODS_AGREE
            return digits, self.CONFIDENCE_SINGLE_METHOD

        if not closest:
            return "", 0.0

        # Nothing produced the expected glyph count - the reading below is
        # a best effort off the closest segmentation and should be treated
        # as unreliable.
        digits = self._classify_segments(
            self._drop_separators(closest, expected_length, separator_positions)
        )
        return digits, self.CONFIDENCE_WRONG_LENGTH
