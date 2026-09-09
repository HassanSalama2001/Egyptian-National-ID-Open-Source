import numpy as np
import logging
import time
import cv2
import base64
from typing import Optional, List, Dict
from concurrent.futures import ThreadPoolExecutor

from ..detection.card_detector import CardDetector
from ..detection.side_classifier import SideClassifier, IDSide
from ..ocr.base import OCREngine, OCRPreprocessor
from ..ocr.digit_classifier_engine import DigitClassifierEngine
from ..postprocessing.numeral_converter import normalize_arabic_numerals, clean_ocr_text
from ..postprocessing.national_id_parser import decode_national_id, find_nid_in_text
from ..models.id_card import IDCard, IDCardFront, IDCardBack
from ..models.enums import ExtractionStatus
from .layout_analyzer import LayoutAnalyzer
from .field_clusterer import FieldClusterer

logger = logging.getLogger(__name__)

class Pipeline:
    def __init__(self, ocr_engine: Optional[OCREngine] = None, tesseract_cmd: str = "tesseract"):
        self.detector = CardDetector()
        self.classifier = SideClassifier()
        if ocr_engine is None:
            # Both cli.py and api/server.py constructed Pipeline() without
            # ever passing an ocr_engine, leaving this None - the first
            # non-numeric field would crash with AttributeError. Default to
            # PaddleOCR (see ocr/paddle_ocr_engine.py): faster and more
            # accurate than EasyOCR for this use case, and this makes the
            # actual production entry points work at all.
            from ..ocr.paddle_ocr_engine import PaddleOCREngine
            ocr_engine = PaddleOCREngine()
        self.ocr_engine = ocr_engine
        self.preprocessor = OCRPreprocessor()
        # Numeric fields (national_id, birth_date, serial_number) use a
        # dedicated lightweight digit classifier instead of the general
        # OCR engine - see ocr/digit_classifier_engine.py for why.
        self.digit_engine = DigitClassifierEngine()

    # Tried in this order because align_card's contour-based detection can
    # itself be orientation-sensitive (a portrait photo of a landscape card
    # rarely produces a plausible card-shaped contour), so we can't rely on
    # it alone to notice the image is sideways.
    ROTATIONS_TO_TRY = [
        (None, "0°"),
        (cv2.ROTATE_90_CLOCKWISE, "90°"),
        (cv2.ROTATE_180, "180°"),
        (cv2.ROTATE_90_COUNTERCLOCKWISE, "270°"),
    ]

    def process_image(self, image: np.ndarray) -> IDCard:
        """
        Input contract: expects a well-framed card image - the card
        filling (or nearly filling) the frame, as a guided-crop UI would
        produce (user aligns the card to an on-screen outline before
        submitting), NOT an unconstrained photo of a card sitting on an
        arbitrary background. This is a deliberate scope decision, not an
        oversight: align_card's contour-based detection for the latter
        case has a known bug (see scripts/training/generate_realistic_
        photos.py's docstring) and free-form document detection is a
        substantially harder, separate problem. Accuracy figures (98%+ on
        data/synthetic_front/) only hold for well-framed input.

        Tries the image at 0/90/180/270 degree rotations (see
        ROTATIONS_TO_TRY) and returns the best result - this covers "the
        card is framed correctly but the photo itself is sideways/upside
        down" (e.g. phone held in a different orientation), which a
        guided-crop UI does not otherwise prevent. A checksum-valid
        National ID is a hard correctness signal, so the first rotation
        that produces one is returned immediately without trying the rest -
        the common case (already-upright photo) stays a single fast pass.
        Otherwise, the highest-confidence attempt across all four is
        returned so there's always *some* result with an honest status/
        message, rather than silently only ever trying one orientation.
        """
        overall_start = time.time()

        if image is None or image.size == 0:
            result = IDCard()
            result.status = ExtractionStatus.UNREADABLE_IMAGE
            result.messages = [
                "This file couldn't be read as an image. Please upload a "
                "clear JPG or PNG photo of the ID card."
            ]
            result.processing_time_ms = int((time.time() - overall_start) * 1000)
            return result

        best_result: Optional[IDCard] = None
        for rotation, label in self.ROTATIONS_TO_TRY:
            attempt_image = cv2.rotate(image, rotation) if rotation is not None else image
            result = self._process_single_orientation(attempt_image)
            logger.debug(f"Orientation {label}: status={result.status} confidence={result.confidence:.2f}")

            if result.decoded is not None:
                result.processing_time_ms = int((time.time() - overall_start) * 1000)
                return result

            if best_result is None or result.confidence > best_result.confidence:
                best_result = result

        best_result.processing_time_ms = int((time.time() - overall_start) * 1000)
        return best_result

    def _process_single_orientation(self, image: np.ndarray) -> IDCard:
        start_time = time.time()
        result = IDCard()
        field_confidences: List[float] = []

        try:
            # 1. Detect & Align Card (Warp to 1200x750)
            analyzer = LayoutAnalyzer()
            rectified, M = analyzer.align_card(image)
            _, card_buffer = cv2.imencode('.jpg', rectified)
            result.card_image = f"data:image/jpeg;base64,{base64.b64encode(card_buffer).decode('utf-8')}"

            # 2. Classify Side
            side = self.classifier.classify(rectified)
            result.side = side.value
            
            if side == IDSide.FRONT:
                # 3. Get Fixed-Coordinate Crops
                field_crops = analyzer.get_field_crops(rectified)
                
                # 4. OCR on Crops
                refined_data = {}
                refined_crops = {}

                for field_name, crop in field_crops.items():
                    if crop.size > 0:
                        # Process each crop for OCR
                        txt, crop_b64 = self._process_crop(crop, field_name)
                        refined_data[field_name] = txt
                        refined_crops[field_name] = crop_b64
                        field_confidences.append(self._last_field_confidence)
                
                # 5. Populate Data
                result.front = self._populate_front_v4(refined_data, refined_crops)
                
                # 6. Checksum-based Repair & Decoding
                from ..postprocessing.national_id_parser import repair_nid, decode_national_id
                repaired_nid = repair_nid(result.front.national_id)
                if repaired_nid:
                    result.front.national_id = repaired_nid
                    result.decoded = decode_national_id(repaired_nid)
                    if result.decoded:
                        result.front.date_of_birth = result.decoded.birth_date

            # Confidence reflects what we actually measured: mean per-field OCR
            # score, gated by whether the NID passed its Mod-11 checksum (a
            # hard correctness signal that should dominate a soft OCR score -
            # a checksum failure means we *know* something is wrong, no
            # matter how confident the OCR engine was on individual glyphs).
            ocr_confidence = float(np.mean(field_confidences)) if field_confidences else 0.0
            if result.decoded:
                result.confidence = ocr_confidence
            else:
                result.confidence = min(ocr_confidence, 0.3)

            # Plain-language status for a UI to key off of directly, instead
            # of inferring quality from confidence thresholds or null-
            # checking fields itself.
            if result.decoded is not None:
                result.status = ExtractionStatus.SUCCESS
                result.messages = ["National ID extracted and verified successfully."]
            elif result.front is not None:
                result.status = ExtractionStatus.LOW_CONFIDENCE
                result.messages = [
                    "We found a card but couldn't fully verify the National "
                    "ID number. Please check the extracted data carefully, "
                    "or try again with a clearer, well-lit photo."
                ]
            else:
                result.status = ExtractionStatus.NO_CARD_DETECTED
                result.messages = [
                    "We couldn't find an ID card in this image. Make sure "
                    "the whole card is visible, in focus, and well-lit, "
                    "then try again."
                ]

        except Exception as e:
            logger.error(f"Pipeline processing failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            result.status = ExtractionStatus.NO_CARD_DETECTED
            result.messages = [
                "Something went wrong while processing this image. Please "
                "try again with a different photo."
            ]

        result.processing_time_ms = int((time.time() - start_time) * 1000)
        return result

    NUMERIC_FIELDS = {"national_id", "birth_date", "serial_number"}

    def _process_crop(self, crop: np.ndarray, field_name: str) -> tuple[str, str]:
        """Utility for processing individual field crops."""
        # UI Crop (Original color)
        _, buffer = cv2.imencode('.jpg', crop)
        crop_b64 = f"data:image/jpeg;base64,{base64.b64encode(buffer).decode('utf-8')}"

        if field_name in self.NUMERIC_FIELDS:
            # Dedicated per-character digit classifier instead of general
            # OCR - see ocr/digit_classifier_engine.py. It does its own
            # preprocessing/segmentation, so pass the raw crop.
            text = self.digit_engine.extract_digits(crop)
            # Segmentation-based classification doesn't produce a
            # per-detection confidence the way EasyOCR does; treat a
            # plausible-length result as reasonably confident and let
            # checksum validation (a much stronger signal) be the real
            # arbiter for national_id specifically.
            self._last_field_confidence = 0.9 if text else 0.0
            return text.strip(), crop_b64

        # PaddleOCR (the default engine) does its own preprocessing and
        # expects a color image - _enhance_crop's binarization/thresholding
        # was tuned for EasyOCR and produces a single-channel image that
        # breaks PaddleOCR's internal pipeline (it expects an (h,w,3) array).
        # Pass the raw crop directly.
        results = self.ocr_engine.read_layout(crop, mode="auto")

        # Sort results: Top-to-Bottom, then Right-to-Left (for Arabic)
        results.sort(key=lambda r: (np.mean(np.array(r[0])[:, 1]), -np.mean(np.array(r[0])[:, 0])))

        text = " ".join([r[1] for r in results])
        confidence = min([r[2] for r in results], default=0.0)
        self._last_field_confidence = confidence

        # Purification based on field type
        if field_name in ["full_name", "first_name"]:
            import re
            text = re.sub(r'[^\s\u0621-\u064A]', '', text)
            text = " ".join(text.split())
        elif field_name == "address":
            import re
            # Addresses legitimately contain building/street numbers (as
            # printed, Arabic-Indic) and a separator comma - stripping to
            # Arabic-letters-only (as first_name/full_name do) silently
            # dropped them, e.g. "\u0634\u0627\u0631\u0639 \u0667\u0666\u060C \u0627\u0644\u0634\u0631\u0642\u064A\u0629" -> "\u0634\u0627\u0631\u0639 \u0627\u0644\u0634\u0631\u0642\u064A\u0629".
            text = re.sub(r'[^\s\u0621-\u064A0-9\u0660-\u0669\u060C]', '', text)
            text = " ".join(text.split())
        # Numeric fields (national_id, birth_date, serial_number) return
        # early above via the digit classifier and never reach this point.

        return text.strip(), crop_b64

    def _populate_front_v4(self, refined_data: dict, refined_crops: dict) -> IDCardFront:
        return IDCardFront(
            first_name=refined_data.get("first_name", ""),
            full_name=refined_data.get("full_name", ""),
            address=refined_data.get("address", ""),
            national_id=refined_data.get("national_id", ""),
            date_of_birth=refined_data.get("birth_date", ""),
            card_serial_number=refined_data.get("serial_number", ""),
            field_crops=refined_crops,
            raw_arabic=refined_data
        )

    def _enhance_crop(self, crop: np.ndarray) -> np.ndarray:
        """Applies upscaling, sharpening and adaptive thresholding for high-res OCR pass."""
        try:
            if crop.size == 0: return crop
            
            # 1. Upscaling (2x) using Lanczos interpolation
            crop = cv2.resize(crop, None, fx=2, fy=2, interpolation=cv2.INTER_LANCZOS4)
            
            # 2. To Grayscale
            if len(crop.shape) == 3:
                gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            else:
                gray = crop

            # 3. Denoising
            denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
            
            # 4. Adaptive Thresholding (Hamza's B&W approach)
            thresh = cv2.adaptiveThreshold(denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 5)
            
            # 5. Contrast Stretching
            min_val, max_val, _, _ = cv2.minMaxLoc(denoised)
            stretched = cv2.convertScaleAbs(denoised, alpha=255.0/(max_val - min_val), beta=-min_val * 255.0/(max_val - min_val))
            
            return thresh
        except Exception as e:
            logger.error(f"Enhancement failed: {e}")
            return crop
