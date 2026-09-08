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
from ..postprocessing.numeral_converter import normalize_arabic_numerals, clean_ocr_text
from ..postprocessing.national_id_parser import decode_national_id, find_nid_in_text
from ..models.id_card import IDCard, IDCardFront, IDCardBack
from .layout_analyzer import LayoutAnalyzer
from .field_clusterer import FieldClusterer

logger = logging.getLogger(__name__)

class Pipeline:
    def __init__(self, ocr_engine: Optional[OCREngine] = None, tesseract_cmd: str = "tesseract"):
        self.detector = CardDetector()
        self.classifier = SideClassifier()
        self.ocr_engine = ocr_engine
        self.preprocessor = OCRPreprocessor()

    def process_image(self, image: np.ndarray) -> IDCard:
        """
        Executes the classical CV-based pipeline (Alignment -> Fixed Boxing -> Extraction).
        """
        start_time = time.time()
        result = IDCard()
        field_confidences: List[float] = []

        try:
            # 1. Detect & Align Card (Warp to 1200x750)
            analyzer = LayoutAnalyzer()
            rectified, M = analyzer.align_card(image)
            
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
            
        except Exception as e:
            logger.error(f"Pipeline processing failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            
        result.processing_time_ms = int((time.time() - start_time) * 1000)
        return result

    NUMERIC_FIELDS = {"national_id", "birth_date", "serial_number"}

    def _process_crop(self, crop: np.ndarray, field_name: str) -> tuple[str, str]:
        """Utility for processing individual field crops."""
        # UI Crop (Original color)
        _, buffer = cv2.imencode('.jpg', crop)
        crop_b64 = f"data:image/jpeg;base64,{base64.b64encode(buffer).decode('utf-8')}"

        # Enhancement for OCR
        enhanced_crop = self._enhance_crop(crop)

        # OCR - constrain to a digit allowlist for numeric fields so the
        # engine can't emit letters into fields that must be pure digits
        mode = "numeric" if field_name in self.NUMERIC_FIELDS else "auto"
        results = self.ocr_engine.read_layout(enhanced_crop, mode=mode)

        # Sort results: Top-to-Bottom, then Right-to-Left (for Arabic)
        results.sort(key=lambda r: (np.mean(np.array(r[0])[:, 1]), -np.mean(np.array(r[0])[:, 0])))

        text = " ".join([r[1] for r in results])
        confidence = min([r[2] for r in results], default=0.0)
        self._last_field_confidence = confidence

        # Purification based on field type
        if field_name in ["full_name", "first_name", "address"]:
            import re
            text = re.sub(r'[^\s\u0621-\u064A]', '', text)
            text = " ".join(text.split())
        elif field_name in ["national_id", "birth_date"]:
             text = normalize_arabic_numerals(text).replace(" ", "")
             text = "".join([c for c in text if c.isdigit()])

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
