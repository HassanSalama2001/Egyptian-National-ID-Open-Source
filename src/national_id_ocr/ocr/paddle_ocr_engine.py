"""
PaddleOCR-based engine for open-vocabulary Arabic text fields (name,
address) - replaces EasyOCR for these fields.

Why: EasyOCR (PyTorch) took ~4s/field and required position-based
sorting/joining of fragmented detections to reconstruct multi-word text.
PaddleOCR's mobile detector + Arabic mobile recognizer
(PP-OCRv5_mobile_det + arabic_PP-OCRv5_mobile_rec) does detection+
recognition+reading-order in one pass, returning whole-line text
directly: ~0.2-0.3s/field once loaded, exact-matched both full_name and
address in testing. Also drops the PyTorch/EasyOCR dependency (~1-2GB)
entirely once this fully replaces it (see pipeline.py wiring).

Note: enable_mkldnn=False is required - the default oneDNN CPU backend
in this PaddlePaddle build (3.3.1) raises
"NotImplementedError: ConvertPirAttribute2RuntimeAttribute not support
[pir::ArrayAttribute<pir::DoubleAttribute>]" on inference on this
machine. Disabling it is slightly slower per-call but actually works.
"""
import logging
from typing import List, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class PaddleOCREngine:
    def __init__(self):
        from paddleocr import PaddleOCR
        self._ocr = PaddleOCR(
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            enable_mkldnn=False,
            text_detection_model_name="PP-OCRv5_mobile_det",
            text_recognition_model_name="arabic_PP-OCRv5_mobile_rec",
        )

    def extract_text(self, image: np.ndarray, lang: str = "ara", **kwargs) -> str:
        results = self.read_layout(image, lang=lang, **kwargs)
        return " ".join(r[1] for r in results).strip()

    def read_layout(self, image: np.ndarray, lang: str = "ara", **kwargs) -> List[Tuple]:
        """Returns (bbox, text, confidence) tuples - bbox is a 4x2 polygon
        array, matching the shape EasyOCREngine produces, so pipeline.py's
        existing position-sort logic keeps working unchanged."""
        if image is None or image.size == 0:
            return []

        try:
            pages = self._ocr.predict(image)
        except Exception as e:
            logger.error(f"PaddleOCR prediction failed: {e}")
            return []

        results = []
        for page in pages:
            texts = page.get("rec_texts", [])
            scores = page.get("rec_scores", [])
            polys = page.get("rec_polys", [])
            for text, score, poly in zip(texts, scores, polys):
                if text:
                    results.append((np.array(poly), text, float(score)))
        return results
