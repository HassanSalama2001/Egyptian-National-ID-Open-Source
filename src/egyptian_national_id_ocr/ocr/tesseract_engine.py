import pytesseract
import numpy as np
from .base import OCREngine
import logging

logger = logging.getLogger(__name__)

class TesseractEngine(OCREngine):
    def __init__(self, tesseract_cmd: str = "tesseract"):
        self.tesseract_cmd = tesseract_cmd
        pytesseract.pytesseract.tesseract_cmd = tesseract_cmd

    def extract_text(self, image: np.ndarray, lang: str = "ara", psm: int = 7) -> str:
        """
        Extracts text using Tesseract.
        Default PSM 7 is 'Treat the image as a single text line'.
        """
        try:
            # Config for better Arabic/Numeral support if needed
            config = f'--psm {psm}'
            
            text = pytesseract.image_to_string(image, lang=lang, config=config)
            return text.strip()
        except Exception as e:
            logger.error(f"Tesseract OCR failed: {e}")
            return ""
