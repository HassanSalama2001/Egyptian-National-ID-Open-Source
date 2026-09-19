import easyocr
import numpy as np
from .base import OCREngine
import logging
import cv2

logger = logging.getLogger(__name__)

class EasyOCREngine(OCREngine):
    def __init__(self, gpu=False):
        """
        Initializes EasyOCR reader. 
        Note: The first run will download models.
        """
        self.reader = easyocr.Reader(['ar', 'en'], gpu=gpu)

    def extract_text(self, image: np.ndarray, lang: str = "ara", allowlist: str = None, mode: str = "auto") -> str:
        """Extracts plain text using EasyOCR."""
        results = self.read_layout(image, lang=lang, allowlist=allowlist, mode=mode)
        return " ".join([r[1] for r in results]).strip()

    def read_layout(self, image: np.ndarray, lang: str = "ara", allowlist: str = None, mode: str = "auto") -> list:
        """
        Extracts text and spatial metadata (bounding boxes) using EasyOCR.
        Returns: List of tuples (bbox, text, confidence)
        """
        if mode == "numeric" and not allowlist:
            allowlist = "0123456789\u0660\u0661\u0662\u0663\u0664\u0665\u0666\u0667\u0668\u0669"
            
        try:
            if image is None or image.size == 0:
                return []
                
            if len(image.shape) == 3 and image.shape[2] == 3:
                rgb_image = image[:, :, ::-1]
            else:
                rgb_image = image

            # Primary Attempt with detail=1
            results = self.reader.readtext(rgb_image, detail=1, allowlist=allowlist)
            
            # Rotation Fallback: If no results found, try rotating
            if not results:
                for angle in [cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_180, cv2.ROTATE_90_COUNTERCLOCKWISE]:
                    rotated = cv2.rotate(rgb_image, angle)
                    results = self.reader.readtext(rotated, detail=1, allowlist=allowlist)
                    if results:
                        # Note: bbox coordinates will be relative to the rotated image.
                        # For now, we return them as is, but later we might want to rotate them back.
                        break
                        
            return results
        except Exception as e:
            logger.error(f"EasyOCR layout extraction failed: {e}")
            return []
        except Exception as e:
            logger.error(f"EasyOCR extraction failed: {e}")
            return ""
