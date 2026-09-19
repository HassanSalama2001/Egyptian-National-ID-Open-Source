from abc import ABC, abstractmethod
import numpy as np
import cv2

class OCREngine(ABC):
    @abstractmethod
    def extract_text(self, image: np.ndarray, lang: str = "ara", **kwargs) -> str:
        """Extracts plain text from image."""
        pass

    @abstractmethod
    def read_layout(self, image: np.ndarray, lang: str = "ara", **kwargs) -> list:
        """
        Extracts text and spatial metadata (bounding boxes).
        Returns: List of tuples (bbox, text, confidence)
        """
        pass

class OCRPreprocessor:
    @staticmethod
    def preprocess(image: np.ndarray, mode: str = "neural") -> np.ndarray:
        """
        Enhances image for better OCR results.
        Args:
            image: ROI crop
            mode: 'neural' (EasyOCR) or 'binarized' (Tesseract)
        Returns:
            Preprocessed image
        """
        if image is None or image.size == 0:
            return image
            
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
            
        # 1. CLAHE (Contrast Limited Adaptive Histogram Equalization)
        # This is CRITICAL for seeing through holograms
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        # 2. Adaptive Upscaling (Min height 64px for neural OCR)
        h, w = enhanced.shape
        scale_factor = 1.0
        if h < 64:
            scale_factor = 64.0 / h
            
        # If we are upscaling, add logical padding to prevent edge-cutting
        pad = int(h * 0.1) # 10% padding
        padded = cv2.copyMakeBorder(enhanced, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
        
        target_w = int(padded.shape[1] * scale_factor)
        target_h = int(padded.shape[0] * scale_factor)
        upscaled = cv2.resize(padded, (target_w, target_h), interpolation=cv2.INTER_CUBIC)
        
        if mode == "binarized":
            # Adaptive Thresholding for Tesseract
            # Add back sharpening only for binarized mode if needed
            gaussian_blur = cv2.GaussianBlur(upscaled, (0, 0), 3)
            sharpened = cv2.addWeighted(upscaled, 1.5, gaussian_blur, -0.5, 0)
            thresh = cv2.adaptiveThreshold(sharpened, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                         cv2.THRESH_BINARY, 11, 2)
            return thresh
            
        # For EasyOCR (Neural), raw grayscale-enhanced is usually better
        return upscaled
