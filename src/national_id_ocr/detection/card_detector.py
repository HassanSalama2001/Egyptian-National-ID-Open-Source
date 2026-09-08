import cv2
import numpy as np
import logging
from typing import Optional, Tuple
from ..core.exceptions import CardNotFoundError
from .perspective import four_point_transform

logger = logging.getLogger(__name__)

class CardDetector:
    def __init__(self, target_width: int = 1000, target_height: int = 630):
        self.target_width = target_width
        self.target_height = target_height
        self.aspect_ratio = target_width / target_height

    def detect(self, image: np.ndarray) -> np.ndarray:
        """
        Detects the card boundary in the image and returns the warped top-down view.
        """
        try:
            corners = self.find_card_corners(image)
            if corners is None:
                logger.warning("Card boundary not found, assuming image is already cropped.")
                return self._finalize_image(image)
                
            warped = four_point_transform(image, corners, self.target_width, self.target_height)
            
            # 3. Crop White Borders (Aggressive)
            gray_warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray_warped, 250, 255, cv2.THRESH_BINARY_INV)
            coords = cv2.findNonZero(thresh)
            if coords is not None:
                x, y, w, h = cv2.boundingRect(coords)
                if w > 0.5 * self.target_width and h > 0.5 * self.target_height:
                    warped = warped[y:y+h, x:x+w]
            
            return self._finalize_image(warped)
        except Exception as e:
            logger.error(f"Detection failed: {e}. Returning original.")
            return self._finalize_image(image)

    def _finalize_image(self, image: np.ndarray) -> np.ndarray:
        """Resizes and rotates to standard landscape orientation."""
        res = cv2.resize(image, (self.target_width, self.target_height))
        h, w = res.shape[:2]
        if h > w:
            res = cv2.rotate(res, cv2.ROTATE_90_CLOCKWISE)
        return res

    def find_card_corners(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Robustly finds 4 corners using thresholding and morphological operations.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # 1. Try Adaptive Thresholding to find the card body
        thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                     cv2.THRESH_BINARY_INV, 11, 2)
        
        # Close gaps in the card body
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        
        # 2. Find Contours
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]
        
        for contour in contours:
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
            
            if len(approx) == 4:
                area = cv2.contourArea(approx)
                img_area = image.shape[0] * image.shape[1]
                
                # Check for card aspect ratio (around 1.58)
                (x, y, w, h) = cv2.boundingRect(approx)
                ar = w / float(h) if h > 0 else 0
                
                if area > img_area * 0.1 and (1.0 < ar < 2.5):
                    return approx.reshape(4, 2)
                    
        # 3. Fallback: Standard Canny if thresholding was too noisy
        edged = cv2.Canny(blurred, 50, 200)
        contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]
        
        for contour in contours:
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
            
            if len(approx) == 4:
                area = cv2.contourArea(approx)
                img_area = image.shape[0] * image.shape[1]
                if area > img_area * 0.2: # Must be at least 20% of image
                    return approx.reshape(4, 2)
                    
        return None
