import cv2
import numpy as np
import logging
from typing import List, Tuple, Dict, Any, Optional

logger = logging.getLogger(__name__)

class LayoutAnalyzer:
    """
    Handles ID card detection, perspective alignment, and fixed-coordinate field mapping.
    Uses classical Computer Vision (Contours, Perspective Transform) to normalize the ID.
    """
    def __init__(self, target_width=1200, target_height=750):
        self.target_width = target_width
        self.target_height = target_height
        
        # Normalized coordinates for Front Side (in 1200x750 space)
        # These are approximate and might need further tuning
        self.FRONT_FIELDS = {
            "first_name":    [600, 150, 550, 70],
            "full_name":     [550, 230, 600, 90],
            "address":       [550, 330, 600, 160],
            "national_id":   [550, 560, 600, 90],
            "birth_date":    [50, 560, 450, 90],
            "serial_number": [50, 660, 450, 70],
        }

    def get_card_contour(self, image: np.ndarray) -> Optional[np.ndarray]:
        """Finds the largest rectangular-ish contour in the image."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Try different thresholding methods
        # 1. Otsu thresholding
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # 2. Morphological closing to fill gaps
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=3)
        
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            # Try Canny if thresholding fails
            edged = cv2.Canny(gray, 50, 200)
            contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]
        
        for c in contours:
            area = cv2.contourArea(c)
            if area < (image.shape[0] * image.shape[1] * 0.1): # Too small
                continue
                
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            
            if len(approx) == 4:
                return approx
        
        # If no 4-point contour found, return the bounding box of the largest contour
        if contours:
            rect = cv2.minAreaRect(contours[0])
            box = cv2.boxPoints(rect)
            return box.astype(np.intp)
            
        return None

    def align_card(self, image: np.ndarray, contour: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Warps the image to the standard 1200x750 canvas using perspective transform.
        """
        if contour is None:
            contour = self.get_card_contour(image)
            
        if contour is None:
            logger.warning("Could not detect card contour. Falling back to simple resize.")
            return cv2.resize(image, (self.target_width, self.target_height)), None

        # Reorder points: top-left, top-right, bottom-right, bottom-left
        pts = contour.reshape(-1, 2)
        rect = np.zeros((4, 2), dtype="float32")
        
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        
        dst = np.array([
            [0, 0],
            [self.target_width - 1, 0],
            [self.target_width - 1, self.target_height - 1],
            [0, self.target_height - 1]
        ], dtype="float32")
        
        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(image, M, (self.target_width, self.target_height))
        
        return warped, M

    def get_field_crops(self, aligned_image: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Intelligently detects fields using horizontal projections to find text lines.
        Falls back to fixed coordinates if dynamic detection fails.
        """
        # 1. Grayscale and threshold for projection
        gray = cv2.cvtColor(aligned_image, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
        
        # 2. Right Side Projection (Names, Address, NID)
        # Scan from x=500 to x=1150
        right_search = thresh[100:700, 500:1150]
        h_proj = np.sum(right_search, axis=1)
        
        # Find peaks (text lines)
        peaks = []
        in_peak = False
        start = 0
        threshold = np.max(h_proj) * 0.1
        for i, val in enumerate(h_proj):
            if val > threshold and not in_peak:
                in_peak = True
                start = i
            elif val < threshold and in_peak:
                in_peak = False
                peaks.append((start + 100, i + 100)) # Add vertical offset
        
        # 3. Assign fields to peaks
        # Filter out headers (usually peaks before y=180)
        content_peaks = [p for p in peaks if p[0] > 170]
        
        dynamic_boxes = {}
        
        if len(content_peaks) >= 2:
            # First Name
            y1, y2 = content_peaks[0]
            dynamic_boxes["first_name"] = [600, y1-5, 550, (y2-y1)+10]
            
            # Full Name
            y1, y2 = content_peaks[1]
            dynamic_boxes["full_name"] = [550, y1-5, 600, (y2-y1)+10]
            
            # Address (next 1-2 peaks)
            if len(content_peaks) >= 3:
                y1_addr, _ = content_peaks[2]
                _, y2_addr = content_peaks[min(len(content_peaks)-1, 3)]
                dynamic_boxes["address"] = [550, y1_addr-5, 600, (y2_addr-y1_addr)+10]
            
            # National ID (look for peak around y=600)
            nid_peaks = [p for p in content_peaks if 550 < p[0] < 650]
            if nid_peaks:
                y1, y2 = nid_peaks[0]
                dynamic_boxes["national_id"] = [550, y1-10, 600, (y2-y1)+20]
        
        # 4. Left Side (Birth Date, Serial)
        left_search = thresh[500:750, 40:450]
        l_proj = np.sum(left_search, axis=1)
        l_peaks = []
        in_peak = False
        start = 0
        l_thresh = np.max(l_proj) * 0.1 if np.max(l_proj) > 0 else 0
        for i, val in enumerate(l_proj):
            if val > l_thresh and not in_peak:
                in_peak = True
                start = i
            elif val < l_thresh and in_peak:
                in_peak = False
                l_peaks.append((start + 500, i + 500))
        
        if len(l_peaks) >= 2:
            y1, y2 = l_peaks[0]
            dynamic_boxes["birth_date"] = [40, y1-10, 420, (y2-y1)+20]
            y1, y2 = l_peaks[1]
            dynamic_boxes["serial_number"] = [40, y1-5, 420, (y2-y1)+10]

        # 5. Merge with fallbacks
        final_boxes = {}
        for field_name, fixed_coords in self.FRONT_FIELDS.items():
            if field_name in dynamic_boxes:
                # Sanity check: if dynamic box is too small or weird, use fixed
                d_box = dynamic_boxes[field_name]
                if d_box[3] > 10: # Height check
                    final_boxes[field_name] = d_box
                    continue
            final_boxes[field_name] = fixed_coords

        # 6. Crop
        crops = {}
        for field_name, [x, y, w, h] in final_boxes.items():
            # Constrain to image boundaries
            y_end = min(y + h, aligned_image.shape[0])
            x_end = min(x + w, aligned_image.shape[1])
            crop = aligned_image[max(0, y):y_end, max(0, x):x_end]
            crops[field_name] = crop
            
        # Store final boxes for debugging/visualization if needed
        self._last_boxes = final_boxes
        
        return crops

    def get_field_boxes_original(self, M_inv: np.ndarray) -> Dict[str, List[Tuple[int, int]]]:
        """
        Maps the dynamic coordinates back to the original image space.
        """
        boxes = getattr(self, '_last_boxes', self.FRONT_FIELDS)
        original_boxes = {}
        for field_name, [x, y, w, h] in boxes.items():
            pts = np.array([[[x, y]], [[x+w, y]], [[x+w, y+h]], [[x, y+h]]], dtype="float32")
            transformed = cv2.perspectiveTransform(pts, M_inv)
            original_boxes[field_name] = [tuple(p[0].astype(int)) for p in transformed]
        return original_boxes
