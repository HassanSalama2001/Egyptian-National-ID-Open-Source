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
        
        # Field boxes [x, y, w, h] in the 1200x750 rectified space, derived
        # from measuring assets/front_template.jpg directly (see
        # scripts/training/calibrate_template.py) and scaling by the same
        # factor align_card uses to warp to this canvas. These were
        # previously hand-guessed and did not match the real template at
        # all (e.g. first_name pointed at the card's green title text) -
        # see the Phase 1/2 OCR redesign notes.
        self.FRONT_FIELDS = {
            # x starts at 280, not 15: the photo placeholder box (with its
            # own "الصورة الشخصية" label) occupies roughly x=56-267,
            # y=73-363 in this space - a wider box would bleed that label
            # text into the name/address crops.
            "first_name":    [280, 205, 618, 110],
            "full_name":     [280, 265, 546, 110],
            "address":       [280, 351, 526, 110],
            # x starts at 380, past the DOB row's own label text
            # ("تاريخ الميلاد" spans scaled x=170-360), which sits at
            # a similar height and was bleeding into this crop.
            "national_id":   [380, 575, 415, 110],
            "birth_date":    [10, 568, 150, 90],
            "serial_number": [10, 669, 156, 90],
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
        
        # If no 4-point contour found, fall back to the bounding box of the
        # largest contour - but only if it's actually plausible as a card
        # (same area threshold as above). Without this check, a tiny
        # spurious contour (e.g. a logo or a line of text) could be warped
        # to fill the entire output canvas, corrupting every field crop
        # taken from the "rectified" result.
        min_area = image.shape[0] * image.shape[1] * 0.1
        if contours and cv2.contourArea(contours[0]) >= min_area:
            rect = cv2.minAreaRect(contours[0])
            box = cv2.boxPoints(rect)
            return box.astype(np.intp)

        # No plausible card-sized contour found - the caller's fallback
        # (treat the whole image as the card, no warping) is more honest
        # than forcing a bad warp.
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
        Crops each field using fixed, calibrated boxes (self.FRONT_FIELDS).

        This used to attempt "dynamic" detection first via horizontal-
        projection peak counting, positionally assigning peaks 0/1/2.. to
        first_name/full_name/address/etc. That approach's box formulas used
        hardcoded x-offsets (e.g. x=600, width=550) that were never
        validated against where the text actually sits, and peak-counting
        silently breaks the moment a field wraps to an extra line or a peak
        merges/splits differently than assumed. Since align_card now
        reliably produces a consistent 1200x750 canvas (see the contour
        area-check fix), fixed calibrated boxes are simpler and more
        reliable than guessing peak order. See
        scripts/training/calibrate_template.py for how these were derived.
        """
        final_boxes = self.FRONT_FIELDS

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
