import numpy as np
from typing import List, Tuple, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class FieldClusterer:
    """
    Groups raw OCR text boxes into logical identity fields (Name, Address, NID)
    using spatial clustering and character set density.
    """
    def __init__(self, image_shape: Tuple[int, int]):
        self.height, self.width = image_shape

    def cluster(self, ocr_results: List[Tuple[Any, str, float]]) -> Dict[str, List[float]]:
        """
        Processes discovery-pass OCR results and returns bounding boxes for each field.
        Format: {field_name: [x1, y1, x2, y2]} in normalized coordinates (0.0 to 1.0).
        """
        raw_boxes = []
        for bbox, text, conf in ocr_results:
            pts = np.array(bbox)
            min_x = np.min(pts[:, 0]) / self.width
            max_x = np.max(pts[:, 0]) / self.width
            min_y = np.min(pts[:, 1]) / self.height
            max_y = np.max(pts[:, 1]) / self.height
            
            raw_boxes.append({
                "text": text,
                "bbox": [min_x, min_y, max_x, max_y],
                "x_mid": (min_x + max_x) / 2,
                "y_mid": (min_y + max_y) / 2,
                "is_arabic": any("\u0621" <= c <= "\u064A" for c in text),
                "is_numeric": any(c.isdigit() for c in text.replace(" ", ""))
            })

        fields = {}

        # 1. National ID Cluster (Look for 14-digit sequences in bottom half)
        nid_boxes = []
        for b in raw_boxes:
            clean_digits = "".join([c for c in b["text"] if c.isdigit()])
            if b["y_mid"] > 0.60 and len(clean_digits) >= 4:
                nid_boxes.append(b)
        
        if nid_boxes:
            fields["national_id"] = self._get_merged_bbox(nid_boxes)

        # 2. Name Cluster (Look for Arabic blocks in the Top-Middle right area)
        name_boxes = []
        for b in raw_boxes:
            # Name ROI: y between 0.25 and 0.55, primarily right side
            if 0.25 <= b["y_mid"] <= 0.55 and b["x_mid"] > 0.25:
                if b["is_arabic"]:
                    name_boxes.append(b)
        
        if name_boxes:
            fields["full_name"] = self._get_merged_bbox(name_boxes)

        # 3. Address Cluster (Look for Arabic blocks below Name)
        addr_boxes = []
        for b in raw_boxes:
            # Address ROI: y between 0.50 and 0.85
            if 0.50 <= b["y_mid"] <= 0.85 and b["x_mid"] > 0.15:
                if b["is_arabic"] and b not in name_boxes:
                    # Filter out NID boxes that might have been picked up
                    if not b["is_numeric"] or len("".join([c for c in b["text"] if c.isdigit()])) < 10:
                        addr_boxes.append(b)

        if addr_boxes:
            fields["address"] = self._get_merged_bbox(addr_boxes)

        # 4. Serial Number (Bottom Left)
        serial_boxes = []
        for b in raw_boxes:
            if b["y_mid"] > 0.75 and b["x_mid"] < 0.45:
                if b["is_numeric"]:
                    serial_boxes.append(b)
        
        if serial_boxes:
            fields["card_serial_number"] = self._get_merged_bbox(serial_boxes)

        return fields

    def _get_merged_bbox(self, boxes: List[Dict], padding: float = 0.02) -> List[float]:
        """Calculates the union of all boxes with optional padding."""
        if not boxes: return [0,0,0,0]
        
        x1 = min([b["bbox"][0] for b in boxes])
        y1 = min([b["bbox"][1] for b in boxes])
        x2 = max([b["bbox"][2] for b in boxes])
        y2 = max([b["bbox"][3] for b in boxes])
        
        # Apply padding and clamp to [0, 1]
        return [
            max(0.0, float(x1 - padding)),
            max(0.0, float(y1 - padding)),
            min(1.0, float(x2 + padding)),
            min(1.0, float(y2 + padding))
        ]
