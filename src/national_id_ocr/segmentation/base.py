from typing import Dict, Tuple
import numpy as np

class RegionSegmenter:
    def __init__(self, target_width: int = 1000, target_height: int = 630):
        self.target_width = target_width
        self.target_height = target_height

    def extract_rois(self, image: np.ndarray, config: Dict[str, Tuple[float, float, float, float]]) -> Dict[str, np.ndarray]:
        """
        Extracts regions of interest based on a configuration of percentages.
        Args:
            image: Normalized card image (likely 1000x630)
            config: Dict mapping field names to (x%, y%, w%, h%)
        Returns:
            Dict mapping field names to cropped images
        """
        rois = {}
        h, w = image.shape[:2]
        
        for field, (px, py, pw, ph) in config.items():
            x = int(px * w / 100)
            y = int(py * h / 100)
            rw = int(pw * w / 100)
            rh = int(ph * h / 100)
            
            # Clamp to image boundaries
            y2 = min(y + rh, h)
            x2 = min(x + rw, w)
            y1 = max(y, 0)
            x1 = max(x, 0)
            
            rois[field] = image[y1:y2, x1:x2]
            
        return rois
