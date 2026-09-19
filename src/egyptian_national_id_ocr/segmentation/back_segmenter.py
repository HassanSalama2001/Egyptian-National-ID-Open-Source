from .base import RegionSegmenter
import numpy as np
from typing import Dict

class BackSegmenter(RegionSegmenter):
    # Mapping: field -> (x%, y%, w%, h%)
    # Estimated based on layout, will needs refinement
    BACK_CONFIG = {
        "national_id": (28.0, 5.0, 55.0, 10.0),
        "issue_date": (28.0, 10.0, 15.0, 10.0),
        "profession": (40.0, 15.0, 45.0, 15.0),
        "gender": (70.0, 30.0, 15.0, 10.0),
        "religion": (55.0, 30.0, 15.0, 10.0),
        "marital_status": (35.0, 30.0, 15.0, 10.0),
        "expiry_date": (33.0, 48.0, 50.0, 12.0),
    }

    def segment(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """Segments the back of the ID card into ROIs."""
        return self.extract_rois(image, self.BACK_CONFIG)
