from .base import RegionSegmenter
import numpy as np
from typing import Dict

class FrontSegmenter(RegionSegmenter):
    # Mapping: field -> (x%, y%, w%, h%)
    FRONT_CONFIG = {
        "first_name": (40.0, 15.0, 55.0, 12.0),
        "full_name": (25.0, 27.0, 70.0, 12.0),
        "address": (20.0, 40.0, 75.0, 20.0),
        "national_id": (25.0, 72.0, 70.0, 15.0),
        "date_of_birth": (5.0, 65.0, 30.0, 12.0),
        "card_serial_number": (3.0, 85.0, 25.0, 10.0),
    }

    def segment(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """Segments the front of the ID card into ROIs."""
        return self.extract_rois(image, self.FRONT_CONFIG)
