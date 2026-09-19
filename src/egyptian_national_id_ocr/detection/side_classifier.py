import cv2
import numpy as np
import os
from enum import Enum
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class IDSide(Enum):
    FRONT = "front"
    BACK = "back"
    UNKNOWN = "unknown"

class SideClassifier:
    def __init__(self):
        # Load templates once at initialization
        template_dir = Path(__file__).parent / "templates"
        self.front_template = cv2.imread(str(template_dir / "front_header.jpg"), 0)
        self.back_template = cv2.imread(str(template_dir / "back_eagle.jpg"), 0)
        
        if self.front_template is None or self.back_template is None:
            logger.error(f"Failed to load templates from {template_dir}")

    def classify(self, card_image: np.ndarray) -> IDSide:
        """
        Classifies the ID side (front/back/unknown).

        Deliberately does NOT rotate card_image in-place, even though it
        checks both the normal and 180-rotated template-match score to
        decide the side robustly. That in-place "fix upside-down" flip
        used to exist here, but it's redundant with (and actively fights)
        Pipeline.ROTATIONS_TO_TRY, which already tries the whole image at
        0/90/180/270 as independent attempts and keeps whichever produces
        the best real result. Template-match scores are noisy on a real
        (as opposed to synthetic) photo - on a genuinely real card, the
        180-rotated score sometimes edges out the upright score by a small
        margin even when the photo is already the right way up. Letting
        this classifier silently flip the image on that noise meant a
        correctly-oriented photo could get flipped upside-down right
        before field cropping, while the outer rotation loop's genuinely
        correct attempt got overridden by this false "correction" - field
        crops (calibrated for upright) would then land on nothing.
        """
        if card_image is None or card_image.size == 0:
            return IDSide.UNKNOWN

        gray_card = cv2.cvtColor(card_image, cv2.COLOR_BGR2GRAY)

        # 1. Normal orientation scores
        f_up, b_up = self._get_match_score(gray_card, self.front_template), self._get_match_score(gray_card, self.back_template)

        # 2. 180-degree rotation scores
        gray_rot = cv2.rotate(gray_card, cv2.ROTATE_180)
        f_down, b_down = self._get_match_score(gray_rot, self.front_template), self._get_match_score(gray_rot, self.back_template)

        scores = {
            IDSide.FRONT: max(f_up, f_down),
            IDSide.BACK: max(b_up, b_down)
        }

        best_side = max(scores, key=scores.get)
        best_score = scores[best_side]

        logger.debug(f"Scores -> Front: {scores[IDSide.FRONT]:.3f}, Back: {scores[IDSide.BACK]:.3f}")

        if best_score < 0.2: # Low confidence
            return IDSide.UNKNOWN

        return best_side

    def _get_match_score(self, image: np.ndarray, template: np.ndarray) -> float:
        """Helper to get highest template match score."""
        if template is None or image is None:
            return 0.0
            
        # Standard cross-correlation matching
        # Note: We assume the card is already perspective-corrected to approx 1000px width
        res = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(res)
        return max_val
