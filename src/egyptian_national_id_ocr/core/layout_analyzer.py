import cv2
import numpy as np
import logging
from typing import List, Tuple, Dict, Any, Optional

logger = logging.getLogger(__name__)

# ISO/IEC 7810 ID-1 card ratio (85.60mm x 53.98mm) - a real ID card contour
# should be close to this regardless of size. Without this check,
# get_card_contour could accept any sufficiently-large 4-point blob (e.g. a
# row of text/boxes spanning most of the width) as "the card", producing a
# nonsensical perspective warp - this actually happened with the back
# template's gender/religion/marital-status row (a ~4:1 strip that cleared
# the area threshold easily). Tolerance is generous (real contours from
# approxPolyDP are rarely exact) but tight enough to reject shapes that
# aren't card-like at all.
CARD_ASPECT_RATIO = 85.60 / 53.98
CARD_ASPECT_TOLERANCE = 0.4


# Separate, much tighter tolerance for "is this ENTIRE photo's frame
# already just the card" (align_card's whole-frame shortcut) - a
# different question from "does this candidate shape found WITHIN a
# photo look card-like enough", which is what CARD_ASPECT_TOLERANCE above
# is for. Confirmed necessary: reusing CARD_ASPECT_TOLERANCE's wide +/-0.4
# band for the whole-frame check caused a stress-test canvas whose own
# aspect ratio (1.39, chosen somewhat arbitrarily for its background
# margin) to fall inside that band purely by coincidence, wrongly
# triggering the shortcut and treating an entire scene (card + background)
# as if it were just the card - a bigger, silent error than the one the
# shortcut was added to fix. A real tightly-framed or guided-crop photo's
# aspect ratio should be CLOSE to the card's, not merely in the same
# ballpark - +/-0.15 comfortably covers real-world framing/lens variance
# (confirmed against a real photo at 1.633, only 0.047 off) while
# correctly rejecting a frame that clearly still has room for background.
FRAME_ASPECT_TOLERANCE = 0.15


class LayoutAnalyzer:
    """
    Handles ID card detection, perspective alignment, and fixed-coordinate field mapping.
    Uses classical Computer Vision (Contours, Perspective Transform) to normalize the ID.
    """
    def __init__(self, target_width=1200, target_height=750):
        self.target_width = target_width
        self.target_height = target_height
        
        # Field boxes [x, y, w, h] in the 1200x750 rectified space.
        #
        # IMPORTANT: these were re-derived directly from a real card photo
        # (via scripts/debug_field_grid.py - a privacy-preserving tool that
        # overlays a labeled coordinate grid on the aligned card so the
        # user can read off real text positions without ever sending us
        # the photo itself), NOT from the blank template asset. The
        # earlier template-based calibration was measuring the wrong
        # reference point - every field's real printed value sits further
        # right and wider than the template suggested (the user first
        # flagged this; the grid overlay confirmed it precisely: e.g.
        # first_name's real text sat at rect x~1030-1110, while the old
        # box only reached x=898, missing it entirely).
        self.FRONT_FIELDS = {
            # Height tightened (was 110, bled into full_name's row below it
            # and confused PaddleOCR into returning nothing for either
            # line - confirmed against a real card: first_name's crop
            # visibly showed both name lines stacked).
            "first_name":    [750, 200, 390, 65],
            # Height trimmed from 110 to 95 - it overlapped address's box
            # by 10px (265+110=375 vs address's y0=365), so a detection
            # whose center landed in that shared strip could get bucketed
            # into either field depending on dict order rather than which
            # one it actually belonged to (caught via synthetic testing:
            # address's line bled into full_name's output).
            "full_name":     [550, 265, 590, 95],
            # Real cards wrap the address across 2 lines (building/street,
            # then district - governorate) - confirmed against a real card
            # photo. Box re-measured from the same real-photo grid; widened
            # further after a real card's 2nd line ("district - governorate")
            # came back truncated.
            "address":       [480, 365, 660, 135],
            "national_id":   [500, 575, 650, 90],
            # This crop targets the security hologram/stamp area beneath
            # the photo, which embeds a faint watermark-style date - low
            # OCR reliability is expected here regardless of box precision;
            # date_of_birth is authoritatively derived from the checksum-
            # validated national_id instead (see pipeline.py) and this is
            # only a fallback. Widened after a real card's crop came back
            # visibly truncated - this box was never grid-verified like the
            # others (only roughly estimated), so it may still need another
            # pass.
            # Shifted right (was x=5, flush against the card's left edge) -
            # real-card feedback: that left portion was capturing blank
            # margin/edge shadow before the actual embossed digits start.
            # Widened further to the right after real-card feedback that
            # the date text still ran past the box's right edge - capped
            # at x=480 (national_id's box starts at x=500) to leave a
            # margin rather than bleed into it.
            "birth_date":    [45, 540, 435, 110],
            # Real card serials are 2 Latin letters + digits (e.g.
            # "AB1234567"), confirmed against a real card photo - box
            # sized for that pattern, not the Arabic-Indic-digit-only
            # assumption this used to have. Widened slightly after a real
            # card's last digit came back clipped. Widened again (375 ->
            # 430) after that recurred on two separate real scans, which
            # both returned the serial exactly one character short at the
            # end - this is Latin, left-to-right text, so a missing final
            # character means the box's RIGHT edge is cutting it off.
            "serial_number": [4, 685, 430, 55],
        }

        # Field boxes for the back side, in the same rectified 1200x750
        # space as FRONT_FIELDS. Derived directly from the generator's own
        # ground-truth draw boxes (scripts/training/generate_trial_ids.py's
        # BACK_FIELDS anchors, scaled by the same 1200/2560, 750/1614
        # factor align_card uses) rather than guessed/row-projected - the
        # back's rows sit close enough together (issue_date/national_id and
        # profession are only ~10px apart unpadded) that a rough estimate
        # produced boxes overlapping into neighboring rows. Padding here is
        # sized from the actual measured max text width per field, not a
        # round guess.
        #
        # issue_date/profession sizes reflect real-card formats confirmed
        # against a real card photo: issue_date is YYYY/MM (6 digits, not
        # the 8-digit DD/MM/YYYY originally assumed - box narrowed
        # accordingly), and profession can be a full degree/qualification
        # phrase (e.g. "بكالوريوس فى علوم الحاسب"), not a single job word -
        # box widened to the true widest such phrase.
        # Re-derived from a real back-card photo via scripts/debug_field_
        # grid.py (same grid-overlay methodology used for FRONT_FIELDS) -
        # the previous values above were sized from the generator's own
        # anchors, not measured against a real card, and were badly off
        # in practice (national_id/issue_date/profession were all several
        # times narrower than the real printed text, cutting most of it
        # off; marital_status/religion/gender were each shifted left of
        # where the real values sit).
        # NOTE ON VERTICAL SLACK (tried and reverted): align_card does
        # not place card content identically across scans of the same
        # card - each scanner filter changes which pixels read as the
        # card edge, so the warp lands the printed rows slightly higher
        # or lower. On a real "eco" scan the national_id text sat at
        # y~50-85 against a box starting at y=60, so the crop caught only
        # the bottom half of every digit - which fails deceptively, since
        # all 14 glyphs still segment (correct digit COUNT) and simply
        # classify wrong. Extending the top edge to compensate was
        # measured on four real scans and made things WORSE: the
        # previously-perfect "enhanced" scan lost its national_id, while
        # greyscale and eco did not improve, because the added margin
        # pulls the card's own edge into the crop as spurious contours.
        # The drift needs fixing at the alignment stage, not by padding
        # boxes that are correctly placed for a correctly-aligned card.
        self.BACK_FIELDS = {
            # Shifted left and widened - real-card feedback: this box's
            # left edge was clipping the year's leading "20" digits.
            # Left edge pulled back 30px (was x=340). On a real greyscale
            # scan the crop visibly cut the leading '٢' off "٢٠٢٤/٠٨",
            # leaving a 5-digit number the classifier then read as a
            # DIFFERENT but entirely legal year - so no validation rule
            # could catch it and no offset retry fired, because two
            # binarization methods agreed on the clipped reading. Extends
            # left only: the card is blank to the left of here, while
            # national_id's row begins immediately to the right.
            "issue_date":     [310, 55, 225, 55],
            "national_id":    [540, 60, 430, 45],
            # Height extended downward - real-card feedback: the phrase's
            # descenders/second line were being clipped at the bottom.
            "profession":     [395, 105, 580, 70],
            "marital_status": [415, 232, 135, 55],
            "religion":       [665, 232, 135, 55],
            "gender":         [870, 232, 115, 55],
            # Shifted left substantially - real-card feedback: this box
            # was positioned too far right, bleeding into "حتى" (until)
            # while missing the date's own year prefix entirely. Right
            # edge narrowed by 30px after diagnostic segment data showed
            # a stray blob ~12px inside the old right edge (still-visible
            # bleed from "حتى"), sitting past every genuine digit segment.
            # Same left-edge clipping as issue_date above, same fix: the
            # crop was cutting the leading '٢' off "٢٠٣١/٠٨/١٢". Blank
            # card to the left; "البطاقة سارية حتى" sits to the right, so
            # the right edge stays put.
            "expiry_date":    [365, 335, 290, 65],
        }

    @staticmethod
    def _is_card_shaped(w: float, h: float) -> bool:
        if w <= 0 or h <= 0:
            return False
        ratio = max(w, h) / min(w, h)
        return abs(ratio - CARD_ASPECT_RATIO) <= CARD_ASPECT_TOLERANCE

    @staticmethod
    def _is_tightly_cropped_to_card(w: float, h: float) -> bool:
        """Stricter than _is_card_shaped - see FRAME_ASPECT_TOLERANCE for
        why align_card's whole-frame shortcut needs its own tolerance
        instead of reusing the one meant for candidate shapes found
        WITHIN a photo."""
        if w <= 0 or h <= 0:
            return False
        ratio = max(w, h) / min(w, h)
        return abs(ratio - CARD_ASPECT_RATIO) <= FRAME_ASPECT_TOLERANCE

    def get_card_contour(self, image: np.ndarray) -> Optional[np.ndarray]:
        """Finds the largest rectangular-ish contour in the image.

        Rewritten after real-world testing (both a raw phone-camera drag-
        and-drop photo AND a laptop webcam capture that visually filled
        the frame almost perfectly) showed the previous approach silently
        failing on BOTH: requiring approxPolyDP to collapse a contour to
        EXACTLY 4 points is fragile in practice - JPEG compression noise,
        a card's rounded corners, and glare off the holographic security
        strip routinely produce 5-15 points instead of a clean quad, so a
        perfectly well-framed card was still being rejected and silently
        downgraded to a naive full-frame resize (no perspective
        correction at all) that then fed the fixed-coordinate field boxes
        completely wrong crops - garbling national_id/birth_date while
        looking like a normal, if low-confidence, result instead of an
        honest failure.

        Fix: use cv2.minAreaRect's rotated bounding box directly (always
        exactly 4 points by construction, immune to polygon-approximation
        noise) as the candidate shape instead of approxPolyDP's point
        count, gated by the two checks that actually matter - is it big
        enough to plausibly be the card, and is its aspect ratio card-
        shaped.

        Tries three independent thresholding strategies (Otsu, adaptive,
        Canny+dilate) - but as a strict fallback CHAIN, not a pooled "best
        of all three": only moves to the next method if the current one
        finds NO valid candidate at all. Pooling them (taking whichever
        method's candidate had the largest area) was tried and reverted -
        confirmed on a real card photo where the true card-vs-background
        edge had too little brightness contrast for Otsu to trace, but
        Canny+heavy-dilation, given equal footing, latched onto the
        card's own security-hologram texture (a smaller, high-contrast
        region whose bounding box happened to also pass the aspect-ratio
        check) and won on raw area - a confidently WRONG crop, worse than
        an honest failure. Otsu (global brightness split) is the most
        reliable signal when it works at all, so it goes first and is
        trusted alone if it finds anything; adaptive (handles uneven
        lighting) and Canny (handles similar-brightness card/background)
        are reserved for when Otsu comes back completely empty, not
        invited to outbid a valid Otsu result.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        image_area = image.shape[0] * image.shape[1]
        close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))

        def masks():
            _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            yield otsu
            # Uneven lighting (a shadow across half the card) can defeat a
            # single global Otsu cutoff - a local, per-neighborhood cutoff
            # handles that case Otsu can't.
            yield cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 35, 10
            )
            # Edge-based detection catches the case where the card and
            # background are similar brightness (so neither threshold
            # separates them) but still have a visible boundary line.
            edges = cv2.Canny(gray, 50, 150)
            yield cv2.dilate(edges, np.ones((5, 5), np.uint8), iterations=2)

        for mask in masks():
            closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel, iterations=2)
            contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            best_box = None
            best_area = 0.0
            for c in sorted(contours, key=cv2.contourArea, reverse=True)[:5]:
                area = cv2.contourArea(c)
                # Upper bound: a contour spanning nearly the whole frame
                # is virtually always the BACKGROUND being classified as
                # the threshold's "foreground" class (it touches all 4
                # image edges, so cv2.findContours' outermost boundary IS
                # the image border) - confirmed on a stress test where
                # this was silently winning as "the card" every time on a
                # zoomed-out synthetic photo, because the image's own
                # frame aspect ratio coincidentally fell inside the
                # CARD_ASPECT_TOLERANCE band below. An image already this
                # close to edge-to-edge with the card is align_card's
                # whole-frame-shortcut's job (a much tighter aspect check,
                # see FRAME_ASPECT_TOLERANCE), not this interior search's.
                if area > image_area * 0.9:
                    continue
                # Lowered from 0.1 - a "zoomed out" real-world photo (the
                # card occupying a smaller fraction of the frame) is a
                # documented condition this pipeline needs to handle, not
                # just the well-framed case. Safe to lower because the
                # aspect-ratio check below is the real discriminator here:
                # a random small blob passing a tight card-aspect-ratio
                # tolerance by chance is rare, so this isn't just opening
                # the door to noise the way dropping the aspect check
                # would.
                if area < image_area * 0.03 or area <= best_area:
                    continue
                (_, (rw, rh), _) = cv2.minAreaRect(c)
                if min(rw, rh) <= 0 or not self._is_card_shaped(rw, rh):
                    continue
                best_area = area
                best_box = cv2.minAreaRect(c)

            if best_box is not None:
                return cv2.boxPoints(best_box).astype(np.intp)

        # No plausible card-sized, card-shaped region found in any of the
        # three attempts - the caller's fallback (treat the whole image as
        # the card, no warping) is more honest than forcing a bad warp,
        # but see align_card/pipeline.py for how that fallback is now
        # flagged rather than trusted silently.
        return None

    def align_card(self, image: np.ndarray, contour: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Warps the image to the standard 1200x750 canvas using perspective
        transform. Returns (warped_image, M) - M is None only when the
        input is BOTH missing a detectable card boundary AND not already
        card-shaped itself (see the aspect-ratio check below) - i.e. a
        genuine "we don't know where the card is" failure. Callers use
        `M is not None` as a real confidence signal (see pipeline.py's
        card_aligned) so this distinction matters: don't return None for
        an input that's actually fine.
        """
        if contour is None:
            # Check the whole FRAME's own aspect ratio before ever
            # attempting interior contour search. This check costs
            # nothing and, critically, can't be fooled by low contrast or
            # internal texture the way brightness/edge-based thresholding
            # can - confirmed against a real photo where the true card-vs-
            # background edge had too little contrast for Otsu to trace,
            # so get_card_contour instead locked onto the card's own
            # security-hologram texture (a smaller, high-contrast region
            # that happened to also pass the aspect-ratio check) and
            # returned a confidently WRONG crop instead of an honest
            # failure. If the frame is already close to card-shaped, it's
            # virtually certain to already BE (approximately) just the
            # card - true for the guided-crop camera capture (deliberately
            # captures only the on-screen outline's contents) and equally
            # true for a real photo taken tightly around the card - so
            # skip the fragile interior search entirely rather than risk
            # it misfiring on exactly the inputs it's most likely to.
            # Interior contour search is reserved for frames whose own
            # shape clearly indicates real background/margin is present.
            h, w = image.shape[:2]
            if self._is_tightly_cropped_to_card(w, h):
                logger.debug(
                    "Frame's own aspect ratio already matches a card - "
                    "treating the whole frame as the card without "
                    "attempting interior contour search."
                )
                src = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype="float32")
                dst = np.array([
                    [0, 0], [self.target_width - 1, 0],
                    [self.target_width - 1, self.target_height - 1], [0, self.target_height - 1],
                ], dtype="float32")
                M = cv2.getPerspectiveTransform(src, dst)
                warped = cv2.warpPerspective(image, M, (self.target_width, self.target_height))
                return warped, M

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

    def get_field_crops(self, aligned_image: np.ndarray, fields: Optional[Dict[str, List[int]]] = None) -> Dict[str, np.ndarray]:
        """
        Crops each field using fixed, calibrated boxes (self.FRONT_FIELDS by
        default; pass `fields=self.BACK_FIELDS` for the back side - see
        get_back_field_crops).

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
        final_boxes = fields if fields is not None else self.FRONT_FIELDS

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

    def get_back_field_crops(self, aligned_image: np.ndarray) -> Dict[str, np.ndarray]:
        """Same as get_field_crops, using the back-side field boxes."""
        return self.get_field_crops(aligned_image, fields=self.BACK_FIELDS)

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
