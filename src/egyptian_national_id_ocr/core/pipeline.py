import numpy as np
import logging
import os
import time
import cv2
import base64
from typing import Optional, List, Dict
from concurrent.futures import ThreadPoolExecutor

from ..detection.card_detector import CardDetector
from ..detection.side_classifier import SideClassifier, IDSide
from ..ocr.base import OCREngine, OCRPreprocessor
from ..ocr.digit_classifier_engine import DigitClassifierEngine
from ..postprocessing.numeral_converter import normalize_arabic_numerals, clean_ocr_text
from ..postprocessing.national_id_parser import decode_national_id, find_nid_in_text
from ..models.id_card import IDCard, IDCardFront, IDCardBack
from ..models.enums import ExtractionStatus, Gender, Religion, MaritalStatus
from ..postprocessing.enum_matcher import match_enum, GENDER_VALUES, RELIGION_VALUES, MARITAL_STATUS_VALUES
from .layout_analyzer import LayoutAnalyzer
from .field_clusterer import FieldClusterer

logger = logging.getLogger(__name__)

# Opt-in geometry trace for the free-text bucketing step (set
# EGY_NID_DEBUG_BUCKETS=1). Logs each OCR detection's box, centroid and
# CHARACTER COUNT - never the recognised text itself, so a real card can
# be diagnosed without its contents reaching a log file.
_BUCKET_DEBUG = os.environ.get("EGY_NID_DEBUG_BUCKETS") == "1"

class Pipeline:
    def __init__(self, ocr_engine: Optional[OCREngine] = None, tesseract_cmd: str = "tesseract"):
        self.detector = CardDetector()
        self.classifier = SideClassifier()
        if ocr_engine is None:
            # Both cli.py and api/server.py constructed Pipeline() without
            # ever passing an ocr_engine, leaving this None - the first
            # non-numeric field would crash with AttributeError. Default to
            # PaddleOCR (see ocr/paddle_ocr_engine.py): faster and more
            # accurate than EasyOCR for this use case, and this makes the
            # actual production entry points work at all.
            from ..ocr.paddle_ocr_engine import PaddleOCREngine
            ocr_engine = PaddleOCREngine()
        self.ocr_engine = ocr_engine
        self.preprocessor = OCRPreprocessor()
        # Numeric fields (national_id, birth_date, serial_number) use a
        # dedicated lightweight digit classifier instead of the general
        # OCR engine - see ocr/digit_classifier_engine.py for why.
        self.digit_engine = DigitClassifierEngine()

    # Tried in this order because align_card's contour-based detection can
    # itself be orientation-sensitive (a portrait photo of a landscape card
    # rarely produces a plausible card-shaped contour), so we can't rely on
    # it alone to notice the image is sideways.
    ROTATIONS_TO_TRY = [
        (None, "0°"),
        (cv2.ROTATE_90_CLOCKWISE, "90°"),
        (cv2.ROTATE_180, "180°"),
        (cv2.ROTATE_90_COUNTERCLOCKWISE, "270°"),
    ]

    # Mean per-field score above which an orientation is taken as
    # confirmed correct, ending the rotation search early - see
    # orientation_confirmed in _process_single_orientation for the
    # measurements behind this value.
    ORIENTATION_CONFIRMED_CONFIDENCE = 0.6

    def process_image(self, image: np.ndarray) -> IDCard:
        """
        Input contract: expects a well-framed card image - the card
        filling (or nearly filling) the frame, as a guided-crop UI would
        produce (user aligns the card to an on-screen outline before
        submitting), NOT an unconstrained photo of a card sitting on an
        arbitrary background. This is a deliberate scope decision, not an
        oversight: align_card's contour-based detection for the latter
        case has a known bug (see scripts/training/generate_realistic_
        photos.py's docstring) and free-form document detection is a
        substantially harder, separate problem. Accuracy figures (98%+ on
        data/synthetic_front/) only hold for well-framed input.

        Tries the image at 0/90/180/270 degree rotations (see
        ROTATIONS_TO_TRY) and returns the best result - this covers "the
        card is framed correctly but the photo itself is sideways/upside
        down" (e.g. phone held in a different orientation), which a
        guided-crop UI does not otherwise prevent. A checksum-valid
        National ID is a hard correctness signal, so the first rotation
        that produces one is returned immediately without trying the rest -
        the common case (already-upright photo) stays a single fast pass.
        Otherwise, the highest-confidence attempt across all four is
        returned so there's always *some* result with an honest status/
        message, rather than silently only ever trying one orientation.
        """
        overall_start = time.time()

        if image is None or image.size == 0:
            result = IDCard()
            result.status = ExtractionStatus.UNREADABLE_IMAGE
            result.messages = [
                "This file couldn't be read as an image. Please upload a "
                "clear JPG or PNG photo of the ID card."
            ]
            result.processing_time_ms = int((time.time() - overall_start) * 1000)
            return result

        best_result: Optional[IDCard] = None
        for rotation, label in self.ROTATIONS_TO_TRY:
            attempt_image = cv2.rotate(image, rotation) if rotation is not None else image
            result, orientation_confirmed = self._process_single_orientation(attempt_image)
            logger.debug(f"Orientation {label}: status={result.status} confidence={result.confidence:.2f}")

            if result.decoded is not None:
                result.processing_time_ms = int((time.time() - overall_start) * 1000)
                return result

            if best_result is None or result.confidence > best_result.confidence:
                best_result = result

            if orientation_confirmed:
                # This orientation is demonstrably the right way up (see
                # _process_single_orientation for the signal), so the
                # remaining rotations cannot beat it - they would just
                # re-run the whole pipeline, including PaddleOCR, against
                # field boxes that land on nothing. Only the cases that
                # FAIL to verify a National ID reached here at all, which
                # is exactly when the 4x cost was being paid: real scans
                # that failed took 28-35s versus ~5s for ones that
                # verified on the first attempt.
                break

        best_result.processing_time_ms = int((time.time() - overall_start) * 1000)
        return best_result

    def _process_single_orientation(self, image: np.ndarray) -> tuple[IDCard, bool]:
        """Returns (result, orientation_confirmed) - see the
        orientation_confirmed assignment below for what the flag means and
        why process_image uses it to skip the remaining rotations."""
        start_time = time.time()
        result = IDCard()
        field_confidences: List[float] = []
        # True only when the National ID was read cleanly, or was repaired
        # AND independently confirmed by another field on the card. A
        # repair with nothing to confirm it is still surfaced, but must
        # never be presented as verified - see repair_nid_detailed.
        nid_trusted = False

        try:
            # 1. Detect & Align Card (Warp to 1200x750)
            analyzer = LayoutAnalyzer()
            rectified, M = analyzer.align_card(image)
            # M is None exactly when get_card_contour found no plausible
            # card boundary and align_card fell back to a naive full-frame
            # resize (see its docstring) - the FRONT_FIELDS/BACK_FIELDS
            # fixed pixel coordinates are calibrated against a properly
            # perspective-warped 1200x750 card, so a fallback resize means
            # every field box is cutting from an unverified, likely wrong
            # position. Confirmed on real photos (both a wide phone shot
            # and a laptop webcam capture that visually filled the frame)
            # that this fallback still produced plausible-looking-but-
            # wrong digit strings instead of an honest low-confidence
            # result - card_aligned lets confidence scoring and the status
            # message below tell the user the truth about what happened,
            # instead of presenting a fixed-box extraction as if it came
            # from a verified crop.
            card_aligned = M is not None
            _, card_buffer = cv2.imencode('.jpg', rectified)
            result.card_image = f"data:image/jpeg;base64,{base64.b64encode(card_buffer).decode('utf-8')}"

            # 1b. Preprocessing normalization phase - see _normalize_card.
            # Runs once, globally, on the whole aligned card, BEFORE any
            # field-level cutting or OCR - a black-and-white photocopy, a
            # washed-out phone photo, and a well-lit color photo should
            # all reach the field-cutting stage at a comparable contrast
            # level, instead of every downstream OCR/digit call separately
            # guessing how much enhancement its own small crop needs (the
            # ad hoc per-field CLAHE fallbacks elsewhere in this file are
            # symptoms of that gap). `rectified` itself is left untouched
            # for the UI's "card we detected" preview, which should show
            # the real photo, not a contrast-boosted one.
            normalized = self._normalize_card(rectified)

            # 2. Classify Side
            side = self.classifier.classify(normalized)
            result.side = side.value

            refined_data: Dict[str, str] = {}

            if side == IDSide.FRONT:
                # 3. Get Fixed-Coordinate Crops (still needed for numeric
                # fields' digit classifier and for the UI's per-field crop
                # previews - see step 4 for why free-text fields don't use
                # these crops for OCR itself any more)
                # Crops come from the UN-normalized card: measured 10/10 vs
                # 6/10 correct national_id readings on the same 10 cards
                # with and without _normalize_card applied first. CLAHE
                # amplifies the card's background watermark right where the
                # digits sit, and the per-crop binarization ladder in
                # DigitClassifierEngine already adapts contrast far better
                # than one global pre-pass can. The normalized image is
                # still used for free-text OCR below.
                field_crops = analyzer.get_field_crops(rectified)

                # 4. OCR
                refined_data, refined_crops = self._process_fields(
                    rectified, field_crops, analyzer.FRONT_FIELDS, field_confidences,
                    alt_field_crops=analyzer.get_field_crops(normalized),
                    crop_source=rectified,
                )

                # 5. Populate Data
                result.front = self._populate_front_v4(refined_data, refined_crops)

                # 6. Checksum-based Repair & Decoding, cross-checked
                # against the date of birth printed on the card. That
                # OCR'd date is an INDEPENDENT reading of the same facts
                # the NID encodes, which is what makes repairing the NID
                # safe at all - see repair_nid_detailed for the measured
                # difference (at one OCR error: 28%->55% recovered and
                # 9.8%->0% wrong). Passed only when it parses as a real
                # date, so a garbled read can't corrupt the check.
                from ..postprocessing.national_id_parser import repair_nid_detailed, decode_national_id
                ocr_birth_date = self._ocr_date_to_iso(result.front.date_of_birth)
                repair = repair_nid_detailed(
                    result.front.national_id, expected_birth_date=ocr_birth_date
                )
                if repair.nid:
                    result.front.national_id = repair.nid
                    result.decoded = decode_national_id(repair.nid)
                    nid_trusted = not repair.repaired or repair.corroborated
                    if result.decoded:
                        # The checksum-validated number is authoritative
                        # for the date once it passes; the OCR'd value
                        # already did its job as the cross-check above.
                        result.front.date_of_birth = result.decoded.birth_date
                elif ocr_birth_date is None:
                    # Neither the NID nor the date verified, and the date
                    # we read isn't a possible birth date (real scans
                    # produced "4136/11/51" and "3001/11/05"). Showing it
                    # anyway just presents noise as data.
                    result.front.date_of_birth = ""

            elif side == IDSide.BACK:
                # ALIGNMENT DRIFT (investigated, not yet solved - do not
                # retry the obvious fix without reading this):
                # align_card lands the same physical card at a different
                # offset AND scale depending on the scanner filter used,
                # because each filter changes which pixels read as the
                # card's edge. Measured across four real scans of one
                # card, relative to the scan BACK_FIELDS was calibrated
                # on: offsets up to 28px and a horizontal scale error up
                # to 6%. Scale error is what predicts failure - a scan
                # 28px out still read perfectly, while the 5-6% stretched
                # ones failed, because a stretch displaces the
                # national_id row by 30-50px.
                #
                # Pinning the card to a canonical frame via its barcode
                # (a big, unmistakable landmark) was implemented and
                # REVERTED: it made things worse, turning a
                # previously-perfect scan into an all-fields-wrong one.
                # The reason is the useful part - the barcode's
                # displacement does NOT predict the text rows'
                # displacement, so the distortion is not uniform across
                # the card and no single global affine can correct it.
                # A fix has to be per-field/local, not one transform for
                # the whole card.

                # 3. Get Fixed-Coordinate Crops
                # Un-normalized, for the same measured reason as the front.
                field_crops = analyzer.get_back_field_crops(rectified)

                # 4. OCR
                refined_data, refined_crops = self._process_fields(
                    rectified, field_crops, analyzer.BACK_FIELDS, field_confidences,
                    alt_field_crops=analyzer.get_back_field_crops(normalized),
                    crop_source=rectified,
                )

                # 5. Populate Data
                result.back = self._populate_back(refined_data, refined_crops)

                # 6. Checksum-based Repair & Decoding (national_id is
                # repeated on the back, same validation as the front).
                # The back has no printed date of birth to cross-check
                # against, but it does print gender - a weaker signal
                # than a full date (one bit, so it only rejects about
                # half of wrong candidates) but an independent one.
                from ..postprocessing.national_id_parser import repair_nid_detailed, decode_national_id
                expected_gender = (
                    result.back.gender if result.back.gender != Gender.UNKNOWN else None
                )
                repair = repair_nid_detailed(
                    result.back.national_id, expected_gender=expected_gender
                )
                if repair.nid:
                    result.back.national_id = repair.nid
                    result.decoded = decode_national_id(repair.nid)
                    nid_trusted = not repair.repaired or repair.corroborated

                # 7a-pre. The two dates are each other's only cross-check
                # (neither carries a checksum), so where they disagree,
                # try to recover the issue year from the expiry year -
                # the mirror of the expiry repair below.
                result.back.issue_date = self._repair_issue_year_from_expiry(
                    result.back.issue_date, result.back.expiry_date
                )

                # 7a. Discard an issue date that could not belong to a
                # real card - issued in the future, or longer ago than a
                # card stays valid (see _issue_date_is_plausible). Runs
                # BEFORE the expiry cross-check below, which uses the
                # issue date as its reference: checking expiry against a
                # date we already know is impossible would only spread
                # the error.
                issue_digits = "".join(
                    ch for ch in (result.back.issue_date or "") if ch.isdigit()
                )
                if issue_digits and not self._issue_date_is_plausible(issue_digits):
                    result.back.issue_date = ""

                # 7b. Cross-check expiry_date's year against issue_date - an
                # Egyptian adult National ID's validity is always exactly
                # 7 years (confirmed by the user; also already assumed by
                # this project's own synthetic generator). expiry_date's
                # year comes from the same digit classifier that
                # misreads '3' as '2' elsewhere (confirmed on a real
                # card: expiry decoded to a year BEFORE its own issue
                # date, which is never valid) - a second, independent
                # source for what the year should be is a stronger
                # signal than either OCR read alone.
                result.back.expiry_date = self._repair_expiry_year(
                    result.back.issue_date, result.back.expiry_date
                )

                # 7c. Is the card expired as of today? Compared on the
                # DAY, not the month: a card issued 03/2019 expires
                # 03/2026, so scanning it in 04/2026 means expired, while
                # scanning it during 03/2026 depends on which day of that
                # month the expiry falls on. Left as None when expiry
                # could not be read or validated - "we don't know" must
                # stay distinguishable from "not expired".
                result.back.is_expired = self._card_is_expired(result.back.expiry_date)
                if result.back.is_expired:
                    result.messages.append(
                        f"This card expired on {result.back.expiry_date}."
                    )

            # Confidence reflects what we actually measured: mean per-field
            # score, gated by whether the NID passed its Mod-11 checksum (a
            # hard correctness signal that should dominate a soft OCR score -
            # a checksum failure means we *know* something is wrong, no
            # matter how confident the OCR engine was on individual glyphs)
            # and by whether the card was actually aligned via a detected
            # contour rather than a naive fallback resize (see card_aligned
            # above).
            ocr_confidence = float(np.mean(field_confidences)) if field_confidences else 0.0

            # Whether this rotation is demonstrably the right way up, which
            # lets process_image skip the remaining rotations. The side
            # classifier can't answer this (it deliberately scores
            # max(upright, 180-rotated), so it is orientation-agnostic by
            # design), and simply counting FILLED fields isn't enough
            # either - measured on a sideways card, the wrong orientation
            # still filled 4 of 6 fields with junk, which would have ended
            # the search on a wrong answer.
            #
            # The mean field score separates them cleanly: on that same
            # card the three wrong orientations scored 0.14, 0.26 and 0.18
            # while the correct one scored 0.91. The threshold sits well
            # above the junk cluster and below a genuine partial success,
            # so a real card that simply fails to verify its NID can still
            # exit early rather than paying 4x for rotations that cannot
            # help it.
            filled = sum(1 for value in refined_data.values() if value and value.strip())
            orientation_confirmed = bool(
                card_aligned
                and side != IDSide.UNKNOWN
                and refined_data
                and filled >= max(2, (len(refined_data) + 1) // 2)
                and ocr_confidence >= self.ORIENTATION_CONFIRMED_CONFIDENCE
            )
            if result.decoded and nid_trusted:
                result.confidence = ocr_confidence
            elif result.decoded:
                # Reconstructed to satisfy the checksum, with no second
                # field agreeing it's right. Measured false-repair rates
                # make this materially less reliable than a clean read
                # (see repair_nid_detailed), so it must not inherit a
                # clean read's confidence.
                result.confidence = min(ocr_confidence, 0.5)
            elif not card_aligned:
                result.confidence = min(ocr_confidence, 0.1)
            else:
                result.confidence = min(ocr_confidence, 0.3)

            # Plain-language status for a UI to key off of directly, instead
            # of inferring quality from confidence thresholds or null-
            # checking fields itself.
            if result.decoded is not None and nid_trusted:
                result.status = ExtractionStatus.SUCCESS
                result.messages = ["National ID extracted and verified successfully."]
            elif result.decoded is not None:
                result.status = ExtractionStatus.LOW_CONFIDENCE
                result.messages = [
                    "We had to correct a digit in the National ID to make "
                    "it valid, and nothing else on the card confirms the "
                    "correction. The number below may not be right - "
                    "please check it against the card."
                ]
            elif not card_aligned:
                result.status = ExtractionStatus.LOW_CONFIDENCE
                result.messages = [
                    "We couldn't confidently detect the card's edges in this "
                    "photo, so the extracted fields below are likely "
                    "misaligned and unreliable. Try a photo with more "
                    "contrast between the card and its background, better, "
                    "even lighting, and the card held flat and filling most "
                    "of the frame."
                ]
            elif result.front is not None or result.back is not None:
                result.status = ExtractionStatus.LOW_CONFIDENCE
                result.messages = [
                    "We found a card but couldn't fully verify the National "
                    "ID number. Please check the extracted data carefully, "
                    "or try again with a clearer, well-lit photo."
                ]
            else:
                result.status = ExtractionStatus.NO_CARD_DETECTED
                result.messages = [
                    "We couldn't find an ID card in this image. Make sure "
                    "the whole card is visible, in focus, and well-lit, "
                    "then try again."
                ]

        except Exception as e:
            logger.error(f"Pipeline processing failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            result.status = ExtractionStatus.NO_CARD_DETECTED
            result.messages = [
                "Something went wrong while processing this image. Please "
                "try again with a different photo."
            ]
            orientation_confirmed = False

        result.processing_time_ms = int((time.time() - start_time) * 1000)
        return result, orientation_confirmed

    # Back's issue_date/expiry_date are digit-plus-separator strings, same
    # shape as front's birth_date - the digit classifier already handles
    # that pattern (see DigitClassifierEngine._segment_digits).
    #
    # serial_number is deliberately NOT here even though it looks numeric:
    # real card serials are 2 Latin letters + digits (e.g. "AB1234567",
    # confirmed against a real card photo), not Arabic-Indic digits - the
    # digit classifier only recognizes Arabic-Indic glyph shapes 0-9, so it
    # would never read this field correctly. It goes through the general
    # OCR engine instead, like the other free-text fields.
    NUMERIC_FIELDS = {"national_id", "birth_date", "issue_date", "expiry_date"}

    # Fixed digit counts (separators excluded) for fields whose format is
    # known - lets DigitClassifierEngine drop a segmented "/" separator
    # (see extract_digits' expected_length) instead of misclassifying it
    # as a digit. national_id has no separator, so this is a no-op for it,
    # but passing it anyway costs nothing and stays correct if that ever
    # changes.
    NUMERIC_FIELD_LENGTHS = {
        "national_id": 14,
        "birth_date": 8,    # YYYY/MM/DD
        "issue_date": 6,    # YYYY/MM
        "expiry_date": 8,   # YYYY/MM/DD
    }

    # 0-indexed position(s) of each "/" within the raw digits+separators
    # sequence, for fields with a fixed date format - see
    # DigitClassifierEngine.extract_digits' separator_positions for why
    # this is more reliable than guessing which segment is the separator.
    NUMERIC_FIELD_SEPARATORS = {
        "birth_date": [4, 7],    # YYYY/MM/DD
        "issue_date": [4],       # YYYY/MM
        "expiry_date": [4, 7],   # YYYY/MM/DD
    }

    # Where to re-insert "/" for display, as positions within the CLEAN
    # digit string (after separators are already dropped) - not the same
    # numbers as NUMERIC_FIELD_SEPARATORS above, which count the
    # separator itself as part of the raw sequence. E.g. "202408" gets a
    # "/" inserted at index 4 -> "2024/08".
    NUMERIC_FIELD_DISPLAY_SEPARATORS = {
        "birth_date": [4, 6],    # YYYY/MM/DD
        "issue_date": [4],       # YYYY/MM
        "expiry_date": [4, 6],   # YYYY/MM/DD
    }

    # How far (in the 1200x750 rectified space) two field boxes' edges can
    # be from each other and still get merged into one OCR region by
    # _cluster_field_boxes. Large enough to pull genuinely nearby fields
    # (e.g. first_name/full_name/address, ~50-100px apart) into a shared
    # region for detection context; small enough not to merge unrelated
    # fields on opposite sides of the card into one slow, unnecessarily
    # large region.
    CLUSTER_PADDING = 80

    # Upscale factor for the free-text rescan pass (1 disables it).
    FREE_TEXT_RESCAN_SCALE = 3

    # Field pairs that must never end up in the same OCR cluster no matter
    # how much CLUSTER_PADDING overlaps their boxes - see
    # _cluster_field_boxes' never_merge parameter for why this exists
    # instead of a smaller shared padding value. Confirmed on a real back
    # card: profession's row and the marital/religion/gender row below it
    # are only ~57px apart, well inside CLUSTER_PADDING's reach, so they
    # merged into one large, noisy region where profession's own text
    # wasn't detected at all - but simply shrinking the padding to keep
    # them apart cost marital_status/gender's own detection reliability
    # too (confirmed: a detection dropped from 4-5 characters to 2).
    # Excluding the specific pairs keeps CLUSTER_PADDING generous for
    # everyone while still guaranteeing these two rows are never merged.
    NEVER_MERGE_PAIRS = {
        frozenset({"profession", "marital_status"}),
        frozenset({"profession", "religion"}),
        frozenset({"profession", "gender"}),
    }

    # Extra padding (in the same space) given to a field that clusters
    # with no one - larger than CLUSTER_PADDING since it has to make up
    # for the missing neighbor's own text/whitespace alone, not just
    # bridge a gap between two fields' text.
    SOLO_CLUSTER_PADDING = 180

    # Vertical counterpart to SOLO_CLUSTER_PADDING, kept much smaller for
    # the same reason as CLUSTER_PADDING_Y - a solo field expanding the
    # full 180px vertically would still reach into an unrelated row just
    # 57px away (profession's actual situation), pulling its text into
    # the OCR call even though bucketing would correctly discard it -
    # wasted noise the detector has to work around for nothing.
    SOLO_CLUSTER_PADDING_Y = 40

    def _cluster_field_boxes(
        self,
        field_boxes: Dict[str, list],
        pad_x: int,
        canvas_w: int,
        canvas_h: int,
        pad_y: Optional[int] = None,
        never_merge: Optional[set] = None,
    ) -> List[tuple]:
        """
        Groups field boxes into clusters of mutually-overlapping (once
        padded by `pad_x` horizontally and `pad_y` vertically) regions, so
        OCR can run once per cluster instead of once per field or once for
        the whole card - see _process_fields for why. `pad_y` defaults to
        `pad_x` if not given.

        never_merge: an optional set of frozenset({field_a, field_b})
        pairs that should never end up in the same cluster no matter how
        much their padded boxes overlap. Needed because generous padding
        (which some small, isolated fields genuinely need for reliable
        detection - confirmed on a real back card: cutting padding to
        keep two unrelated rows apart cost marital_status/gender's own
        detection quality) and "keep two specific rows apart" can be
        directly opposed goals that no single padding value satisfies for
        both at once - explicit exclusion decouples them instead of
        splitting the difference.

        Returns [([x, y, w, h], [field_names]), ...] with each region
        clamped to the canvas.
        """
        import itertools

        if pad_y is None:
            pad_y = pad_x
        never_merge = never_merge or set()

        padded = {
            name: [x - pad_x, y - pad_y, x + w + pad_x, y + h + pad_y]
            for name, (x, y, w, h) in field_boxes.items()
        }

        names = list(field_boxes.keys())
        parent = {name: name for name in names}

        def find(n):
            while parent[n] != n:
                n = parent[n]
            return n

        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb

        def overlaps(b1, b2):
            return not (b1[2] < b2[0] or b2[2] < b1[0] or b1[3] < b2[1] or b2[3] < b1[1])

        for a, b in itertools.combinations(names, 2):
            if frozenset({a, b}) in never_merge:
                continue
            if overlaps(padded[a], padded[b]):
                union(a, b)

        groups: Dict[str, list] = {}
        for name in names:
            groups.setdefault(find(name), []).append(name)

        clusters = []
        for members in groups.values():
            x0 = max(0, min(padded[n][0] for n in members))
            y0 = max(0, min(padded[n][1] for n in members))
            x1 = min(canvas_w, max(padded[n][2] for n in members))
            y1 = min(canvas_h, max(padded[n][3] for n in members))
            if len(members) == 1:
                # A field with no nearby neighbor to cluster with gets
                # none of the "shared region" context that made detection
                # work for everyone else - confirmed on the back side:
                # profession (isolated, no other free-text field within
                # CLUSTER_PADDING of it) came back garbled the same way
                # first_name did before clustering was introduced. Give a
                # solo field extra padding of its own so it isn't
                # penalized for having no neighbors.
                name = members[0]
                fx, fy, fw, fh = field_boxes[name]
                extra_x, extra_y = self.SOLO_CLUSTER_PADDING, self.SOLO_CLUSTER_PADDING_Y
                x0 = max(0, fx - extra_x)
                y0 = max(0, fy - extra_y)
                x1 = min(canvas_w, fx + fw + extra_x)
                y1 = min(canvas_h, fy + fh + extra_y)
            clusters.append(([x0, y0, x1 - x0, y1 - y0], members))
        return clusters

    def _process_fields(
        self,
        rectified: np.ndarray,
        field_crops: Dict[str, np.ndarray],
        field_boxes: Dict[str, list],
        field_confidences: List[float],
        alt_field_crops: Optional[Dict[str, np.ndarray]] = None,
        crop_source: Optional[np.ndarray] = None,
    ) -> tuple[dict, dict]:
        """
        Extracts every field's text plus its base64 crop preview.

        Numeric fields still go through the dedicated digit classifier on
        their own small crop (unaffected by the free-text issue below - it
        does its own segmentation, not detection).

        Free-text fields (names, address, serial number, profession, ...)
        used to each get their own OCR call on a small, individually
        padded crop. That doesn't work: testing found PaddleOCR's detector
        returns *zero* detections on an isolated small crop - even padded
        20-40px or upscaled - for text it detects perfectly well when the
        same region is part of a larger image. The detector evidently
        needs more surrounding context/scale cues than a single tight
        field crop can offer. It also silently fixed a second bug: joining
        several independently-cropped detections was producing scrambled
        word order (e.g. a name's first word ending up last), because each
        crop's internal sort had no relationship to another crop's.

        A single pass over the *whole* card fixed word order (every
        detection gets real, comparable coordinates to sort by) but on a
        real card it wasn't always enough context for the smallest field
        (first_name, a single short word) - and scanning the whole
        1200x750 canvas for detection+recognition is the slow part of the
        pipeline (~13s). So instead: cluster nearby field boxes (padded)
        into merged regions and OCR each region once - e.g. first_name +
        full_name + address (all in the same column) become one region.
        This keeps the "enough surrounding context" property that fixed
        detection, while (a) giving the detector a *smaller* image where
        each field's text is proportionally larger and easier to resolve,
        and (b) skipping the empty photo/background area, both of which
        should help detection and cut processing time versus scanning the
        entire card.
        """
        free_text_boxes = {
            name: box for name, box in field_boxes.items()
            if name not in self.NUMERIC_FIELDS
        }
        free_text_by_field: Dict[str, tuple] = {}
        if free_text_boxes:
            canvas_h, canvas_w = rectified.shape[:2]
            clusters = self._cluster_field_boxes(
                free_text_boxes, self.CLUSTER_PADDING, canvas_w, canvas_h,
                never_merge=self.NEVER_MERGE_PAIRS,
            )
            # Pass 1: read every cluster from the raw rectified card.
            pending = []
            for (cx, cy, cw, ch), member_names in clusters:
                region = rectified[cy:cy + ch, cx:cx + cw]
                if region.size == 0:
                    continue
                results = self.ocr_engine.read_layout(region, mode="auto")
                # Region-local polygon coords -> full-canvas coords, so the
                # centroid-in-box test in _bucket_free_text (which compares
                # against field_boxes, defined in canvas space) still works.
                shifted = [(poly + np.array([cx, cy]), text, score) for poly, text, score in results]
                member_boxes = {name: free_text_boxes[name] for name in member_names}
                bucketed = self._bucket_free_text(shifted, member_boxes)
                free_text_by_field.update(bucketed)
                pending.append(((cx, cy, cw, ch), member_boxes))

            # Pass 2: re-read ONLY the regions where the raw pass found
            # nothing at all, upscaled.
            #
            # On a low-contrast scan the failure is in PaddleOCR's
            # *detector*, not its recogniser. Measured on a real greyscale
            # back, the marital/religion/gender region returned ZERO
            # detections - the text was never proposed as a region, so no
            # post-processing could have recovered it. Re-reading that
            # region at 3x recovers the whole row.
            #
            # Restricted to entirely-empty regions because that is the
            # only case where it measurably pays. Running it on every
            # region was tried against all eight scans of a real card: it
            # fixed nothing extra and took fronts from 4.4s to 12.6s.
            # In particular it does NOT rescue a region that read
            # PARTIALLY - a truncated profession stayed truncated,
            # because by then the limit is the resolution the card was
            # scanned at, which upscaling cannot add back. (That fix
            # appeared to work when tested against an already-rectified
            # card image, where the card fills the frame; on the original
            # zoomed-out photo it does not. Test against the original
            # file, not a rectified intermediate.)
            #
            # Contrast normalization was tried here first and is not what
            # works: on the same scan it returned FEWER words than the raw
            # pass (9 characters against 17).
            #
            # Skipped entirely unless the raw pass found something
            # somewhere on this card. Nothing anywhere is the signature of
            # a wrong rotation rather than a hard-to-read card, and every
            # failed image pays for all four rotations - the same guard
            # the per-field fallback ladder uses below.
            if self.FREE_TEXT_RESCAN_SCALE > 1 and any(
                text.strip() for text, _ in free_text_by_field.values()
            ):
                scale = self.FREE_TEXT_RESCAN_SCALE
                for (cx, cy, cw, ch), member_boxes in pending:
                    if any(
                        free_text_by_field.get(name, ("", 0.0))[0].strip()
                        for name in member_boxes
                    ):
                        continue  # this region read; leave it alone
                    region = rectified[cy:cy + ch, cx:cx + cw]
                    upscaled = cv2.resize(
                        region, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC
                    )
                    alt_results = self.ocr_engine.read_layout(upscaled, mode="auto")
                    alt_shifted = [
                        (poly / scale + np.array([cx, cy]), text, score)
                        for poly, text, score in alt_results
                    ]
                    free_text_by_field.update(
                        self._bucket_free_text(alt_shifted, member_boxes)
                    )

        refined_data: Dict[str, str] = {}
        refined_crops: Dict[str, str] = {}

        for field_name, crop in field_crops.items():
            if crop.size == 0:
                continue

            _, buffer = cv2.imencode('.jpg', crop)
            refined_crops[field_name] = f"data:image/jpeg;base64,{base64.b64encode(buffer).decode('utf-8')}"

            if field_name in self.NUMERIC_FIELDS:
                # Dedicated per-character digit classifier instead of
                # general OCR - see ocr/digit_classifier_engine.py. It does
                # its own preprocessing/segmentation, so pass the raw crop.
                # (Tried padding this crop the same way the free-text path
                # is padded, to stop the outermost digit clipping - made
                # things far worse: replicating this crop's edge pixels
                # outward amplifies background-graphic texture, sometimes
                # into more spurious blobs. Reverted.)
                expected_length = self.NUMERIC_FIELD_LENGTHS.get(field_name)
                separators = self.NUMERIC_FIELD_SEPARATORS.get(field_name)
                text, confidence = self.digit_engine.extract_digits_detailed(
                    crop, expected_length=expected_length, separator_positions=separators
                )
                # Retry on the contrast-normalized copy of the same field
                # when the first read lacks full support. Neither version
                # wins everywhere: reading the un-normalized crop fixed a
                # CamScanner scan (10/10 vs 6/10 on synthetic cards),
                # while a real greyscale scan that had been perfect
                # regressed without normalization. Rather than picking one
                # globally and losing the other, try both and keep the
                # better-supported reading - the same "let the evidence
                # choose" approach the binarization ladder uses. Only runs
                # when the first attempt is already weak, so a clean card
                # pays nothing.
                alt_crop = (alt_field_crops or {}).get(field_name)
                if (
                    alt_crop is not None
                    and alt_crop.size > 0
                    and confidence < self.digit_engine.CONFIDENCE_METHODS_AGREE
                ):
                    alt_text, alt_confidence = self.digit_engine.extract_digits_detailed(
                        alt_crop, expected_length=expected_length, separator_positions=separators
                    )
                    if alt_confidence > confidence:
                        text, confidence = alt_text, alt_confidence

                # Still unsupported? Try cutting this field a little off
                # its nominal position. align_card lands the same physical
                # card at a different offset and scale depending on how it
                # was scanned (measured on real scans: offsets to 28px,
                # horizontal scale error to 6%), which slides digits
                # partially out of a correctly-placed box - they still
                # segment, so the count looks right while every glyph is
                # a fragment that classifies wrong.
                #
                # This is deliberately LOCAL. Two whole-card corrections
                # were tried and reverted (padding every box; pinning the
                # card by its barcode), and both broke scans that already
                # worked, because the distortion is not uniform across the
                # card. A per-field nudge can only ever engage where the
                # field already failed, and is accepted only on the
                # strongest evidence available - two independent
                # binarization methods agreeing on a result of exactly the
                # expected length.
                if crop_source is not None and field_name in self.NUMERIC_OFFSET_RETRY_FIELDS:
                    # Each field gets the strongest acceptance test it can
                    # offer: the national ID validates itself via its
                    # check digit, and expiry_date validates against the
                    # issue date already read from this same card (fields
                    # are processed in box order, so issue_date is
                    # available by the time expiry is).
                    if field_name == "national_id":
                        validator = self._nid_is_self_validating
                    elif field_name == "issue_date":
                        validator = self._issue_date_is_plausible
                    elif field_name == "expiry_date":
                        issue_value = refined_data.get("issue_date", "")
                        validator = (
                            lambda text: self._expiry_agrees_with_issue(text, issue_value)
                        )
                    else:
                        validator = None

                    # Retrying whenever the reading merely FAILS its
                    # validator (even at full method agreement) was tried
                    # and reverted: it roughly doubled processing time on
                    # exactly the scans that were already slowest, and
                    # recovered nothing. That null result is useful - if
                    # no offset yields a valid reading, the digits are
                    # being misread at every position, so the remaining
                    # greyscale/eco date failures are a classification
                    # problem rather than a positioning one.
                    if confidence < self.digit_engine.CONFIDENCE_METHODS_AGREE:
                        box = field_boxes.get(field_name)
                        if box is not None:
                            shifted_text, shifted_confidence = self._retry_numeric_at_offsets(
                                crop_source, box, expected_length, separators, validator=validator
                            )
                            # A validated reading beats an unvalidated one
                            # even when the latter scored higher on method
                            # agreement alone.
                            if shifted_text and validator and validator(shifted_text):
                                text, confidence = shifted_text, shifted_confidence
                            elif shifted_confidence > confidence:
                                text, confidence = shifted_text, shifted_confidence
                # The digit classifier only ever returns bare digits (it
                # has no way to represent "/"), so re-insert it for
                # display - e.g. "202408" -> "2024/08". Insertion points
                # are positions in the clean digit string, right to left
                # so each insertion doesn't shift the position of the
                # next one still to be applied.
                for pos in sorted(self.NUMERIC_FIELD_DISPLAY_SEPARATORS.get(field_name, []), reverse=True):
                    if len(text) > pos:
                        text = text[:pos] + "/" + text[pos:]
                # confidence comes from extract_digits_detailed - how much
                # independent support the reading has (did any config hit
                # the field's known glyph count, did two binarization
                # methods agree). This replaced a flat 0.9 for any
                # non-empty result, which made a garbled read score the
                # same as a clean one.
            else:
                raw_text, confidence = free_text_by_field.get(field_name, ("", 0.0))
                # Only retry a missing field when the clustered pass found
                # SOMETHING somewhere on this card. If it found nothing at
                # all, the field boxes aren't sitting on text - which is
                # the signature of a wrong rotation, not of a hard-to-read
                # field - and the fallback ladder's extra OCR passes per
                # field are pure waste. This matters because every failed
                # image pays for all four rotations: skipping the ladder
                # on the hopeless ones is most of the difference between a
                # ~34s and a ~10s answer. The original reason the ladder
                # exists (one short field undetected while its neighbours
                # read fine) is unaffected - that case has detections.
                if not raw_text and free_text_by_field:
                    # Last resort for a field the clustered pass found
                    # nothing for at all: real-card testing showed a short
                    # single word (first_name) can still go undetected even
                    # with a whole-cluster's worth of surrounding context -
                    # a different failure mode than the "isolated tiny crop"
                    # bug the clustering fixed, so it needs a different
                    # remedy: much heavier upscaling plus contrast
                    # enhancement on just this field's own crop, tried only
                    # when the cheaper pass already came back empty (so it
                    # doesn't cost anything for the fields that work first
                    # try).
                    raw_text, confidence = self._ocr_field_fallback(crop)
                text = self._purify_free_text(field_name, raw_text)

            refined_data[field_name] = text.strip()
            field_confidences.append(confidence)

        return refined_data, refined_crops

    # Crop offsets (dx, dy) tried for a numeric field whose reading lacks
    # full support - see the call site. Ordered smallest-first so the
    # gentlest correction that works is the one taken, and sized against
    # the drift actually measured on real scans (up to ~28px).
    NUMERIC_CROP_OFFSETS = [
        (0, -12), (0, 12), (-12, 0), (12, 0),
        (0, -24), (0, 24), (-24, 0), (24, 0),
        (-12, -12), (12, -12), (-12, 12), (12, 12),
    ]

    # Fields the offset search is worth running for. birth_date is
    # deliberately excluded: it targets the front's security-hologram
    # watermark, which is known-unreliable and essentially never reaches
    # full support - so it triggered the search on EVERY front scan,
    # paying up to 12 extra runs of the whole binarization ladder and
    # taking a normal front from ~5s to ~25s. It buys nothing even when
    # it succeeds, because date_of_birth is derived from the
    # checksum-validated national_id anyway; the OCR'd value only serves
    # as a cross-check, and a cross-check is not worth 20 seconds.
    NUMERIC_OFFSET_RETRY_FIELDS = {"national_id", "issue_date", "expiry_date"}

    # Card validity is exactly 7 years, and the expiry keeps the issue's
    # month - confirmed on a real card (issued 2024/08, expires
    # 2031/08/12) and already assumed by this project's own generator.
    CARD_VALIDITY_YEARS = 7

    # How far beyond its validity period an issue date is still accepted
    # as a genuine reading. People present expired cards, and the right
    # response is to read them and say they are expired - not to reject
    # the reading. Only readings older than this are treated as OCR
    # errors rather than old documents.
    #
    # Kept deliberately tight (was 5). This bound is not just a display
    # filter - it gates _repair_issue_year_from_expiry, and a loose
    # window let that repair corrupt a CORRECT issue date: on a real
    # "enhanced" scan the expiry year misread 2031 as 2021, which implies
    # an issue year of 2014, and a 5-year grace accepted 2014 as
    # plausible. At 2 years the impossible implication is rejected and
    # the correct issue date survives, while a card up to two years past
    # expiry still reads.
    EXPIRED_CARD_GRACE_YEARS = 2

    @classmethod
    def _issue_date_is_plausible(cls, digits: str) -> bool:
        """
        Whether a YYYYMM reading could be a real issue date on a card
        someone is holding today. Every bound is derived from the current
        date, so these stay correct as years pass rather than needing a
        constant bumped.

        Two things a card cannot do:

        * Be issued in the future. Not just a future YEAR - a card issued
          later this year than the month we are in is equally impossible.
        * Be issued so long ago that the reading must be wrong. The lower
          bound is the validity period PLUS a grace window, not the
          validity period itself: an expired card is a real card someone
          can hand over, and it should be read and flagged as expired
          (see is_expired), not rejected as unreadable. The bound exists
          to catch OCR garbage like a 1999 issue date, not to filter out
          expired documents.
        """
        from datetime import date

        if len(digits) != 6 or not digits.isdigit():
            return False
        year, month = int(digits[:4]), int(digits[4:6])
        if not 1 <= month <= 12:
            return False

        today = date.today()
        if year > today.year or (year == today.year and month > today.month):
            return False
        return year >= today.year - (cls.CARD_VALIDITY_YEARS + cls.EXPIRED_CARD_GRACE_YEARS)

    @classmethod
    def _repair_issue_year_from_expiry(cls, issue_value: str, expiry_value: str) -> str:
        """
        Recovers a misread issue YEAR from the expiry year, which must be
        exactly CARD_VALIDITY_YEARS later.

        Neither date carries a check digit, so when they contradict each
        other something has to decide which one moved. The rule used here
        is minimal edit distance: on a real greyscale scan the issue year
        read 2020 against an expiry year of 2031. Correcting the issue
        year to 2024 is a ONE-digit change; "correcting" the expiry year
        to 2027 instead would take two. The single-digit explanation is
        the likelier misread, and is the only one adopted - anything
        needing more than one changed digit is left alone, because at
        that point the evidence no longer says which field is wrong.

        Deliberately only touches the YEAR. The expiry month can be
        equally misread ("81" appears on real scans) and the day exists
        on neither field's counterpart, so there is nothing to derive
        them from.
        """
        import re

        issue_match = re.match(r'^(\d{4})/(\d{2})$', issue_value or "")
        expiry_match = re.match(r'^(\d{4})/(\d{2})/(\d{2})$', expiry_value or "")
        if not issue_match or not expiry_match:
            return issue_value

        issue_year, issue_month = issue_match.group(1), issue_match.group(2)
        expiry_year = int(expiry_match.group(1))
        if int(issue_year) + cls.CARD_VALIDITY_YEARS == expiry_year:
            return issue_value  # already consistent

        candidate_year = expiry_year - cls.CARD_VALIDITY_YEARS
        candidate = f"{candidate_year:04d}"
        differing = sum(1 for a, b in zip(candidate, issue_year) if a != b)
        if differing != 1:
            return issue_value

        repaired = f"{candidate}/{issue_month}"
        if not cls._issue_date_is_plausible(candidate + issue_month):
            return issue_value
        return repaired

    @staticmethod
    def _card_is_expired(expiry_value: str) -> Optional[bool]:
        """
        Whether a card with this expiry date has expired today, or None
        if the date can't be read as a real calendar date.

        Compares the full date, not the month: a card expiring 03/2026 is
        still valid for part of March 2026 and expired from April, and
        within March it comes down to the day - which is why the card
        prints one.

        A day-less "YYYY/MM" (see _repair_expiry_year: the validity rule
        recovers the expiry's year and month even when its digits are
        unreadable, but never its day) is still answerable for every
        month EXCEPT the expiry month itself - before it the card is
        certainly valid, after it certainly expired. Only inside that one
        month does the answer depend on the day we do not have, and there
        this returns None rather than picking the flattering side.
        """
        import re
        from datetime import date

        today = date.today()

        full = re.match(r'^(\d{4})/(\d{2})/(\d{2})$', expiry_value or "")
        if full:
            try:
                expiry = date(int(full.group(1)), int(full.group(2)), int(full.group(3)))
            except ValueError:
                return None
            return expiry < today

        partial = re.match(r'^(\d{4})/(\d{2})$', expiry_value or "")
        if not partial:
            return None
        year, month = int(partial.group(1)), int(partial.group(2))
        if not 1 <= month <= 12:
            return None
        if (year, month) == (today.year, today.month):
            return None  # the missing day is exactly what would decide it
        return (year, month) < (today.year, today.month)

    @classmethod
    def _expiry_agrees_with_issue(cls, expiry_digits: str, issue_value: str) -> bool:
        """
        Whether an expiry reading is consistent with the issue date
        already read from this card.

        This is the date fields' equivalent of the national ID's check
        digit: two independently-read fields confirming each other is far
        stronger evidence that a crop landed correctly than any amount of
        agreement between preprocessing methods on one field alone. Used
        as the acceptance test when re-cutting expiry_date at an offset.
        """
        from datetime import date

        if len(expiry_digits) != 8 or not expiry_digits.isdigit():
            return False
        year, month, day = (
            int(expiry_digits[:4]), int(expiry_digits[4:6]), int(expiry_digits[6:8])
        )
        if not (1 <= month <= 12 and 1 <= day <= 31):
            return False

        issue_digits = "".join(ch for ch in (issue_value or "") if ch.isdigit())
        if not cls._issue_date_is_plausible(issue_digits):
            # No trustworthy issue date to compare against - fall back to
            # the expiry standing on its own.
            return 2000 <= year <= date.today().year + cls.CARD_VALIDITY_YEARS

        issue_year, issue_month = int(issue_digits[:4]), int(issue_digits[4:6])
        return year == issue_year + cls.CARD_VALIDITY_YEARS and month == issue_month

    @staticmethod
    def _nid_is_self_validating(text: str) -> bool:
        """A 14-digit reading that satisfies its own check digit AND
        describes a possible cardholder - see national_id_parser."""
        from ..postprocessing.national_id_parser import (
            is_structurally_valid_nid,
            validate_nid_checksum,
        )

        return validate_nid_checksum(text) and is_structurally_valid_nid(text)

    def _retry_numeric_at_offsets(
        self,
        source: np.ndarray,
        box: list,
        expected_length: Optional[int],
        separators: Optional[List[int]],
        validator=None,
    ) -> tuple[str, float]:
        """Re-cuts one numeric field at small offsets from its nominal box,
        returning the best-supported reading found (or an empty result).

        Two acceptance bars, strongest first:

        * `validator` - for national_id, a reading that satisfies its own
          Mod-11 check digit and structural checks. This is far stronger
          evidence that the crop landed correctly than any amount of
          method agreement: the digits are confirming each other. Worth
          the dedicated path because a shifted crop usually yields a
          reading that fails the checksum, so passing it is close to
          conclusive.
        * CONFIDENCE_METHODS_AGREE - two independent binarization methods
          producing the same expected-length string, used for the date
          fields, which carry no checksum of their own.

        Anything weaker is discarded: moving a field off its calibrated
        position needs real evidence, not a marginally better guess.
        """
        x, y, w, h = box
        canvas_h, canvas_w = source.shape[:2]
        best_text, best_confidence = "", 0.0

        for dx, dy in self.NUMERIC_CROP_OFFSETS:
            nx, ny = x + dx, y + dy
            if nx < 0 or ny < 0 or nx + w > canvas_w or ny + h > canvas_h:
                continue
            crop = source[ny:ny + h, nx:nx + w]
            if crop.size == 0:
                continue
            text, confidence = self.digit_engine.extract_digits_detailed(
                crop, expected_length=expected_length, separator_positions=separators
            )
            if validator is not None and text and validator(text):
                return text, self.digit_engine.CONFIDENCE_METHODS_AGREE
            if confidence >= self.digit_engine.CONFIDENCE_METHODS_AGREE and confidence > best_confidence:
                best_text, best_confidence = text, confidence

        return best_text, best_confidence

    # Configs tried in order by _ocr_field_fallback, stopping at the first
    # that finds anything - (upscale_factor, use_clahe, padding). Confirmed
    # on a real back card that no single config is universally best: 4x+
    # CLAHE (originally tuned for first_name) found NOTHING across every
    # padding for marital_status's crop, while plain 2x-3x upscale (no
    # CLAHE) found the correct-length result at 0.88-1.00 confidence on
    # the same crop. Ordered cheapest-first, ending with the original
    # 4x+CLAHE config so whatever first_name originally needed is still
    # tried if the cheaper options fail.
    FALLBACK_CONFIGS = [
        (2, False, 30),
        (3, False, 30),
        (4, True, 30),
    ]

    def _ocr_field_fallback(self, crop: np.ndarray) -> tuple:
        """
        Heavier last-resort OCR attempt for a single field crop that came
        back with zero detections from the clustered whole-region pass.
        Tries FALLBACK_CONFIGS in order, stopping at the first upscale/
        contrast/padding combination that finds anything - more expensive
        than the primary path, but only ever run on fields that already
        failed, so it doesn't slow down the common case.
        """
        for scale, use_clahe, pad in self.FALLBACK_CONFIGS:
            try:
                upscaled = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_LANCZOS4)
                if use_clahe:
                    gray = cv2.cvtColor(upscaled, cv2.COLOR_BGR2GRAY)
                    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                    enhanced_gray = clahe.apply(gray)
                    upscaled = cv2.cvtColor(enhanced_gray, cv2.COLOR_GRAY2BGR)
                padded = cv2.copyMakeBorder(upscaled, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
                results = self.ocr_engine.read_layout(padded, mode="auto")
            except Exception as e:
                logger.warning(f"OCR fallback failed (scale={scale}, clahe={use_clahe}): {e}")
                continue

            if results:
                results = self._sort_reading_order(results)
                text = " ".join(r[1] for r in results)
                confidence = min(r[2] for r in results)
                return text, confidence

        return "", 0.0

    # Two detections are "the same line" if their vertical centers are
    # within this many pixels - real jitter observed between two words on
    # one visual line was as much as ~1-3px (e.g. "محمد"/"جاد" centered at
    # y=282.5 vs y=281.5), enough to flip a naive exact-y sort and silently
    # reverse RTL word order even though nothing about the text itself was
    # wrong (confirmed: the actual x-positions were correctly right-to-
    # left, only the sort's tie-breaking on noisy y was not).
    LINE_GROUPING_TOLERANCE = 20

    def _sort_reading_order(self, results: list) -> list:
        """Sorts OCR detections top-to-bottom, right-to-left (Arabic
        reading order), grouping near-equal y-centers into the same line
        first so sub-pixel detection noise can't override x-ordering
        within a line - see LINE_GROUPING_TOLERANCE."""
        results = sorted(results, key=lambda r: np.mean(np.array(r[0])[:, 1]))
        lines: list = []
        for r in results:
            cy = np.mean(np.array(r[0])[:, 1])
            if lines and abs(cy - lines[-1][0]) <= self.LINE_GROUPING_TOLERANCE:
                lines[-1][1].append(r)
                lines[-1][0] = (lines[-1][0] * len(lines[-1][1]) + cy) / (len(lines[-1][1]) + 1)
            else:
                lines.append([cy, [r]])

        ordered = []
        for _, line in lines:
            line.sort(key=lambda r: -np.mean(np.array(r[0])[:, 0]))
            ordered.extend(line)
        return ordered

    @staticmethod
    def _merge_free_text_passes(
        primary: Dict[str, tuple], alt: Dict[str, tuple]
    ) -> Dict[str, tuple]:
        """
        Combines the raw and contrast-normalized OCR passes over one
        cluster, deliberately biased towards the raw pass.

        The alt pass is only allowed to do two things:

        * fill a field the raw pass left completely empty, and
        * EXTEND a field whose raw reading it fully contains.

        It can never substitute a different reading for one the raw pass
        already produced. That asymmetry is the whole point: normalization
        helps and hurts the same field on different scans (it recovered a
        word on a real greyscale back, and truncated that same field on a
        cleaner scan of the same card), so "longest wins" would trade one
        card's bug for another's. Requiring the raw text to appear inside
        the alt text means the alt pass has to agree with everything the
        raw pass already read before it is trusted to add to it - so a
        scan where the raw pass is right is unreachable by this code path.
        """
        merged = dict(primary)
        for name, (alt_text, alt_confidence) in alt.items():
            current_text, _ = merged.get(name, ("", 0.0))
            current = current_text.strip()
            candidate = alt_text.strip()
            if not candidate:
                continue
            if not current:
                merged[name] = (alt_text, alt_confidence)
            elif len(candidate) > len(current) and current in candidate:
                merged[name] = (alt_text, alt_confidence)
        return merged

    def _bucket_free_text(self, ocr_results: list, field_boxes: Dict[str, list]) -> Dict[str, tuple]:
        """
        Assigns each whole-card OCR detection to whichever field box
        contains its center point, then joins same-field detections in
        reading order (top-to-bottom, right-to-left) - the same order a
        single crop's detections used to be sorted in, now applied across
        possibly-multiple detections spanning one field (e.g. address's
        two lines). A detection whose center doesn't fall in any field's
        box (card chrome, labels, etc.) is simply dropped.
        """
        buckets: Dict[str, list] = {name: [] for name in field_boxes}
        for poly, text, score in ocr_results:
            pts = np.array(poly)
            cx, cy = float(np.mean(pts[:, 0])), float(np.mean(pts[:, 1]))
            assigned = None
            for field_name, (fx, fy, fw, fh) in field_boxes.items():
                if fx <= cx <= fx + fw and fy <= cy <= fy + fh:
                    buckets[field_name].append((poly, text, score))
                    assigned = field_name
                    break  # a detection belongs to at most one field
            if _BUCKET_DEBUG:
                x0, y0 = float(pts[:, 0].min()), float(pts[:, 1].min())
                x1, y1 = float(pts[:, 0].max()), float(pts[:, 1].max())
                logger.warning(
                    "[bucket] det=(%.0f,%.0f,%.0f,%.0f) centroid=(%.0f,%.0f) "
                    "chars=%d score=%.2f -> %s",
                    x0, y0, x1 - x0, y1 - y0, cx, cy, len(text), score,
                    assigned or "DROPPED",
                )
        if _BUCKET_DEBUG:
            for name, (fx, fy, fw, fh) in field_boxes.items():
                logger.warning("[bucket] box %-16s=(%d,%d,%d,%d)", name, fx, fy, fw, fh)

        joined: Dict[str, tuple] = {}
        for field_name, matches in buckets.items():
            if not matches:
                continue
            matches = self._sort_reading_order(matches)
            text = " ".join(m[1] for m in matches)
            confidence = min(m[2] for m in matches)
            joined[field_name] = (text, confidence)
        return joined

    def _purify_free_text(self, field_name: str, text: str) -> str:
        """Field-specific cleanup applied after free-text OCR/bucketing."""
        if field_name in ("full_name", "first_name"):
            import re
            text = re.sub(r'[^\s\u0621-\u064A]', '', text)
            text = " ".join(text.split())
        elif field_name == "address":
            import re
            # Addresses legitimately contain building/street numbers (as
            # printed, Arabic-Indic), a district-governorate separator
            # dash, and (on some renders) a comma - stripping to Arabic-
            # letters-only (as first_name/full_name do) silently dropped
            # them.
            text = re.sub(r'[^\s\u0621-\u064A0-9\u0660-\u0669\u060C\-]', '', text)
            text = " ".join(text.split())
        elif field_name == "serial_number":
            import re
            # Real serials are 2 Latin letters + digits (e.g. "AB1234567") -
            # keep only that character set, uppercased (OCR sometimes
            # returns lowercase for this font).
            text = re.sub(r'[^A-Za-z0-9]', '', text).upper()
        # profession/gender/religion/marital_status (back side) keep the
        # raw OCR text as-is - gender/religion/marital_status get fuzzy-
        # matched against a closed vocabulary downstream (_populate_back),
        # which tolerates noise better than a stripping regex would.
        return text

    def _populate_front_v4(self, refined_data: dict, refined_crops: dict) -> IDCardFront:
        from ..postprocessing.transliteration import transliterate

        return IDCardFront(
            first_name=refined_data.get("first_name", ""),
            full_name=refined_data.get("full_name", ""),
            address=refined_data.get("address", ""),
            first_name_english=transliterate(refined_data.get("first_name", "")),
            full_name_english=transliterate(refined_data.get("full_name", "")),
            address_english=transliterate(refined_data.get("address", "")),
            national_id=refined_data.get("national_id", ""),
            date_of_birth=refined_data.get("birth_date", ""),
            card_serial_number=refined_data.get("serial_number", ""),
            field_crops=refined_crops,
            raw_arabic=refined_data
        )

    def _populate_back(self, refined_data: dict, refined_crops: dict) -> IDCardBack:
        # gender/religion/marital_status are closed-vocabulary - fuzzy-match
        # the raw OCR text against the known Arabic values instead of
        # trusting it directly (same "constrain to a valid answer" idea as
        # checksum validation for the NID).
        gender, _ = match_enum(refined_data.get("gender", ""), GENDER_VALUES)
        religion, _ = match_enum(refined_data.get("religion", ""), RELIGION_VALUES)
        marital_status, _ = match_enum(refined_data.get("marital_status", ""), MARITAL_STATUS_VALUES)

        from ..postprocessing.transliteration import transliterate

        return IDCardBack(
            national_id=refined_data.get("national_id", ""),
            issue_date=refined_data.get("issue_date", ""),
            profession=refined_data.get("profession", ""),
            profession_english=transliterate(refined_data.get("profession", "")),
            gender=gender or Gender.UNKNOWN,
            religion=religion or Religion.UNKNOWN,
            marital_status=marital_status or MaritalStatus.UNKNOWN,
            expiry_date=refined_data.get("expiry_date", ""),
            field_crops=refined_crops,
            raw_arabic=refined_data,
        )

    @staticmethod
    def _ocr_date_to_iso(value: str) -> Optional[str]:
        """
        Converts the digit classifier's "YYYY/MM/DD" output into an ISO
        date string, or None if it isn't a real date.

        The None case is the important one: this feeds the National ID
        repair's cross-check, and real scans produce readings like
        "0081/71/00", "3001/10", "4136/11/51" and "9" for this field. A
        garbled value must not be handed over as if it were a trustworthy
        independent signal - no corroboration is strictly better than
        false corroboration, since the repair falls back to
        evidence-backed edits only.

        Parsing as a calendar date isn't enough: "4136/11/51" fails that,
        but a misread like "2019/03/04" would pass while describing
        someone too young to hold an Egyptian ID at all. So the date must
        also be a plausible BIRTH date - see is_plausible_birth_date.
        """
        import re
        from datetime import date as _date
        from ..postprocessing.national_id_parser import is_plausible_birth_date

        match = re.fullmatch(r'(\d{4})/(\d{2})/(\d{2})', (value or "").strip())
        if not match:
            return None
        try:
            iso = _date(int(match.group(1)), int(match.group(2)), int(match.group(3))).isoformat()
        except ValueError:
            return None
        return iso if is_plausible_birth_date(iso) else None

    @staticmethod
    def _normalize_card(card: np.ndarray) -> np.ndarray:
        """
        Normalizes lighting/contrast on the whole aligned card, once,
        before any field-level cutting or OCR - see the call site in
        _process_single_orientation for why this belongs here rather than
        left to individual OCR/digit calls to each separately compensate
        for.

        CLAHE on the LAB color space's L (lightness) channel rather than
        converting to plain grayscale: this is what actually normalizes
        contrast (a black-and-white photocopy and a well-lit color photo
        end up with comparable L-channel contrast after this), while
        leaving color untouched for any current or future step that still
        benefits from it - strictly safer than discarding color outright
        for no benefit.
        """
        lab = cv2.cvtColor(card, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        l_channel = clahe.apply(l_channel)
        normalized_lab = cv2.merge((l_channel, a_channel, b_channel))
        return cv2.cvtColor(normalized_lab, cv2.COLOR_LAB2BGR)

    @staticmethod
    def _repair_expiry_year(issue_date: str, expiry_date: str) -> str:
        """
        Corrects expiry_date's year to issue_date's year + 7 if they
        disagree - see the call site in _process_single_orientation for
        why this is trustworthy (a fixed, known validity period, not a
        guess). Only touches the year; month/day are left as OCR'd since
        nothing suggests they're less reliable than the day/month digits
        elsewhere.

        A reading whose month or day CANNOT exist (real scans produced
        "2027/81/01" and "2031/81/25" - there is no month 81) is not
        shown as read. But blanking the whole field throws away more than
        it has to: the validity rule fixes BOTH the year and the month of
        expiry from the issue date, so those two are recoverable even when
        the digits are not. Only the DAY is genuinely unknown.

        So an unreadable expiry degrades to a day-less "YYYY/MM" derived
        from the issue date, rather than to nothing. The day is never
        invented to fill the format out - a fabricated day would read as
        precision the scan does not support, and _card_is_expired answers
        "unknown" instead of guessing for the one month where the missing
        day is what decides the question.

        The day is also not worth searching for: re-running the per-field
        offset search once the issue date is corrected does recover the
        right year and month from the image (measured on a real greyscale
        back), but it costs ~13s and still returned a transposed day
        (21 for 12) that nothing can check. Deriving the same year and
        month from the issue date costs nothing and is corroborated by
        the 7-year rule.
        """
        import re
        issue_match = re.match(r'^(\d{4})/(\d{2})$', issue_date or "")
        expiry_match = re.match(r'^(\d{4})/(\d{2})/(\d{2})$', expiry_date or "")

        if expiry_match:
            month, day = int(expiry_match.group(2)), int(expiry_match.group(3))
            if not (1 <= month <= 12 and 1 <= day <= 31):
                if issue_match:
                    return (
                        f"{int(issue_match.group(1)) + Pipeline.CARD_VALIDITY_YEARS:04d}"
                        f"/{issue_match.group(2)}"
                    )
                return ""

        if not issue_match or not expiry_match:
            return expiry_date

        issue_year = int(issue_match.group(1))
        expiry_year = int(expiry_match.group(1))
        expected_year = issue_year + 7

        if expiry_year != expected_year:
            return f"{expected_year:04d}/{expiry_match.group(2)}/{expiry_match.group(3)}"
        return expiry_date

    def _enhance_crop(self, crop: np.ndarray) -> np.ndarray:
        """Applies upscaling, sharpening and adaptive thresholding for high-res OCR pass."""
        try:
            if crop.size == 0: return crop
            
            # 1. Upscaling (2x) using Lanczos interpolation
            crop = cv2.resize(crop, None, fx=2, fy=2, interpolation=cv2.INTER_LANCZOS4)
            
            # 2. To Grayscale
            if len(crop.shape) == 3:
                gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            else:
                gray = crop

            # 3. Denoising
            denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
            
            # 4. Adaptive Thresholding (Hamza's B&W approach)
            thresh = cv2.adaptiveThreshold(denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 5)
            
            # 5. Contrast Stretching
            min_val, max_val, _, _ = cv2.minMaxLoc(denoised)
            stretched = cv2.convertScaleAbs(denoised, alpha=255.0/(max_val - min_val), beta=-min_val * 255.0/(max_val - min_val))
            
            return thresh
        except Exception as e:
            logger.error(f"Enhancement failed: {e}")
            return crop
