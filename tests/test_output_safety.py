"""
Two output-level fixes from real third-party scans:

  * a National ID reading that could not belong to anyone (a real scan
    returned 00000000200000) must never be shown as if it were data;
  * first_name and full_name, printed on adjacent lines, must not come
    back merged into full_name when the OCR detector proposes both lines
    as one region.

Fast unit tests - no OCR model is loaded (Pipeline.__new__ plus a fake
engine), no real card is used.
"""
import numpy as np
import pytest

from egyptian_national_id_ocr.core.pipeline import Pipeline
from egyptian_national_id_ocr.models.id_card import IDCardFront


@pytest.fixture
def bare_pipeline():
    return Pipeline.__new__(Pipeline)


class TestImpossibleNIDIsHidden:
    @pytest.mark.parametrize("reading", [
        "00000000200000",   # the real-scan case: century 0, month 00
        "29913",            # partial read
        "39913459999999",   # month 13
    ])
    def test_impossible_reading_is_blanked(self, reading):
        side = IDCardFront(national_id=reading)
        assert Pipeline._hide_impossible_nid(side) is True
        assert side.national_id == ""

    def test_possible_but_unverified_reading_stays_visible(self):
        # Structurally possible (2000s century, real date, Cairo code 01)
        # but the checksum is not guaranteed - it may be one digit off, so
        # it stays visible for the user to check against the card.
        side = IDCardFront(national_id="30001010100000")
        assert Pipeline._hide_impossible_nid(side) is False
        assert side.national_id == "30001010100000"

    def test_empty_reading_is_left_alone(self):
        side = IDCardFront(national_id="")
        assert Pipeline._hide_impossible_nid(side) is False


class _FakeOCR:
    """Returns one detection per call, labelled with the crop's height so
    the test can tell which box was read."""
    def __init__(self):
        self.calls = []

    def read_layout(self, image, mode="auto"):
        h, w = image.shape[:2]
        self.calls.append((h, w))
        poly = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=float)
        return [(poly, f"line{h}", 0.9)]


class TestStackedNameLinesAreSplit:
    BOXES = {"first_name": [750, 200, 390, 65], "full_name": [550, 265, 590, 95]}

    def _run(self, pipeline, detections):
        pipeline.ocr_engine = _FakeOCR()
        card = np.full((750, 1200, 3), 255, np.uint8)
        return pipeline._split_merged_lines(card, detections, self.BOXES), pipeline.ocr_engine

    def test_one_detection_spanning_both_rows_is_reread_per_row(self, bare_pipeline):
        merged = np.array([[700, 205], [1130, 205], [1130, 350], [700, 350]], dtype=float)
        out, ocr = self._run(bare_pipeline, [(merged, "both lines", 0.8)])
        assert out["first_name"][0] == "line65"   # read from first_name's own box
        assert out["full_name"][0] == "line95"    # read from full_name's own box
        assert len(ocr.calls) == 2

    def test_separate_detections_are_left_to_normal_bucketing(self, bare_pipeline):
        first = np.array([[900, 205], [1130, 205], [1130, 255], [900, 255]], dtype=float)
        full = np.array([[600, 280], [1130, 280], [1130, 340], [600, 340]], dtype=float)
        out, ocr = self._run(bare_pipeline, [(first, "a", 0.9), (full, "b", 0.9)])
        assert out == {}
        assert ocr.calls == []
