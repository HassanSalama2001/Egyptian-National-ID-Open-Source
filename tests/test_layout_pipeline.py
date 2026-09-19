import cv2
import pytest
from egyptian_national_id_ocr.core.pipeline import Pipeline


def test_pipeline_extracts_a_valid_checksummed_nid(front_image_path, synthetic_card_sample):
    """
    End-to-end smoke test against a synthetic card with known ground
    truth (see conftest.py's synthetic_card_sample - a seeded, calibrated
    generator, never a real card). Unlike the old assets/dataset/-backed
    version of this test (retired along with that dataset - it predated
    field-box calibration and was marked xfail, so it never actually
    verified anything), this asserts the EXACT expected national_id, not
    just "some 14 digits that pass their own checksum".
    """
    img = cv2.imread(str(front_image_path))
    assert img is not None, f"Could not load {front_image_path}"

    pipeline = Pipeline()  # default engine (PaddleOCR) - see pipeline.py
    result = pipeline.process_image(img)

    assert result.front is not None
    assert result.front.national_id == synthetic_card_sample["data"]["national_id"]
    assert result.decoded is not None, (
        "NID failed checksum validation - extraction produced an "
        "internally-inconsistent result"
    )
