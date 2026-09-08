import cv2
import pytest
from national_id_ocr.core.pipeline import Pipeline
from national_id_ocr.ocr.easyocr_engine import EasyOCREngine


@pytest.mark.xfail(
    reason=(
        "Known gap: NID field-crop ROI doesn't reliably capture all 14 "
        "digits on real (non-synthetic) card photos yet - tracked as part "
        "of the OCR redesign (lightweight digit recognizer + ROI accuracy)."
    ),
    strict=False,
)
def test_pipeline_extracts_a_valid_checksummed_nid(front_image_path):
    """
    End-to-end smoke test: the pipeline should extract a 14-digit NID that
    passes its own Mod-11 checksum from a real front-side sample image.
    This does not assert the *exact* digits (we don't have ground truth for
    assets/front.jpg), only that extraction+repair produces something
    internally consistent - the strongest thing we can check without a
    labeled dataset.
    """
    img = cv2.imread(str(front_image_path))
    assert img is not None, f"Could not load {front_image_path}"

    engine = EasyOCREngine(gpu=False)
    pipeline = Pipeline(ocr_engine=engine)
    result = pipeline.process_image(img)

    assert result.front is not None
    assert result.front.national_id is not None
    assert len(result.front.national_id) == 14, (
        f"Expected 14-digit NID, got {result.front.national_id!r}"
    )
    assert result.decoded is not None, (
        "NID failed checksum validation - extraction produced an "
        "internally-inconsistent result"
    )
