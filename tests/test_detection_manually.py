import cv2

from egyptian_national_id_ocr.detection.card_detector import CardDetector


def test_detection(synthetic_card_sample):
    """
    Rewritten - the old version (pointed at the now-retired
    assets/dataset/) had no assertions at all: it printed a message and
    `continue`d past any failure, so it "passed" unconditionally
    regardless of whether detection worked, and it unconditionally wrote
    detected_*.png files into the working directory as a side effect.
    Neither is true of this version.
    """
    detector = CardDetector()

    for path in (synthetic_card_sample["front_path"], synthetic_card_sample["back_path"]):
        image = cv2.imread(str(path))
        assert image is not None, f"Could not load {path}"

        detected = detector.detect(image)
        assert detected is not None, f"CardDetector.detect() returned None for {path}"
        assert detected.size > 0, f"CardDetector.detect() returned an empty image for {path}"
