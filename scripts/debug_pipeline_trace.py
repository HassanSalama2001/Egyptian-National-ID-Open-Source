"""
Privacy-preserving pipeline tracer: runs the full 4-rotation pipeline on a
real photo and prints ONLY metadata - which rotation was tried, whether a
card contour was found, the side classification and its raw scores, the
resulting confidence, and which fields extracted any text at all (field
NAMES only, never the extracted values, and never the image itself).

Nothing here is saved or sent anywhere - it only prints to your own
terminal. Copy/paste that text output back, not the image.

Usage:
    python scripts/debug_pipeline_trace.py path/to/your/photo.jpg
"""
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from egyptian_national_id_ocr.core.layout_analyzer import LayoutAnalyzer
from egyptian_national_id_ocr.core.pipeline import Pipeline
from egyptian_national_id_ocr.detection.side_classifier import SideClassifier, IDSide


def load_image_exif_aware(path: str) -> np.ndarray:
    pil_image = Image.open(path)
    pil_image = ImageOps.exif_transpose(pil_image)
    return cv2.cvtColor(np.array(pil_image.convert("RGB")), cv2.COLOR_RGB2BGR)


def main():
    if len(sys.argv) != 2:
        print("Usage: python scripts/debug_pipeline_trace.py path/to/your/photo.jpg")
        sys.exit(1)

    image = load_image_exif_aware(sys.argv[1])
    analyzer = LayoutAnalyzer()
    classifier = SideClassifier()
    pipeline = Pipeline()

    print(f"Input image size: {image.shape[1]}x{image.shape[0]}")
    print("=" * 70)

    for rotation, label in Pipeline.ROTATIONS_TO_TRY:
        attempt = cv2.rotate(image, rotation) if rotation is not None else image
        rectified, M = analyzer.align_card(attempt)
        contour_found = M is not None

        gray = cv2.cvtColor(rectified, cv2.COLOR_BGR2GRAY)
        f_score = classifier._get_match_score(gray, classifier.front_template)
        b_score = classifier._get_match_score(gray, classifier.back_template)
        gray_rot = cv2.rotate(gray, cv2.ROTATE_180)
        f_score_down = classifier._get_match_score(gray_rot, classifier.front_template)
        b_score_down = classifier._get_match_score(gray_rot, classifier.back_template)

        result, _orientation_confirmed = pipeline._process_single_orientation(attempt)

        extracted_fields = []
        if result.front is not None:
            for name in ["first_name", "full_name", "address", "national_id", "date_of_birth", "card_serial_number"]:
                val = getattr(result.front, name, None)
                if val:
                    extracted_fields.append(name)
        if result.back is not None:
            for name in ["national_id", "issue_date", "expiry_date", "profession"]:
                val = getattr(result.back, name, None)
                if val:
                    extracted_fields.append(name)

        print(f"Rotation {label}:")
        print(f"  card contour found: {contour_found}  (rectified size: {rectified.shape[1]}x{rectified.shape[0]})")
        print(f"  side-match scores: front(up)={f_score:.3f} back(up)={b_score:.3f} "
              f"front(180)={f_score_down:.3f} back(180)={b_score_down:.3f}")
        print(f"  classified side: {result.side}")
        print(f"  status: {result.status}  confidence: {result.confidence:.3f}  decoded: {result.decoded is not None}")
        print(f"  fields with ANY extracted text: {extracted_fields}")
        print("-" * 70)

    print("\nRunning the full process_image (what the app actually uses)...")
    final = pipeline.process_image(image)
    print(f"FINAL: side={final.side}  status={final.status}  confidence={final.confidence:.3f}  "
          f"decoded={final.decoded is not None}")


if __name__ == "__main__":
    main()
