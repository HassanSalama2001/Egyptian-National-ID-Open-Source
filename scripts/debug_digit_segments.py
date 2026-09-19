"""
Privacy-preserving diagnostic for the digit-classifier segmentation step
on a numeric field (default: birth_date, the one still failing). Prints
segment count, position, size, and the classified digit for each blob -
never saves or sends your photo anywhere except your own terminal.

Usage:
    .audit_venv\\Scripts\\python.exe scripts\\debug_digit_segments.py "path\\to\\photo.jpg" [field_name] [front|back]
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import cv2
import numpy as np
from PIL import Image, ImageOps

from egyptian_national_id_ocr.core.layout_analyzer import LayoutAnalyzer
from egyptian_national_id_ocr.ocr.digit_classifier_engine import DigitClassifierEngine


def load_image_exif_safe(path: str) -> np.ndarray:
    pil_image = Image.open(path)
    pil_image = ImageOps.exif_transpose(pil_image)
    return cv2.cvtColor(np.array(pil_image.convert("RGB")), cv2.COLOR_RGB2BGR)


def show_segments(label: str, engine: DigitClassifierEngine, gray: np.ndarray, method: str = "otsu", upscale=None):
    segments = engine._segment_digits(gray, method=method, upscale=upscale)
    print(f"\n--- {label} ---")
    if not segments:
        print("  (no segments found at all)")
        return
    print(f"  {len(segments)} segments found:")
    for x, seg in segments:
        h, w = seg.shape
        prepared = engine._prepare_crop(seg)
        from skimage.feature import hog
        feat = hog(prepared, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2))
        pred = engine.clf.predict(feat.reshape(1, -1))[0]
        print(f"    x={x:>4}  size={w}x{h}  classified_as={pred}")


def main():
    if len(sys.argv) < 2:
        print("Usage: debug_digit_segments.py <path-to-photo> [field_name] [front|back]")
        sys.exit(1)

    field_name = sys.argv[2] if len(sys.argv) > 2 else "birth_date"
    side = sys.argv[3] if len(sys.argv) > 3 else "front"

    image = load_image_exif_safe(sys.argv[1])
    analyzer = LayoutAnalyzer()
    rectified, _ = analyzer.align_card(image)
    field_boxes = analyzer.FRONT_FIELDS if side == "front" else analyzer.BACK_FIELDS
    crop = analyzer.get_field_crops(rectified, fields=field_boxes)[field_name]
    print(f"Field: {field_name} ({side})  crop shape: {crop.shape}")

    engine = DigitClassifierEngine()
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    for method in DigitClassifierEngine.BINARIZATION_METHODS:
        show_segments(f"method={method}", engine, gray, method=method)

    print("\nCopy everything above and paste it back.")


if __name__ == "__main__":
    main()
