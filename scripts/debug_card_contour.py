"""
Privacy-safe diagnostic for the "Could not detect card contour. Falling
back to simple resize." log line - prints ONLY geometry (never any text,
image content, or field values) so this can be run against your own real
card photo and the output pasted back safely.

Usage:
    python scripts/debug_card_contour.py path/to/front.jpg
    python scripts/debug_card_contour.py path/to/back.jpg
"""
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from egyptian_national_id_ocr.core.layout_analyzer import LayoutAnalyzer, CARD_ASPECT_RATIO, CARD_ASPECT_TOLERANCE


def load_like_api(path: str) -> np.ndarray:
    """Mirrors app.py's exact decode path (PIL + exif_transpose -> BGR)."""
    pil_image = Image.open(path)
    pil_image = ImageOps.exif_transpose(pil_image)
    return cv2.cvtColor(np.array(pil_image.convert("RGB")), cv2.COLOR_RGB2BGR)


def diagnose(image: np.ndarray, label: str):
    analyzer = LayoutAnalyzer()
    h, w = image.shape[:2]
    print(f"  input size: {w}x{h}  (aspect {w/h:.3f})")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=3)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    method = "otsu"
    if not contours:
        edged = cv2.Canny(gray, 50, 200)
        contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        method = "canny (otsu found nothing)"

    image_area = w * h
    contours_sorted = sorted(contours, key=cv2.contourArea, reverse=True)[:5]
    print(f"  threshold method: {method}, candidate contours: {len(contours_sorted)}")

    found_4pt = False
    for i, c in enumerate(contours_sorted):
        area = cv2.contourArea(c)
        area_frac = area / image_area
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        (_, (rw, rh), _) = cv2.minAreaRect(c)
        ratio = max(rw, rh) / min(rw, rh) if min(rw, rh) > 0 else 0
        ratio_ok = abs(ratio - CARD_ASPECT_RATIO) <= CARD_ASPECT_TOLERANCE
        big_enough = area_frac >= 0.1
        print(f"  candidate #{i}: area_frac={area_frac:.3f} (need >=0.100)  "
              f"points={len(approx)} (need ==4)  aspect_ratio={ratio:.3f} "
              f"(need {CARD_ASPECT_RATIO - CARD_ASPECT_TOLERANCE:.3f}-{CARD_ASPECT_RATIO + CARD_ASPECT_TOLERANCE:.3f}) "
              f"-> {'PASS' if (len(approx) == 4 and ratio_ok and big_enough) else 'reject'}")
        if len(approx) == 4 and ratio_ok and big_enough:
            found_4pt = True

    if found_4pt:
        print(f"  RESULT: a valid 4-point card contour was found - alignment should succeed.")
    else:
        print(f"  RESULT: NO candidate passed all 3 checks - this image will hit the "
              f"'Could not detect card contour' fallback (naive resize, no perspective "
              f"correction) exactly like the log showed.")

    # Also run the real align_card to confirm which path it actually takes.
    contour = analyzer.get_card_contour(image)
    print(f"  get_card_contour() returned: {'a contour' if contour is not None else 'None (fallback path)'}")


def main():
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)

    path = sys.argv[1]
    image = load_like_api(path)

    print(f"=== {Path(path).name} (0 deg) ===")
    diagnose(image, "0deg")

    for rot, label in [(cv2.ROTATE_90_CLOCKWISE, "90deg"),
                        (cv2.ROTATE_180, "180deg"),
                        (cv2.ROTATE_90_COUNTERCLOCKWISE, "270deg")]:
        rotated = cv2.rotate(image, rot)
        print(f"\n=== {Path(path).name} ({label}) ===")
        diagnose(rotated, label)


if __name__ == "__main__":
    main()
