"""
Privacy-preserving calibration helper: draws the current FRONT_FIELDS (or
BACK_FIELDS) crop boxes (in red) plus a labeled 50px coordinate grid (in
green, with axis numbers) over the aligned/rectified version of a real
card photo, and saves the result to a LOCAL file only - nothing is sent
anywhere.

Run it yourself, open the saved image, and report back (in text) which
boxes need to move and roughly where the real text actually starts/ends
using the grid coordinates - e.g. "national_id text runs from about
x=550..900, y=580..650". That's enough to fix the calibration without ever
sharing the card image itself.

Usage:
    python scripts/debug_field_grid.py path/to/your/photo.jpg [front|back]
"""
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from egyptian_national_id_ocr.core.layout_analyzer import LayoutAnalyzer


def load_image_exif_aware(path: str) -> np.ndarray:
    pil_image = Image.open(path)
    pil_image = ImageOps.exif_transpose(pil_image)
    return cv2.cvtColor(np.array(pil_image.convert("RGB")), cv2.COLOR_RGB2BGR)


def main():
    if len(sys.argv) not in (2, 3):
        print("Usage: python scripts/debug_field_grid.py path/to/your/photo.jpg [front|back]")
        sys.exit(1)

    image_path = sys.argv[1]
    side = sys.argv[2] if len(sys.argv) == 3 else "front"
    if side not in ("front", "back"):
        print("side must be 'front' or 'back'")
        sys.exit(1)

    image = load_image_exif_aware(image_path)

    analyzer = LayoutAnalyzer()
    rectified, M = analyzer.align_card(image)
    if M is None:
        print("Note: card contour not detected - fell back to a simple resize "
              "of the whole photo. If your photo has visible background around "
              "the card, that's likely worth fixing separately.")

    canvas = rectified.copy()
    h, w = canvas.shape[:2]

    # Green grid every 50px with coordinate labels
    for x in range(0, w, 50):
        cv2.line(canvas, (x, 0), (x, h), (0, 200, 0), 1)
        cv2.putText(canvas, str(x), (x + 2, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 200, 0), 1)
    for y in range(0, h, 50):
        cv2.line(canvas, (0, y), (w, y), (0, 200, 0), 1)
        cv2.putText(canvas, str(y), (2, y + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 200, 0), 1)

    # Red boxes = current FRONT_FIELDS or BACK_FIELDS crop coordinates
    field_boxes = analyzer.FRONT_FIELDS if side == "front" else analyzer.BACK_FIELDS
    for name, (x, y, bw, bh) in field_boxes.items():
        cv2.rectangle(canvas, (x, y), (x + bw, y + bh), (0, 0, 255), 2)
        cv2.putText(canvas, name, (x, max(0, y - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    out_path = str(Path(image_path).with_name(Path(image_path).stem + f"_{side}_grid_debug.png"))
    cv2.imwrite(out_path, canvas)
    print(f"Saved (LOCAL ONLY, not sent anywhere): {out_path}")
    print(f"Rectified canvas size: {w}x{h}")
    print("\nOpen that file yourself and tell me, in text, using the green grid numbers:")
    print("  - For each red box: does it cover the real text? If not, where does the")
    print("    real text actually start/end (approx x0,y0 to x1,y1)?")


if __name__ == "__main__":
    main()
