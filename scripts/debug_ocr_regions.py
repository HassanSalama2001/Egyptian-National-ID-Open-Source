"""
Privacy-preserving diagnostic for the free-text OCR step (first_name,
full_name, address, serial_number). Runs the exact same clustered-region
OCR pass the pipeline uses on your real photo, but only prints structural
info back to your terminal - detection position, confidence score, and
text LENGTH (not the actual recognized text) - so this is safe to paste
back without sharing your name/address content itself.

Never saves, sends, or displays the photo or its contents anywhere except
your own terminal.

Usage:
    .audit_venv\\Scripts\\python.exe scripts\\debug_ocr_regions.py "path\\to\\front.jpg"
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import cv2
import numpy as np
from PIL import Image, ImageOps

from egyptian_national_id_ocr.core.layout_analyzer import LayoutAnalyzer
from egyptian_national_id_ocr.core.pipeline import Pipeline
from egyptian_national_id_ocr.ocr.paddle_ocr_engine import PaddleOCREngine


def load_image_exif_safe(path: str) -> np.ndarray:
    pil_image = Image.open(path)
    pil_image = ImageOps.exif_transpose(pil_image)
    return cv2.cvtColor(np.array(pil_image.convert("RGB")), cv2.COLOR_RGB2BGR)


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: debug_ocr_regions.py <path-to-front-photo>")
        sys.exit(1)

    image = load_image_exif_safe(sys.argv[1])
    analyzer = LayoutAnalyzer()
    rectified, _ = analyzer.align_card(image)

    pipeline = Pipeline(ocr_engine=PaddleOCREngine())
    free_text_boxes = {
        name: box for name, box in analyzer.FRONT_FIELDS.items()
        if name not in pipeline.NUMERIC_FIELDS
    }
    clusters = pipeline._cluster_field_boxes(
        free_text_boxes, pipeline.CLUSTER_PADDING, rectified.shape[1], rectified.shape[0],
        never_merge=pipeline.NEVER_MERGE_PAIRS,
    )

    print(f"Rectified canvas: {rectified.shape[1]}x{rectified.shape[0]}\n")

    for (cx, cy, cw, ch), members in clusters:
        print(f"--- Region {members} = [x={cx}, y={cy}, w={cw}, h={ch}] ---")
        region = rectified[cy:cy + ch, cx:cx + cw]
        results = pipeline.ocr_engine.read_layout(region, mode="auto")
        if not results:
            print("  (NO detections at all in this region)")
            continue
        for poly, text, score in results:
            pts = np.array(poly)
            local_cx, local_cy = float(np.mean(pts[:, 0])), float(np.mean(pts[:, 1]))
            canvas_cx, canvas_cy = local_cx + cx, local_cy + cy
            matched_field = None
            for name in members:
                fx, fy, fw, fh = free_text_boxes[name]
                if fx <= canvas_cx <= fx + fw and fy <= canvas_cy <= fy + fh:
                    matched_field = name
                    break
            print(
                f"  detection: center=({canvas_cx:.0f},{canvas_cy:.0f}) "
                f"text_len={len(text)} score={score:.2f} "
                f"matched_field={matched_field or 'NONE (falls outside every field box)'}"
            )

    print("\nCopy everything above (positions/scores/lengths only - no actual text) and paste it back.")


if __name__ == "__main__":
    main()
