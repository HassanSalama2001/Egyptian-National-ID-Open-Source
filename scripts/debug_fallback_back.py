"""
Privacy-preserving diagnostic: tests the heavy-upscale+CLAHE OCR fallback
(the same one that rescues fields like first_name/gender when the
clustered pass finds nothing) directly against a field's own isolated
crop. Only prints position/length/score - never the actual recognized
text - so it's safe to paste back.

Usage:
    .audit_venv\\Scripts\\python.exe scripts\\debug_fallback_back.py "path\\to\\back.jpg" marital_status
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
from egyptian_national_id_ocr.postprocessing.enum_matcher import match_enum, GENDER_VALUES, RELIGION_VALUES, MARITAL_STATUS_VALUES

ENUM_TABLES = {
    "gender": GENDER_VALUES,
    "religion": RELIGION_VALUES,
    "marital_status": MARITAL_STATUS_VALUES,
}


def load_image_exif_safe(path: str) -> np.ndarray:
    pil_image = Image.open(path)
    pil_image = ImageOps.exif_transpose(pil_image)
    return cv2.cvtColor(np.array(pil_image.convert("RGB")), cv2.COLOR_RGB2BGR)


def main() -> None:
    if len(sys.argv) < 3:
        print("Usage: debug_fallback_back.py <path-to-back-photo> <field_name>")
        sys.exit(1)

    field_name = sys.argv[2]
    image = load_image_exif_safe(sys.argv[1])
    analyzer = LayoutAnalyzer()
    rectified, _ = analyzer.align_card(image)
    crop = analyzer.get_field_crops(rectified, fields=analyzer.BACK_FIELDS)[field_name]
    print(f"Field: {field_name}  crop shape: {crop.shape}")

    pipeline = Pipeline(ocr_engine=PaddleOCREngine())

    print("\n--- Plain OCR on isolated crop (no upscale) ---")
    results = pipeline.ocr_engine.read_layout(crop, mode="auto")
    if not results:
        print("  (no detections)")
    for poly, text, score in results:
        print(f"  text_len={len(text)} score={score:.2f}")

    print("\n--- Heavy fallback (4x upscale + CLAHE, current production config) ---")
    raw_text, confidence = pipeline._ocr_field_fallback(crop)
    print(f"  result_len={len(raw_text)} confidence={confidence:.2f} empty={not raw_text}")

    print("\n--- Trying other preprocessing variants (diagnostic only) ---")
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    variants = []
    for scale in (2, 3, 4, 6, 8):
        for use_clahe in (False, True):
            for pad in (10, 30, 50):
                up = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_LANCZOS4)
                if use_clahe:
                    ug = cv2.cvtColor(up, cv2.COLOR_BGR2GRAY)
                    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                    ug = clahe.apply(ug)
                    up = cv2.cvtColor(ug, cv2.COLOR_GRAY2BGR)
                padded = cv2.copyMakeBorder(up, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
                results = pipeline.ocr_engine.read_layout(padded, mode="auto")
                if results:
                    text = " ".join(r[1] for r in results)
                    score = min(r[2] for r in results)
                    variants.append((scale, use_clahe, pad, len(text), score))

    if not variants:
        print("  NONE of the tried variants found any text at all.")
    else:
        print(f"  {len(variants)} variant(s) found something:")
        for scale, use_clahe, pad, tlen, score in variants:
            print(f"    scale={scale}x clahe={use_clahe} pad={pad}: text_len={tlen} score={score:.2f}")

    if field_name in ENUM_TABLES and raw_text:
        matched, match_confidence = match_enum(raw_text, ENUM_TABLES[field_name])
        print(f"\n--- Fuzzy-match against known {field_name} values (current threshold) ---")
        print(f"  matched={'YES' if matched else 'NO'} match_confidence={match_confidence:.2f}")
        # Also check with an intentionally permissive threshold, purely to
        # report HOW close the closest value actually was - this does not
        # change what the live pipeline accepts, just tells us whether a
        # further threshold increase could plausibly help.
        _, permissive_confidence = match_enum(raw_text, ENUM_TABLES[field_name], max_distance_ratio=1.0)
        print(f"  (closest-possible-value confidence if threshold were removed: {permissive_confidence:.2f})")

    print("\nCopy everything above (lengths/scores only - no actual text) and paste it back.")


if __name__ == "__main__":
    main()
