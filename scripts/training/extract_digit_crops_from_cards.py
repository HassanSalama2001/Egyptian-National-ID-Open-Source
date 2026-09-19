"""
Rebuilds the digit-classifier training set from REAL segmented crops
instead of directly-rendered isolated glyphs.

Root cause this fixes: generate_digit_crops.py rendered each digit
directly onto a 64x64 canvas via PIL text layout, preserving the font's
natural (and wildly inconsistent) glyph proportions - e.g. '5' occupies
~98% of the canvas height while '0' occupies ~22%. But at inference,
DigitClassifierEngine._prepare_crop() force-normalizes every segmented
digit to fill 70% of the canvas regardless of its natural size, which
destroys exactly that size signal. A classifier trained on one
distribution and evaluated on the other was systematically confusing
digits whose *shape* (not size) looks similar once normalized (mainly
5 <-> 0). This script closes that gap by construction: extract real
digit segments from the full-card synthetic dataset (data/synthetic_both,
which has manifest.json ground truth for both sides), pass them through
the exact same _segment_digits + _prepare_crop pipeline used at
inference, and use those as training data.

Pulls from front's national_id AND back's national_id - back's fields
print at a noticeably smaller font size than front's (see
layout_analyzer.py's BACK_FIELDS), and a classifier trained only on
front-scale digits under-performed specifically on back's fields (same
distribution-mismatch lesson as above, just along a different axis:
scale instead of glyph proportion).

Deliberately excludes issue_date/expiry_date even though they're
digit fields too: those contain "/" separators, and count-matching
alone isn't a safe alignment guarantee once a separator is in the mix -
a "/" occasionally survives _segment_digits' shape filter as if it were
a digit, and if that coincides with two real digits merging into one
blob elsewhere in the same crop, the total segment count still matches
the (separator-stripped) ground-truth length while every position after
the mismatch is now off by one, silently mislabeling real digit crops
as "/". This was caught by eyeballing the extracted crops (a class
supposedly all "5" included a stray "/"), not by any count check - only
national_id (no separators, so no ambiguity) is safe for this
automated approach.

national_id's ground truth has a fixed digit count, so left-to-right
segment-to-ground-truth alignment is unambiguous - a sample is skipped
if segmentation doesn't find exactly 14 segments.
"""
import json
from pathlib import Path

import cv2
from tqdm import tqdm

from egyptian_national_id_ocr.core.layout_analyzer import LayoutAnalyzer
from egyptian_national_id_ocr.ocr.digit_classifier_engine import DigitClassifierEngine

SOURCE_DIR = Path("data/synthetic_both")
OUTPUT_DIR = Path("data/digits_v2")

# (crop-getter, field name in the crops dict, ground-truth key, digit-only)
# digit-only=True strips non-digit separators ("/") from the ground truth
# before comparing length/zipping against segments.
FRONT_SOURCES = [("national_id", "national_id", False)]
BACK_SOURCES = [("national_id", "national_id", False)]


def _extract_from_crop(crop, gt_value: str, digit_only: bool, engine, out_dir: Path, counts: dict) -> bool:
    if crop is None or crop.size == 0:
        return False
    digits_gt = gt_value.replace("/", "") if digit_only else gt_value

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    segments = engine._segment_digits(gray)
    if len(segments) != len(digits_gt):
        return False

    for (_, seg), true_digit in zip(segments, digits_gt):
        prepared = engine._prepare_crop(seg)
        idx = counts[true_digit]
        out_path = out_dir / true_digit / f"{true_digit}_{idx:05}.png"
        cv2.imwrite(str(out_path), prepared)
        counts[true_digit] += 1
    return True


def main():
    manifest = json.load(open(SOURCE_DIR / "manifest.json", encoding="utf-8"))

    analyzer = LayoutAnalyzer()
    engine = DigitClassifierEngine()  # only used for its _segment_digits/_prepare_crop helpers

    for split in ("train", "val", "test"):
        for digit in "0123456789":
            (OUTPUT_DIR / split / digit).mkdir(parents=True, exist_ok=True)

    n_total = len(manifest)
    n_train = int(n_total * 0.7)
    n_val = int(n_total * 0.85)

    counts = {d: 0 for d in "0123456789"}
    extracted_fields = 0
    skipped_fields = 0

    for i, entry in enumerate(tqdm(manifest, desc="Extracting")):
        split = "train" if i < n_train else ("val" if i < n_val else "test")
        out_dir = OUTPUT_DIR / split

        front_img = cv2.imread(str(SOURCE_DIR / entry["filename"]))
        if front_img is not None:
            rectified, _ = analyzer.align_card(front_img)
            front_crops = analyzer.get_field_crops(rectified)
            for crop_key, gt_key, digit_only in FRONT_SOURCES:
                ok = _extract_from_crop(
                    front_crops.get(crop_key), entry["data"][gt_key], digit_only, engine, out_dir, counts
                )
                extracted_fields += ok
                skipped_fields += not ok

        back_img = cv2.imread(str(SOURCE_DIR / entry["back_filename"])) if "back_filename" in entry else None
        if back_img is not None:
            rectified, _ = analyzer.align_card(back_img)
            back_crops = analyzer.get_back_field_crops(rectified)
            for crop_key, gt_key, digit_only in BACK_SOURCES:
                ok = _extract_from_crop(
                    back_crops.get(crop_key), entry["data"][gt_key], digit_only, engine, out_dir, counts
                )
                extracted_fields += ok
                skipped_fields += not ok

    print(f"Extracted {sum(counts.values())} labeled digit crops from {extracted_fields} fields "
          f"({skipped_fields} fields skipped - segmentation count mismatch)")
    print("Per-class counts:", counts)


if __name__ == "__main__":
    main()
