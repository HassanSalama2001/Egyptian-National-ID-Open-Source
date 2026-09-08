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
digit segments from the full-card synthetic dataset (data/synthetic_front,
which has manifest.json ground truth), pass them through the exact same
_segment_digits + _prepare_crop pipeline used at inference, and use
those as training data.

Only national_id is used (fixed 14-digit length makes left-to-right
segment-to-ground-truth alignment unambiguous - a sample is skipped if
segmentation doesn't find exactly 14 segments, since we can't safely
assume alignment otherwise).
"""
import json
from pathlib import Path

import cv2
from tqdm import tqdm

from national_id_ocr.core.layout_analyzer import LayoutAnalyzer
from national_id_ocr.ocr.digit_classifier_engine import DigitClassifierEngine

SOURCE_DIR = Path("data/synthetic_front")
OUTPUT_DIR = Path("data/digits_v2")


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
    skipped = 0

    for i, entry in enumerate(tqdm(manifest, desc="Extracting")):
        split = "train" if i < n_train else ("val" if i < n_val else "test")

        img = cv2.imread(str(SOURCE_DIR / entry["filename"]))
        if img is None:
            skipped += 1
            continue

        rectified, _ = analyzer.align_card(img)
        crops = analyzer.get_field_crops(rectified)
        nid_crop = crops.get("national_id")
        if nid_crop is None or nid_crop.size == 0:
            skipped += 1
            continue

        gray = cv2.cvtColor(nid_crop, cv2.COLOR_BGR2GRAY)
        segments = engine._segment_digits(gray)

        gt_nid = entry["data"]["national_id"]
        if len(segments) != len(gt_nid):
            # Can't safely align segments to ground truth if counts differ
            skipped += 1
            continue

        for (_, seg), true_digit in zip(segments, gt_nid):
            prepared = engine._prepare_crop(seg)
            idx = counts[true_digit]
            out_path = OUTPUT_DIR / split / true_digit / f"{true_digit}_{idx:05}.png"
            cv2.imwrite(str(out_path), prepared)
            counts[true_digit] += 1

    print(f"Extracted {sum(counts.values())} labeled digit crops "
          f"({skipped} samples skipped - segmentation count mismatch)")
    print("Per-class counts:", counts)


if __name__ == "__main__":
    main()
