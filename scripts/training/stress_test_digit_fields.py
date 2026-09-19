"""
Measures the digit classifier END TO END on real field crops - national_id
and date_of_birth - across the capture conditions a real card actually
meets, rather than on isolated pre-segmented glyphs like
stress_test_digit_classifier.py does.

That distinction is the point. The isolated-glyph benchmark reports ~85%
per-digit accuracy, which says nothing about whether the 14 digits were
SEGMENTED correctly in the first place. Scanning one real card four ways
(normal, "enhanced", greyscale, "eco") produced 11, 12 and 14 digits for
the same 14-digit number - a wrong count is worse than a wrong digit,
because it shifts every position after it and leaves the checksum no
chance of repairing anything.

Conditions mirror those four real scans: a clean render, a washed-out
low-contrast capture, a greyscale scan, and a hard black-and-white "eco"
filter.

Usage:
    python scripts/training/stress_test_digit_fields.py --count 40
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from generate_trial_ids import generate_sample
from egyptian_national_id_ocr.core.layout_analyzer import LayoutAnalyzer
from egyptian_national_id_ocr.ocr.digit_classifier_engine import DigitClassifierEngine

NUMERIC_FIELD_LENGTHS = {
    "national_id": 14,
    "birth_date": 8,
    "issue_date": 6,     # YYYY/MM
    "expiry_date": 8,    # YYYY/MM/DD
}
NUMERIC_FIELD_SEPARATORS = {
    "birth_date": [4, 7],
    "issue_date": [4],
    "expiry_date": [4, 7],
}

# (field on the card, key in the generator's ground truth)
FRONT_NUMERIC = [("national_id", "national_id"), ("birth_date", "date_of_birth")]
# The back side was NOT covered here originally, which meant every
# "accuracy" figure this script produced described the front alone. Real
# back scans then turned out to degrade much more under greyscale/eco
# filters than the front does - invisible to a front-only benchmark.
BACK_NUMERIC = [
    ("national_id", "national_id"),
    ("issue_date", "issue_date"),
    ("expiry_date", "expiry_date"),
]


def _scan_chain(img: np.ndarray, filter_name: str, scale: float = 0.30,
                jpeg_quality: int = 70) -> np.ndarray:
    """Models the whole capture chain a phone scanner app puts a card
    through, not just its colour filter.

    This matters more than the filter itself. In a real "zoomed out" scan
    the card occupies a small part of a page, so the pipeline is
    UPSCALING a small region back to 1200x750 - the digits arrive soft
    before any filter is involved. Conditions that applied a filter to a
    full-resolution render skipped that entirely, which is why they
    scored 93-100% on greyscale backs while real greyscale backs failed.

    Order matters and mirrors the real device: the page is captured and
    filtered at scan resolution, and the card is simply a small part of
    that page - so the filter runs on well-sampled pixels and the card
    region is what ends up under-sampled. Shrinking FIRST and filtering
    the tiny image was tried and is far harsher than reality: it drove
    the front national_id to 6.5% per-digit accuracy, where a real eco
    scan of the same card misread a single digit.
    """
    filtered = apply_condition(img, filter_name)
    small = cv2.resize(filtered, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    ok, buffer = cv2.imencode(".jpg", small, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality])
    if not ok:
        return filtered
    return cv2.imdecode(buffer, cv2.IMREAD_COLOR)


def apply_condition(img: np.ndarray, condition: str) -> np.ndarray:
    if condition == "clean":
        return img
    # Scan-chain variants, named after the four real scans they model.
    if condition == "normal_scan":
        return _scan_chain(img, "clean")
    if condition == "enhanced_scan":
        return _scan_chain(img, "enhanced")
    if condition == "greyscale_scan":
        return _scan_chain(img, "greyscale")
    if condition == "eco_scan":
        return _scan_chain(img, "eco")
    if condition == "enhanced":
        # A scanner app's "enhanced" mode: contrast and brightness pushed
        # up to make print pop off the page.
        return cv2.convertScaleAbs(img, alpha=1.35, beta=10)
    if condition == "low_contrast":
        # Washed out, as a phone photo under flat indoor light
        return cv2.convertScaleAbs(img, alpha=0.45, beta=90)
    if condition == "greyscale":
        return cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
    if condition == "eco":
        # A scanner app's "eco"/black-and-white filter, which uses LOCAL
        # adaptive binarization - it keeps text legible while flattening
        # the background. Not a global Otsu cut: that swallows the card's
        # watermark and photo into solid black and destroys the digits,
        # which is a harsher input than any real scanner produces (kept
        # below as "harsh_binary" so the extreme is still measured, just
        # not mistaken for the ordinary eco case).
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 10
        )
        return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    if condition == "eco_thin":
        # Closer to what a real scanner-app "eco" filter produces than the
        # plain adaptive threshold above: local (Sauvola) binarization
        # followed by slight stroke erosion. Added because the adaptive
        # "eco" condition scored 10/10 while a REAL eco scan of the same
        # card misread two digits - so that condition plainly wasn't
        # reproducing the real failure. The errors on the real scan
        # (٣->٢ and ٥->٠) are both cases where a thin distinguishing
        # stroke disappears, which is exactly what erosion models.
        from skimage.filters import threshold_sauvola
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        local = threshold_sauvola(gray, window_size=25, k=0.2)
        binary = ((gray > local) * 255).astype(np.uint8)
        binary = cv2.erode(binary, np.ones((2, 2), np.uint8), iterations=1)
        return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    if condition == "harsh_binary":
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    raise ValueError(condition)


def run(count: int, tmp_dir: Path, seed: int) -> None:
    # Seeded so two runs generate the SAME cards. Without this each run
    # drew different random cards, so a 20-card batch moved +/-10% on
    # nothing but luck - which is not enough resolution to tell whether a
    # preprocessing change helped or hurt.
    import random as _random
    _random.seed(seed)

    analyzer = LayoutAnalyzer()
    engine = DigitClassifierEngine()

    # The *_scan conditions are the ones that mirror real captures (see
    # _scan_chain); the bare filters are kept as controls, to separate
    # "this filter is hard" from "low resolution is hard".
    conditions = [
        "clean",
        "normal_scan",
        "enhanced_scan",
        "greyscale_scan",
        "eco_scan",
        "greyscale",
        "eco",
    ]
    labels = [f"front.{f}" for f, _ in FRONT_NUMERIC] + [f"back.{f}" for f, _ in BACK_NUMERIC]
    stats = {
        c: {label: {"exact": 0, "right_count": 0, "digit_hits": 0, "digit_total": 0}
            for label in labels}
        for c in conditions
    }

    def load(path: Path) -> np.ndarray:
        return cv2.cvtColor(np.array(Image.open(path).convert("RGB")), cv2.COLOR_RGB2BGR)

    for i in range(1, count + 1):
        entry = generate_sample(i, tmp_dir)
        truth = entry["data"]
        front_card = load(tmp_dir / entry["filename"])
        back_card = load(tmp_dir / entry["back_filename"])

        for condition in conditions:
            for card, fields, crop_fn, prefix in (
                (front_card, FRONT_NUMERIC, analyzer.get_field_crops, "front"),
                (back_card, BACK_NUMERIC, analyzer.get_back_field_crops, "back"),
            ):
                rectified, _ = analyzer.align_card(apply_condition(card, condition))
                crops = crop_fn(rectified)

                for field, truth_key in fields:
                    expected = "".join(ch for ch in truth[truth_key] if ch.isdigit())
                    got = engine.extract_digits(
                        crops[field],
                        expected_length=NUMERIC_FIELD_LENGTHS[field],
                        separator_positions=NUMERIC_FIELD_SEPARATORS.get(field),
                    )
                    s = stats[condition][f"{prefix}.{field}"]
                    if got == expected:
                        s["exact"] += 1
                    if len(got) == len(expected):
                        s["right_count"] += 1
                        s["digit_hits"] += sum(1 for a, b in zip(got, expected) if a == b)
                    s["digit_total"] += len(expected)

    print(f"\nDigit field accuracy over {count} synthetic cards per condition")
    print("-" * 78)
    print(f"{'condition':14s} {'field':18s} {'exact':>8s} {'right len':>10s} {'per-digit':>11s}")
    print("-" * 78)
    for condition in conditions:
        for label in labels:
            s = stats[condition][label]
            exact = s["exact"] / count * 100
            right_len = s["right_count"] / count * 100
            per_digit = s["digit_hits"] / s["digit_total"] * 100 if s["digit_total"] else 0.0
            print(f"{condition:14s} {label:18s} {exact:7.1f}% {right_len:9.1f}% {per_digit:10.1f}%")
        print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=40)
    parser.add_argument("--seed", type=int, default=4242)
    args = parser.parse_args()

    tmp_dir = PROJECT_ROOT / "data" / "_stress_digit_fields_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    try:
        run(args.count, tmp_dir, args.seed)
    finally:
        import shutil
        shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
