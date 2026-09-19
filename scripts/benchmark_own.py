"""
Accuracy + speed regression run against the maintainer's own real ID card.

The card is a real identity document, so nothing about it lives in this
file. The images and their expected values sit in assets/own_benchmark/,
which .gitignore excludes wholesale; this script only ever PRINTS field
names and OK/WRONG, never a value. Running it on a clone without that
folder prints a skip notice and exits 0.

    python scripts/benchmark_own.py                 # check every variant
    python scripts/benchmark_own.py --write-expected  # re-derive baseline

--write-expected reads the two scans confirmed correct by eye (front.jpg
and back.jpg, the un-zoomed originals) and records their output as the
expected values for every other variant. Only re-run it when the card or
the reference scans change - never to paper over a regression, which is
exactly what it would silently do.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from egyptian_national_id_ocr.core.pipeline import Pipeline  # noqa: E402

BENCHMARK_DIR = PROJECT_ROOT / "assets" / "own_benchmark"
EXPECTED_PATH = BENCHMARK_DIR / "expected.json"

FRONT_VARIANTS = ["front.jpg", "front-enhanced.jpeg", "front-greyscale.jpeg", "front-eco.jpeg"]
BACK_VARIANTS = ["back.jpg", "back-enhanced.jpeg", "back-greyscale.jpeg", "back-eco.jpeg"]

FRONT_FIELDS = ["national_id", "first_name", "full_name", "address", "card_serial_number"]
BACK_FIELDS = ["national_id", "issue_date", "expiry_date", "profession",
               "gender", "religion", "marital_status"]


def load(path: Path) -> np.ndarray:
    pil = ImageOps.exif_transpose(Image.open(path))
    return cv2.cvtColor(np.array(pil.convert("RGB")), cv2.COLOR_RGB2BGR)


def value_of(obj, field: str) -> str:
    got = getattr(obj, field, None) if obj is not None else None
    got = got.value if hasattr(got, "value") else got
    return (got or "").strip()


def read_side(pipe: Pipeline, path: Path):
    started = time.time()
    result = pipe.process_image(load(path))
    return result, time.time() - started


def write_expected(pipe: Pipeline) -> None:
    front, _ = read_side(pipe, BENCHMARK_DIR / "front.jpg")
    back, _ = read_side(pipe, BENCHMARK_DIR / "back.jpg")
    expected = {
        "front": {f: value_of(front.front, f) for f in FRONT_FIELDS},
        "back": {f: value_of(back.back, f) for f in BACK_FIELDS},
    }
    EXPECTED_PATH.write_text(json.dumps(expected, ensure_ascii=False, indent=2), encoding="utf-8")
    filled = sum(1 for v in expected["front"].values() if v) + \
             sum(1 for v in expected["back"].values() if v)
    print(f"Wrote {EXPECTED_PATH.name}: {filled}/"
          f"{len(FRONT_FIELDS) + len(BACK_FIELDS)} fields populated.")
    print("Field VALUES are not printed here - open the file yourself to verify them.")


def run(pipe: Pipeline) -> int:
    expected = json.loads(EXPECTED_PATH.read_text(encoding="utf-8"))
    failures = 0
    for side, variants, fields in (
        ("front", FRONT_VARIANTS, FRONT_FIELDS),
        ("back", BACK_VARIANTS, BACK_FIELDS),
    ):
        print(f"\n--- {side} ---")
        for name in variants:
            path = BENCHMARK_DIR / name
            if not path.exists():
                print(f"  {name:26s} MISSING")
                continue
            result, seconds = read_side(pipe, path)
            got_side = getattr(result, side)
            marks, wrong = [], []
            for field in fields:
                want = (expected[side].get(field) or "").strip()
                if not want:
                    continue
                if value_of(got_side, field) == want:
                    marks.append(field)
                else:
                    wrong.append(field)
                    failures += 1
            status = "OK" if not wrong else "FAIL"
            print(f"  {name:26s} {seconds:5.1f}s  {len(marks)}/{len(marks) + len(wrong)} "
                  f"{status}" + (f"  wrong: {', '.join(wrong)}" if wrong else ""))
    print(f"\n{failures} field(s) wrong." if failures else "\nAll fields match.")
    return 1 if failures else 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write-expected", action="store_true")
    args = parser.parse_args()

    if not BENCHMARK_DIR.exists():
        print(f"SKIP: {BENCHMARK_DIR} not present (private benchmark, not distributed).")
        return 0
    pipe = Pipeline()
    if args.write_expected:
        write_expected(pipe)
        return 0
    if not EXPECTED_PATH.exists():
        print(f"SKIP: {EXPECTED_PATH.name} missing - run with --write-expected first.")
        return 0
    return run(pipe)


if __name__ == "__main__":
    raise SystemExit(main())
