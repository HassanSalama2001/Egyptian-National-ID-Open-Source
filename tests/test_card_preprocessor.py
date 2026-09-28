"""
Crop accuracy of CardPreprocessor on synthetic scenes with a KNOWN card
placement - one per failure type seen on real third-party scans (thin
background margin, card small and sideways on a page, upside down,
negative scan, angled on a textured background). Uses the same scene
builder and corner-error metric as scripts/training/benchmark_card_crop.py,
which is the place to look at the full numbers; this only guards against
regressions. No real card is used.
"""
import random
import sys
from pathlib import Path

import cv2
import pytest

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "training"))

import benchmark_card_crop as bench  # noqa: E402
from egyptian_national_id_ocr.core.card_preprocessor import CardPreprocessor  # noqa: E402


@pytest.fixture(scope="module")
def preprocessor():
    return CardPreprocessor()


@pytest.fixture(scope="module")
def cards(synthetic_card_sample):
    return (cv2.imread(str(synthetic_card_sample["front_path"])),
            cv2.imread(str(synthetic_card_sample["back_path"])))


@pytest.mark.parametrize("condition", list(bench.CONDITIONS))
@pytest.mark.parametrize("side", [0, 1], ids=["front", "back"])
def test_card_is_cropped_within_tolerance(preprocessor, cards, condition, side):
    rng = random.Random(f"test-{condition}-{side}")
    photo, H = bench.make_scene(cards[side], condition, rng)
    first = preprocessor.prepare(photo)[0]
    err = bench.corner_error(first.M, H)
    assert err <= bench.PASS_PX, f"{condition}: corners {err:.1f}px off in 1200x750 space"


def test_negative_scan_is_flagged_and_corrected(preprocessor, cards):
    photo, _ = bench.make_scene(cards[0], "inverted_whole", random.Random("neg"))
    first = preprocessor.prepare(photo)[0]
    assert first.inverted
    assert first.image.mean() > 127  # the card reads dark-on-light again


def test_exact_crop_is_not_trimmed(preprocessor, cards):
    """A photo that already IS the card must not lose its edges."""
    photo, H = bench.make_scene(cards[1], "exact_tight", random.Random("exact"))
    assert bench.corner_error(preprocessor.prepare(photo)[0].M, H) <= 2.0
