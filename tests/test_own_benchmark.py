"""
Runs every scan variant of the maintainer's own real card and asserts each
field still reads correctly - the accuracy regression net for changes to
alignment, cropping, OCR or post-processing.

The card is a real identity document. Its images and expected values live
in assets/own_benchmark/, excluded wholesale by .gitignore, and nothing
here hard-codes a value from it: expectations are loaded at runtime and
assertion messages name only the FIELD that broke, never what it holds.
On a clone without that folder every test skips, so the public suite
still passes.

Known-failing variants are listed in KNOWN_GAPS rather than deleted, so
they keep running and announce themselves the moment they start passing.
"""
import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from benchmark_own import (  # noqa: E402
    BACK_FIELDS,
    BACK_VARIANTS,
    BENCHMARK_DIR,
    EXPECTED_PATH,
    FRONT_FIELDS,
    FRONT_VARIANTS,
    load,
    value_of,
)

# (variant, field) pairs known not to read yet, with the reason. These
# xfail instead of failing the suite - but they are NOT skipped, so if a
# change fixes one, pytest reports XPASS and the entry should be removed.
KNOWN_GAPS = {
    # back-greyscale expiry_date/profession and back-eco profession were
    # listed here under align_card; all three read correctly since the
    # CardPreprocessor crop and were removed.
    ("back.jpg", "profession"):
        "same length, one character different since the CardPreprocessor "
        "crop; recogniser noise on a card cut off at the photo's top edge",
    ("front-enhanced.jpeg", "address"):
        "one extra character (39 vs 38); recogniser noise, not truncation",
    ("front-greyscale.jpeg", "address"):
        "one extra character (39 vs 38); recogniser noise, not truncation",
    ("front-eco.jpeg", "address"):
        "heaviest filter; several characters wrong",
}

pytestmark = pytest.mark.skipif(
    not EXPECTED_PATH.exists(),
    reason="private benchmark not present on this machine",
)


@pytest.fixture(scope="module")
def expected():
    return json.loads(EXPECTED_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def pipeline():
    from egyptian_national_id_ocr.core.pipeline import Pipeline
    return Pipeline()


@pytest.fixture(scope="module")
def readings(pipeline):
    """Each variant processed once; tests then assert per field."""
    out = {}
    for name in FRONT_VARIANTS + BACK_VARIANTS:
        path = BENCHMARK_DIR / name
        if path.exists():
            out[name] = pipeline.process_image(load(path))
    return out


def _check(readings, expected, variant, side, field, request):
    if variant not in readings:
        pytest.skip(f"{variant} not present")
    want = (expected[side].get(field) or "").strip()
    if not want:
        pytest.skip(f"no baseline recorded for {side}.{field}")
    got = value_of(getattr(readings[variant], side), field)
    if got != want and (variant, field) in KNOWN_GAPS:
        pytest.xfail(KNOWN_GAPS[(variant, field)])
    assert got == want, (
        f"{variant}: {side}.{field} does not match the baseline "
        f"(got {len(got)} chars, expected {len(want)})"
    )
    if (variant, field) in KNOWN_GAPS:
        pytest.fail(
            f"{variant}: {side}.{field} now PASSES - remove it from KNOWN_GAPS"
        )


@pytest.mark.parametrize("variant", FRONT_VARIANTS)
@pytest.mark.parametrize("field", FRONT_FIELDS)
def test_front_variant_field(readings, expected, variant, field, request):
    _check(readings, expected, variant, "front", field, request)


@pytest.mark.parametrize("variant", BACK_VARIANTS)
@pytest.mark.parametrize("field", BACK_FIELDS)
def test_back_variant_field(readings, expected, variant, field, request):
    _check(readings, expected, variant, "back", field, request)


# The -enhanced/-greyscale/-eco variants of each side are the SAME scan
# saved through different scanner filters, so the card sits at the same
# photo pixels in all three. That makes the best-contrast variant a real
# ground truth for where the others' crops must land - a crop regression
# test on real scans, not only synthetic ones. Before this check existed
# the eco back crop was ~20px inside the card on two sides (its top edge
# is invisible under that filter) and nothing flagged it.
SAME_SCAN_GROUPS = [FRONT_VARIANTS[1:], BACK_VARIANTS[1:]]
SAME_SCAN_TOLERANCE_PX = 6


@pytest.mark.parametrize("group", SAME_SCAN_GROUPS, ids=["front", "back"])
def test_filter_variants_of_one_scan_crop_to_the_same_place(group):
    import cv2
    import numpy as np
    from egyptian_national_id_ocr.core.card_preprocessor import CardPreprocessor, _CANON

    pre = CardPreprocessor()
    quads = {}
    for name in group:
        path = BENCHMARK_DIR / name
        if not path.exists():
            pytest.skip(f"{name} not present")
        M = pre.prepare(load(path))[0].M
        quads[name] = cv2.perspectiveTransform(
            _CANON.reshape(-1, 1, 2), np.linalg.inv(M)).reshape(-1, 2)
    ref_name = group[0]
    for name in group[1:]:
        worst = float(np.linalg.norm(quads[name] - quads[ref_name], axis=1).max())
        assert worst <= SAME_SCAN_TOLERANCE_PX, (
            f"{name}: card corner {worst:.1f}px from where {ref_name} puts it"
        )
