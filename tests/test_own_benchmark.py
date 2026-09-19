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
    ("back-greyscale.jpeg", "expiry_date"):
        "day digits unreadable; degrades to YYYY/MM via the 7-year rule",
    ("back-greyscale.jpeg", "profession"):
        "last word not proposed by the detector at this scan resolution",
    ("back-eco.jpeg", "profession"):
        "last word not proposed by the detector at this scan resolution",
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
