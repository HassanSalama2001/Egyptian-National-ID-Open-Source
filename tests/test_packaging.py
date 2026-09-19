"""
Proves the built WHEEL is installable and usable on its own - not that the
source tree works (every other test already covers that).

Building a wheel with setuptools' default file-finder collects .py files
only. That silently dropped ocr/models/digit_classifier_svm.joblib and
detection/templates/*.jpg from the package: `pip install` reported
success, and the first scan then raised "Digit classifier model not
found" for every numeric field (national_id, serial_number, both dates).
Nothing in the unit/integration suite would have caught this, since they
all run against the source tree, where the files are simply on disk next
to the code regardless of what the wheel would contain.

Slow (builds a wheel, spins up a venv, installs full dependencies) and
opt-in: skipped unless RUN_PACKAGING_TEST=1, so it doesn't slow down the
normal test run or require network access in every CI job. Run it
explicitly before a release:

    RUN_PACKAGING_TEST=1 pytest tests/test_packaging.py -v -s
"""
import os
import subprocess
import sys
import venv
import zipfile
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_PACKAGING_TEST") != "1",
    reason="slow (builds a wheel + fresh venv); set RUN_PACKAGING_TEST=1 to run",
)


@pytest.fixture(scope="module")
def built_wheel(tmp_path_factory):
    out_dir = tmp_path_factory.mktemp("wheel")
    subprocess.run(
        [sys.executable, "-m", "pip", "wheel", str(PROJECT_ROOT),
         "--no-deps", "-w", str(out_dir)],
        check=True, capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    wheels = list(out_dir.glob("*.whl"))
    assert len(wheels) == 1, f"expected exactly one wheel, got {wheels}"
    return wheels[0]


def test_wheel_contains_the_runtime_model_and_templates(built_wheel):
    with zipfile.ZipFile(built_wheel) as z:
        names = z.namelist()
    models = [n for n in names if n.endswith(".joblib")]
    templates = [n for n in names if "templates/" in n and n.endswith(".jpg")]
    assert models, "digit_classifier_svm.joblib missing from wheel - check [tool.setuptools.package-data]"
    assert len(templates) == 2, f"expected 2 detection templates in wheel, found {templates}"


@pytest.fixture(scope="module")
def clean_venv(tmp_path_factory, built_wheel):
    """A venv with the wheel installed and full deps - source tree not on sys.path."""
    venv_dir = tmp_path_factory.mktemp("venv")
    venv.create(venv_dir, with_pip=True)
    python = venv_dir / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    subprocess.run(
        [str(python), "-m", "pip", "install", "-q", str(built_wheel)],
        check=True, capture_output=True, text=True, timeout=900,
        encoding="utf-8", errors="replace",
    )
    return python


def test_installed_package_loads_its_model_with_no_source_tree(clean_venv):
    result = subprocess.run(
        [str(clean_venv), "-c",
         "from egyptian_national_id_ocr.ocr.digit_classifier_engine import DigitClassifierEngine; "
         "DigitClassifierEngine(); print('OK')"],
        capture_output=True, text=True, cwd=str(clean_venv.parent),  # NOT the project root
        encoding="utf-8", errors="replace",
    )
    assert "OK" in result.stdout, (
        f"installed package could not load its own model:\n{result.stderr}"
    )


def test_installed_cli_entry_point_works(clean_venv, tmp_path):
    # clean_venv IS the python.exe/python path (see the fixture above),
    # so its parent is already the Scripts/bin directory - no need to
    # append it again. Caught by actually running this in CI: the
    # doubled path (.../Scripts/Scripts/egy-nid-ocr.exe) doesn't exist,
    # so the test correctly failed rather than silently passing on a
    # wrong path.
    bin_dir = clean_venv.parent
    cli = bin_dir / ("egy-nid-ocr.exe" if os.name == "nt" else "egy-nid-ocr")
    assert cli.exists(), f"entry point script not installed at {cli}"

    # A fresh seeded synthetic card, not a fixed file on disk - this
    # test previously pointed at assets/dataset/ID0.png, which got
    # deleted (see CHANGELOG.md, Phase 3) without this reference being
    # updated: caught by actually running this in CI, where it silently
    # SKIPPED instead of testing anything. Skipping was the right
    # behavior for a missing file, but the file should never have gone
    # missing out from under an unrelated test in the first place.
    sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "training"))
    from generate_trial_ids import generate_sample

    generated = generate_sample(0, tmp_path)
    sample = tmp_path / generated["filename"]
    assert sample.exists(), f"generator did not produce {sample}"

    result = subprocess.run(
        [str(cli), "extract", str(sample), "--output", "json"],
        capture_output=True, text=True, cwd=str(clean_venv.parent),
        timeout=300,
        # The CLI's whole subject is Arabic text - decoding its captured
        # stdout with Windows' locale-default codepage (cp1252) instead
        # of UTF-8 crashed this exact subprocess.run() with
        # UnicodeDecodeError when this test was actually run (the same
        # class of bug already fixed on the CLI's WRITER side in
        # cli.py; this is the reader side).
        encoding="utf-8", errors="replace",
    )
    assert result.returncode == 0, f"installed CLI failed:\n{result.stderr}"
