import sys
import pytest
from pathlib import Path
import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent

@pytest.fixture
def assets_dir():
    return PROJECT_ROOT / "assets"

@pytest.fixture
def template_image_path(assets_dir):
    return assets_dir / "Egyptian_ID_Card.jpg"

@pytest.fixture
def sample_image():
    # Return a dummy blank image for general testing
    return np.zeros((1000, 1000, 3), dtype=np.uint8)

@pytest.fixture(scope="session")
def synthetic_card_sample(tmp_path_factory):
    """
    A well-formed synthetic card with known ground truth, generated at a
    fixed seed - not a real ID card, and not the old assets/dataset/
    (720 unlabelled files that predated field-box calibration against
    assets/front_template.jpg; retired rather than fixed - see the
    session history in CHANGELOG.md for why, and
    scripts/benchmark_own.py for what replaced it as the real accuracy
    signal). Session-scoped: generated once per test run, not once per
    test.

    A real photo of Hassan's own ID (assets/front.jpg) was previously
    used as a test fixture here and has been scrubbed from the repo/
    history entirely - never reintroduce a real card image as a test
    fixture.
    """
    import random
    sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "training"))
    from generate_trial_ids import generate_sample

    # generate_sample draws from the module-level `random` directly and
    # is NOT seeded on its own - only the CLI's main() seeds it (--seed).
    # Seed explicitly here so this fixture is actually reproducible
    # across runs, not just consistent within one run by accident of
    # whatever the interpreter's default entropy happened to be.
    random.seed(42)
    out_dir = tmp_path_factory.mktemp("conftest_synthetic_card")
    result = generate_sample(0, out_dir, data=None)
    return {
        "front_path": out_dir / result["filename"],
        "back_path": out_dir / result["back_filename"],
        "data": result["data"],
    }

@pytest.fixture
def front_image_path(synthetic_card_sample):
    return synthetic_card_sample["front_path"]

@pytest.fixture
def back_image_path(synthetic_card_sample):
    return synthetic_card_sample["back_path"]
