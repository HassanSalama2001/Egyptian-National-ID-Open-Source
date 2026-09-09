import pytest
from pathlib import Path
import cv2
import numpy as np

@pytest.fixture
def assets_dir():
    return Path(__file__).parent.parent / "assets"

@pytest.fixture
def front_image_path(assets_dir):
    # A real photo of Hassan's own ID (assets/front.jpg) was previously
    # used here and has been scrubbed from the repo/history entirely -
    # never reintroduce a real card image as a test fixture. This points
    # to a synthetic, non-identifying sample from the augmented dataset
    # instead (see assets/dataset/).
    return assets_dir / "dataset" / "ID0.png"

@pytest.fixture
def back_image_path(assets_dir):
    return assets_dir / "dataset" / "IDB0.png"

@pytest.fixture
def template_image_path(assets_dir):
    return assets_dir / "Egyptian_ID_Card.jpg"

@pytest.fixture
def sample_image():
    # Return a dummy blank image for general testing
    return np.zeros((1000, 1000, 3), dtype=np.uint8)
