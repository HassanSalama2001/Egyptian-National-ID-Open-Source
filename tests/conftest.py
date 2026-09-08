import pytest
from pathlib import Path
import cv2
import numpy as np

@pytest.fixture
def assets_dir():
    return Path(__file__).parent.parent / "assets"

@pytest.fixture
def front_image_path(assets_dir):
    return assets_dir / "front.jpg"

@pytest.fixture
def back_image_path(assets_dir):
    return assets_dir / "back.jpg"

@pytest.fixture
def template_image_path(assets_dir):
    return assets_dir / "Egyptian_ID_Card.jpg"

@pytest.fixture
def sample_image():
    # Return a dummy blank image for general testing
    return np.zeros((1000, 1000, 3), dtype=np.uint8)
