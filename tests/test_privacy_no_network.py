"""
Proves the project's central privacy claim - "extraction happens
entirely on your machine; no card image, and no extracted field, is
sent anywhere" (README, docs/LIMITATIONS.md) - rather than just stating
it. Blocks real outbound network connections during a pipeline run and
asserts none are attempted.

Loopback/AF_UNIX connections are allowed: paddle can use local IPC for
its own multiprocessing, which has nothing to do with the privacy claim
this test exists to check (whether card data leaves the machine to a
REMOTE host). Model downloads are a separate, one-time, explicit setup
step (`egy-nid-ocr doctor` / first PaddleOCR init) - out of scope here,
which specifically covers a `process_image()` call, the thing that
happens on every extraction of every card.
"""
import socket
import sys
from pathlib import Path
from unittest.mock import patch

import cv2
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent

LOOPBACK_HOSTS = {"127.0.0.1", "::1", "localhost", "0.0.0.0"}


def _is_outbound(address) -> bool:
    """True for a real network destination; False for loopback/local IPC
    (AF_UNIX addresses are plain strings, not (host, port) tuples)."""
    if not isinstance(address, tuple) or not address:
        return False
    host = address[0]
    return host not in LOOPBACK_HOSTS


@pytest.fixture(scope="module")
def synthetic_card(tmp_path_factory):
    """A quick, well-formed synthetic front - not assets/dataset/ID0.png,
    which is a degenerate fixture that takes ~50s under the new time
    budget for reasons unrelated to what this test checks."""
    sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "training"))
    from generate_trial_ids import generate_sample

    out_dir = tmp_path_factory.mktemp("privacy_test_card")
    result = generate_sample(0, out_dir)
    return cv2.imread(str(out_dir / result["filename"]))


@pytest.fixture(scope="module")
def pipeline():
    from egyptian_national_id_ocr.core.pipeline import Pipeline
    return Pipeline()


def test_process_image_attempts_no_outbound_network_connection(pipeline, synthetic_card):
    attempted = []
    original_connect = socket.socket.connect

    def guarded_connect(self, address):
        if _is_outbound(address):
            attempted.append(address)
            raise AssertionError(f"process_image attempted an outbound connection to {address}")
        return original_connect(self, address)

    with patch.object(socket.socket, "connect", guarded_connect):
        result = pipeline.process_image(synthetic_card)

    assert not attempted, f"outbound connection(s) attempted during extraction: {attempted}"
    # Also confirms the patch didn't just silently no-op - a real
    # extraction ran and returned a real result under it.
    assert result is not None
    assert result.status is not None
