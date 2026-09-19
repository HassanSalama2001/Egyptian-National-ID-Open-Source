"""
MAX_PROCESSING_SECONDS caps the combinatorial cost of the numeric
offset-retry ladder x binarization ladder x rotation search. Measured on
a real failure before this existed: a square image that misclassified
front-as-back took 175s in a single process_image() call - a denial-of-
service surface on a public API/MCP endpoint.

These are fast, deterministic unit tests of the gating logic itself (an
already-expired deadline), not slow end-to-end timing runs - those live
in scripts/benchmark_own.py, which is where "is the cap set high enough
to not regress a real scan" actually gets decided.
"""
import time

import numpy as np
import pytest

from egyptian_national_id_ocr.core.pipeline import Pipeline


@pytest.fixture(scope="module")
def pipeline():
    return Pipeline()


class TestRetryNumericAtOffsetsRespectsDeadline:
    def test_already_expired_deadline_tries_nothing(self, pipeline):
        # A deadline in the past should stop the FIRST offset from being
        # tried at all - a fast, cheap way to prove the check fires
        # without needing a crop that genuinely takes 175s to search.
        crop = np.zeros((80, 200, 3), dtype=np.uint8)
        box = [100, 100, 200, 80]
        source = np.zeros((750, 1200, 3), dtype=np.uint8)
        expired = time.time() - 1.0

        start = time.time()
        text, confidence = pipeline._retry_numeric_at_offsets(
            source, box, expected_length=14, separators=None, deadline=expired
        )
        elapsed = time.time() - start

        assert text == ""
        assert confidence == 0.0
        assert elapsed < 1.0, "an expired deadline should skip the search entirely, not run it"

    def test_no_deadline_means_unbounded_behavior_is_unchanged(self, pipeline):
        # None is the default every existing caller (tests, scripts) that
        # doesn't know about the budget gets - must not silently start
        # skipping offsets for them.
        source = np.zeros((750, 1200, 3), dtype=np.uint8)
        box = [100, 100, 200, 80]
        # Should not raise, and should behave identically to omitting
        # `deadline` altogether (it defaults to None).
        pipeline._retry_numeric_at_offsets(
            source, box, expected_length=14, separators=None, deadline=None
        )


class TestProcessImageReportsWhenTimeLimited:
    def test_deadline_hit_between_rotations_adds_an_honest_message(self, pipeline, monkeypatch):
        # Force every orientation attempt to look like it already blew
        # the budget, without waiting 45 real seconds for it to happen.
        monkeypatch.setattr(Pipeline, "MAX_PROCESSING_SECONDS", 0)
        image = np.zeros((750, 1200, 3), dtype=np.uint8)
        result = pipeline.process_image(image)
        assert any("time limit" in m.lower() for m in result.messages), (
            "a time-limited result should say so, not silently return a "
            "partial answer with no indication anything was cut short"
        )
