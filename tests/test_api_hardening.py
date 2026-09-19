"""
CORS allowlist and upload-size cap on src/app.py's /ocr endpoint.

Found in a pre-publish audit: `allow_origins=["*"]` combined with
`allow_credentials=True` lets any website's JavaScript call this API
using the visiting browser's own credentials, not just the intended
frontend - and there was no upload size limit at all, so a single
request could buffer an arbitrarily large body into memory. Both are
straightforward to get wrong again on a future edit, so both are
covered here.

Uses FastAPI's TestClient (no real server/network needed) except where
noted.
"""
import io

import pytest
from fastapi.testclient import TestClient

from egyptian_national_id_ocr.core.config import Settings, settings


class TestCorsAllowlist:
    def test_wildcard_is_not_the_default(self):
        # The precise regression this exists to catch: "*" combined with
        # allow_credentials=True is a real vulnerability, not just an
        # unusual config - any origin can then use the visiting browser's
        # own credentials against this API.
        assert "*" not in settings.cors_allowed_origins_list

    def test_dev_frontend_origin_is_allowed_by_default(self):
        # web_ui's vite dev server (see web_ui/vite.config) - the
        # documented reason this middleware exists at all.
        assert "http://localhost:5173" in settings.cors_allowed_origins_list

    def test_origins_list_parses_a_comma_separated_override(self):
        overridden = Settings(CORS_ALLOWED_ORIGINS="https://example.com, https://other.example.com")
        assert overridden.cors_allowed_origins_list == [
            "https://example.com",
            "https://other.example.com",
        ]

    def test_empty_entries_are_dropped(self):
        overridden = Settings(CORS_ALLOWED_ORIGINS="https://example.com,,  ,")
        assert overridden.cors_allowed_origins_list == ["https://example.com"]


class TestUploadSizeCap:
    @classmethod
    @pytest.fixture(scope="class")
    def client(cls):
        import app as app_module
        return TestClient(app_module.app)

    def test_oversized_upload_is_rejected_with_413(self, client, monkeypatch):
        import app as app_module
        # Small cap so the test doesn't need to actually push megabytes -
        # what's under test is that SOMETHING stops an oversized body,
        # not the specific default limit (that's a plain settings.py
        # default, not logic worth a test of its own).
        monkeypatch.setattr(app_module.settings, "MAX_UPLOAD_SIZE_MB", 1)
        oversized = b"\xff\xd8\xff\xe0" + b"\x00" * (2 * 1024 * 1024)  # 2MB > 1MB cap
        response = client.post(
            "/ocr?include_images=false",
            files={"file": ("big.jpg", io.BytesIO(oversized), "image/jpeg")},
        )
        assert response.status_code == 413
        assert "limit" in response.json()["detail"].lower()

    def test_upload_within_the_cap_is_not_rejected_for_size(self, client, monkeypatch):
        import app as app_module
        monkeypatch.setattr(app_module.settings, "MAX_UPLOAD_SIZE_MB", 1)
        small = b"not a real image but under the size cap"
        response = client.post(
            "/ocr?include_images=false",
            files={"file": ("small.jpg", io.BytesIO(small), "image/jpeg")},
        )
        # Rejected for being unreadable as an image (400), not for size
        # (413) - proves the cap doesn't fire below the threshold.
        assert response.status_code != 413
