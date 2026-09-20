# Changelog

Notable changes to the Python package (`egyptian-national-id-ocr`). Format
follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions
follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Accuracy and speed claims here are measured against the benchmark
described in [docs/LIMITATIONS.md](docs/LIMITATIONS.md), not estimated.

## [Unreleased]

## [0.3.0] - 2026-09-20

### Added
- `docs/INTERFACES.md`: full input/output reference for every interface
  (CLI, HTTP API, MCP server, Python SDK) - exact request/response
  shapes, every field documented, real example output from a synthetic
  card. Linked from the README's own per-interface sections rather than
  duplicating the detail inline.
- Docker: fixed `docker/backend.Dockerfile`, which had drifted to
  target a dead duplicate API module (`egyptian_national_id_ocr.api.
  server`, an unhardened pre-Phase-1 copy of the real app with none of
  this release's CORS allowlist or upload-size-cap fixes) and installed
  `tesseract-ocr` system packages for an engine that hasn't been the
  default since before this changelog starts. Removed the dead module
  entirely; the Dockerfile now builds `src/app.py` (the same app the
  README documents and CI tests) and installs `libgomp1` (PaddlePaddle's
  OpenMP runtime - present on a full OS image, easy to miss on a
  minimal one) and pre-downloads PaddleOCR's models at build time rather
  than on the container's first request. Verified by actually running
  the built image and posting a real image to `/ocr`: identical output
  to the same card processed outside Docker.
- `.github/workflows/docker-publish.yml`: builds and pushes
  `docker/backend.Dockerfile` to GitHub Container Registry (`ghcr.io`)
  on every published release, using the workflow's own built-in
  `GITHUB_TOKEN` - no external account or stored secret to configure.

### Fixed (post-push, round 2)
CI's second run failed on a third real bug, in `scripts/training/
generate_trial_ids.py`: `FONT_PATH = "C:/Windows/Fonts/tahomabd.ttf"` -
a hardcoded Windows path that always existed, but only became
CI-blocking once the Phase 3 hygiene pass made the whole test suite
depend on this generator (via `tests/conftest.py`'s
`synthetic_card_sample` fixture, replacing the retired
`assets/dataset/`). GitHub's Linux runners have no such path, so every
synthetic-card-dependent test failed with `OSError: cannot open
resource`.

Fixed by bundling an open-source Arabic font (`assets/fonts/`, Amiri,
SIL Open Font License - see `assets/fonts/OFL.txt`) directly in the
repo, rather than depending on any OS's installed fonts. This also
fixes a subtler, pre-existing problem: "seeded generation is
reproducible" was never actually true across machines - nothing pinned
which font file "Tahoma" even resolved to, so the same seed could
render different pixels on different Windows installs, let alone
across OSes. A single bundled file is identical everywhere this
repository is cloned.

Verified the font swap doesn't regress accuracy before treating it as
safe: 5 seeded synthetic cards, national ID 5/5, first_name 4/5,
full_name 5/5, serial_number 5/5. The one systematic mismatch
(`address`, 0/5 - missing a "-" separator) was proven pre-existing and
font-independent via direct A/B against the original Tahoma font on the
same seed (identical omission with or without the font change) -
already documented as a known gap in `docs/LIMITATIONS.md`.

The digit-classifier training tooling (`generate_digit_crops.py`,
`stress_test_digit_classifier.py`) keeps Tahoma as its default when
available rather than switching outright: their own code comments
already document a measured, reverted regression with calligraphic
Naskh-style fonts (which the bundled fallback also is) specifically for
isolated digit glyphs, so the bundled font is used there only as an
explicitly-unverified last resort for a non-Windows contributor, not
treated as equivalent.

Given two failed pushes already, this fix was verified in an actual
Linux container (Docker, `python:3.11-slim`, matching CI's Python
version) before pushing again, not just locally on Windows a third
time: 85 passed, 4 skipped (one more skip than Windows' 3 - the
Windows-only encoding test correctly skips on Linux), 6 xfailed.

### Fixed (post-push, round 1)
CI's actual first run on GitHub's runners (unverifiable locally - see
the Phase 2 entry above) failed on push, in all 4 jobs. Root causes,
found from the real run logs rather than guessed:
- `pytest` was never declared as an installable dependency anywhere -
  it only ran locally because a dev venv happened to have it installed
  outside `pyproject.toml`. Added a `test` extra (`pytest`, `httpx` -
  the latter needed by FastAPI's `TestClient`) and had both CI jobs
  install it.
- The `packaging` job installed only `pytest` itself, but the outer
  `pytest tests/test_packaging.py` invocation still collects
  `tests/conftest.py` (which imports `cv2`/`numpy` to build its
  synthetic-card fixture) even when told to run one file - so it now
  installs the full package via the same `test` extra.
- `tests/test_packaging.py`'s CLI-entry-point test had a path-doubling
  bug (`clean_venv.parent / "Scripts"` when `clean_venv.parent` was
  already the Scripts directory) and, separately, still pointed at
  `assets/dataset/ID0.png` - deleted in the Phase 3 hygiene pass without
  this reference being updated, so it was silently skipping instead of
  testing anything. Fixed the path; switched the image to a freshly
  generated seeded synthetic card instead of a file that could go
  missing again.

All three re-verified end to end in a genuinely fresh venv (not the dev
environment that had been masking the missing `pytest` dependency)
before pushing the fix.

### Removed
- 18 committed `debug_*`/`detected_*` files (dumped at the repo root
  and in `debug_rois/` from earlier debugging sessions) - synthetic,
  not real card data, but stale clutter with no place in a published
  repo.
- `assets/dataset/` (720 unlabelled files). Predated field-box
  calibration against `assets/front_template.jpg`; measured at 0.30
  confidence and up to 175s per image (see the time-budget entry
  below - this was the reproduction case for that fix), and the one
  test that used it was already marked `xfail`. Superseded entirely by
  the seeded synthetic generator (`scripts/training/generate_trial_ids.py`)
  and the maintainer's own real-card benchmark
  (`scripts/benchmark_own.py`). `tests/conftest.py`'s fixtures and the
  tests that used them (`test_layout_pipeline.py`,
  `test_detection_manually.py`) now generate a fresh seeded card
  instead - and assert real expected values from the generator's own
  ground truth, where the old xfail test only checked "some 14 digits
  that pass a checksum", and the old detection test had **no
  assertions at all** (it printed and `continue`d past failures, so it
  "passed" unconditionally regardless of whether detection worked).
- `pytesseract` from required dependencies. Nothing in the package
  imports `TesseractEngine` by default (confirmed: grepped for it,
  found only a comment); moved to the existing `legacy-ocr` optional
  extra alongside `easyocr`. Verified by uninstalling it from the dev
  environment and confirming every core import and the full test suite
  still pass.

### Security
- Overwrote the 2D barcode region in `assets/Egyptian_ID_Card.jpg` (a
  stock two-sided card image in the repo since its initial commit) and
  `assets/back_template.jpg` (cut from it) with random noise. A real
  Egyptian ID's barcode encodes the holder's data, and this stock
  image's provenance - genuine photograph vs. purpose-built mock-up -
  could not be established from the file itself. The pipeline never
  reads this barcode (no such field exists in `BACK_FIELDS`), so this
  costs nothing functionally. See
  `scripts/training/scrub_barcode.py` for the reproducible procedure
  and the reasoning in full. Verified post-scrub: shift-robust
  cross-correlation against a real card's barcode is at chance level
  (0.171 vs. a 0.091 chance ceiling).

### Added
- PyPI metadata: `classifiers`, `keywords`, `[project.urls]`
  (Homepage/Repository/Issues/Changelog). Verified in the built wheel's
  METADATA, not just in `pyproject.toml`.
- `SECURITY.md`, `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`.
- `tests/test_privacy_no_network.py`: blocks real outbound network
  connections during a `process_image()` call and fails if one is
  attempted - the project's central "extraction is fully local" claim,
  checked directly instead of only asserted in the README.
- CI (`.github/workflows/ci.yml`): pytest across Python 3.10-3.12, plus
  a dedicated job that builds the wheel and installs it into a clean
  venv (`tests/test_packaging.py`, the same check done manually for
  Phase 0). Runs against synthetic data only - the private benchmark
  never reaches CI, by design.
- `MAX_PROCESSING_SECONDS` (45s) hard wall-clock ceiling on one
  extraction, checked between rotation attempts and before entering the
  numeric offset-retry ladder, the free-text fallback, and the free-text
  rescan pass - not just at the outer loop, since the combinatorial cost
  lives inside those ladders. Measured on a real failure: an image that
  defeated card detection took 175s before this existed (a
  denial-of-service surface on a public API/MCP endpoint), and 45-50s
  after. Verified against the maintainer's own benchmark to add zero
  regression on any scan that already worked (worst real case measured
  at 25.1s, well under the 45s budget).
- Configurable CORS allowlist (`NID_CORS_ALLOWED_ORIGINS`) replacing
  `allow_origins=["*"]`, which combined with `allow_credentials=True`
  let any website's JavaScript call the API using the visiting browser's
  own credentials.
- Upload size cap on `/ocr` (`NID_MAX_UPLOAD_SIZE_MB`, default 15),
  enforced by reading in bounded chunks and aborting mid-stream rather
  than checking after buffering the whole body - which would not have
  bounded memory for an attacker who just sends more bytes.

### Fixed
- The built wheel did not contain the trained digit classifier or the
  detection templates - setuptools' default file finder collects `.py`
  files only. A fresh `pip install` looked successful and then failed on
  the first scan ("Digit classifier model not found"), for every numeric
  field. Added `[tool.setuptools.package-data]`; proven by building a
  wheel and installing it into a venv with no source tree available
  (`tests/test_packaging.py`, opt-in via `RUN_PACKAGING_TEST=1`).
- The CLI crashed with `UnicodeEncodeError` on Windows' default console
  codepage (cp1252) printing the first Arabic field - which is most
  fields, since Arabic is this tool's whole subject. Force UTF-8 on both
  the native Windows console codepage and Python's stdout/stderr at CLI
  startup (`tests/test_cli_encoding.py`).

### Added
- `LICENSE` (MIT) - `pyproject.toml` declared the license but no file
  granted it.

## [0.2.0]

### Added
- English transliteration for every free-text field
  (`first_name_english`, `full_name_english`, `address_english`,
  `profession_english`). Machine-generated for display and search — not
  the official spelling on the holder's documents. See LIMITATIONS.
- `include_images` / `--no-include-images` on the HTTP API, Python SDK and
  CLI, to drop base64 payloads from responses.
- MCP server (`egy-nid-ocr-mcp`) exposing `extract_id_card` and
  `validate_national_id`.
- `is_expired` on the back model, with `null` meaning "cannot tell"
  rather than "not expired".
- `scripts/benchmark_own.py` and `tests/test_own_benchmark.py`: an
  accuracy + speed regression run over every scan variant of a real card.
  The card is not distributed; both skip themselves when it is absent.
- `docs/LIMITATIONS.md`, documenting every measured gap.

### Changed
- Card detection no longer requires a clean four-point contour, so
  zoomed-out and low-contrast scans align instead of failing outright.
- An unreadable `expiry_date` now degrades to `YYYY/MM` recovered from
  the issue date via the 7-year validity rule, instead of blanking. The
  day is never invented.
- A free-text region that returns no detections at all is re-read
  upscaled 3x, recovering `marital_status` on greyscale scans.

### Fixed
- `expiry_date`'s offset search was validated against the *unrepaired*
  issue date, so on a scan whose issue year misread it searched for a
  date that could not exist.
- `birth_date` no longer triggers the per-field offset search, which had
  taken front scans from ~5s to ~25s for a field that is derived from the
  checksum-validated national ID anyway.
- National ID repair no longer reports `corroborated` when there was
  nothing to corroborate against.

## [0.1.0]

- Initial extraction pipeline: card detection, field cropping, PaddleOCR
  for free text, a dedicated HOG+SVM classifier for digits, mod-11
  checksum validation and repair, FastAPI service and React demo UI.

[Unreleased]: https://github.com/HassanSalama2001/Egyptian-National-ID-Open-Source/compare/v0.3.0...HEAD
[0.3.0]: https://github.com/HassanSalama2001/Egyptian-National-ID-Open-Source/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/HassanSalama2001/Egyptian-National-ID-Open-Source/releases/tag/v0.2.0
[0.1.0]: https://github.com/HassanSalama2001/Egyptian-National-ID-Open-Source/releases/tag/v0.1.0
