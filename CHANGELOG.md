# Changelog

Notable changes to the Python package (`egyptian-national-id-ocr`). Format
follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions
follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Accuracy and speed claims here are measured against the benchmark
described in [docs/LIMITATIONS.md](docs/LIMITATIONS.md), not estimated.

## [Unreleased]

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

[Unreleased]: https://github.com/HassanSalama2001/National-ID-Open-Source/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/HassanSalama2001/National-ID-Open-Source/releases/tag/v0.2.0
[0.1.0]: https://github.com/HassanSalama2001/National-ID-Open-Source/releases/tag/v0.1.0
