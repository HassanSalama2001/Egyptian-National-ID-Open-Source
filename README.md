# Egyptian National ID OCR

A local, open-source pipeline that extracts structured data from Egyptian
National ID cards — front and back, any scan condition. No cloud APIs, no
LLMs: everything runs on your machine, and nothing about the card ever
leaves it.

**Egyptian IDs only.** The card layout, the checksum, the governorate
table and the field vocabularies are all specific to this document. It
will not read another country's ID.

Read **[docs/LIMITATIONS.md](docs/LIMITATIONS.md)** before relying on
this for anything — it documents exactly what does and doesn't work yet,
measured against a real card scanned four different ways, not estimated.

Four ways to call it — command line, Python, HTTP API, or MCP server, all
below. Every one of them takes the same input and returns the same
fields; **[docs/INTERFACES.md](docs/INTERFACES.md)** is the full
reference for each — exact request/response shapes, every field
documented, and real example output (synthetic card, never a real one).

## What it extracts

**Front:** national ID number (checksum-validated), first/full name,
address, date of birth, card serial number.

**Back:** national ID number, issue date, expiry date (with `is_expired`),
profession, gender, religion, marital status.

Every free-text field also gets a machine-transliterated Latin-script
version (`first_name_english`, `address_english`, …) for display and
search. That transliteration is **not** the official spelling on the
holder's documents — see the field's own description in
[`models/id_card.py`](src/egyptian_national_id_ocr/models/id_card.py) for
why that can't be derived from the Arabic.

## Install

```bash
pip install egyptian-national-id-ocr
```

From source:

```bash
git clone https://github.com/HassanSalama2001/Egyptian-National-ID-Open-Source.git
cd Egyptian-National-ID-Open-Source
pip install -e .
```

## Command line

```bash
egy-nid-ocr extract path/to/card.jpg
egy-nid-ocr extract path/to/card.jpg --output json
egy-nid-ocr doctor    # checks the runtime is set up correctly
```

Full flag reference and example JSON output:
[docs/INTERFACES.md § Command line](docs/INTERFACES.md#command-line-cli).

## Python

```python
import cv2
from egyptian_national_id_ocr.core.pipeline import Pipeline

pipeline = Pipeline()
image = cv2.imread("card.jpg")
result = pipeline.process_image(image)

print(result.status, result.confidence)
if result.front:
    print(result.front.national_id, result.front.first_name)
```

Full `IDCard` field reference (every field, every enum value):
[docs/INTERFACES.md § Python SDK](docs/INTERFACES.md#python-sdk).

## HTTP API

```bash
uvicorn src.app:app --reload
```

```bash
curl -X POST "http://localhost:8000/ocr?include_images=false" \
     -F "file=@card.jpg"
```

`include_images=false` drops the base64 card/crop images from the
response — the difference between a payload measured in kilobytes and
one measured in megabytes.

Full request/response shapes, error codes, and example output:
[docs/INTERFACES.md § HTTP API](docs/INTERFACES.md#http-api).

## MCP server

Exposes the pipeline as tools an AI assistant can call directly.

```bash
pip install "egyptian-national-id-ocr[mcp]"
egy-nid-ocr-mcp
```

Tool signatures and example results:
[docs/INTERFACES.md § MCP server](docs/INTERFACES.md#mcp-server).

## Web demo

A React UI under [`web_ui/`](web_ui/) that shows the extraction alongside
the crop it came from, for verifying results by eye.

```bash
cd web_ui && npm install && npm run dev
```

## Why it's reliable on bad scans

Real ID photos are rarely clean: black-and-white scans, zoomed-out shots,
odd lighting, off-angle cameras. The pipeline is built around that, not
around a clean-input assumption:

- **Card detection first, fields second.** The card itself is located and
  perspective-corrected before any field coordinates are applied — fixed
  boxes are only meaningful once the card is rectified.
- **A binarization ladder for digits** (multiple thresholding methods ×
  upscale factors), accepted only when two independent methods agree —
  not the first plausible-looking read.
- **Checksum validation and repair** on the national ID number: Egyptian
  IDs carry a mod-11 check digit, so a misread digit can often be
  detected and corrected rather than silently returned wrong.
- **Cross-field validation**, not just per-field: the back's issue and
  expiry dates check each other against the card's fixed 7-year validity
  period; an unreadable expiry degrades to what the issue date implies
  (year and month) rather than to nothing — but the day is never
  invented, and `is_expired` answers `null` rather than guess when the
  day is what it would take to decide.
- **The national ID and the names are the priority fields** — everything
  else (birth date, governorate, gender, century) is derivable from the
  checksum-validated ID number, so those get the most validation.

None of this is asserted — see [`docs/LIMITATIONS.md`](docs/LIMITATIONS.md)
for the actual numbers, field by field, scan condition by scan condition,
regenerated by [`scripts/benchmark_own.py`](scripts/benchmark_own.py) on
every change.

## Privacy

Extraction happens entirely on your machine. No card image, and no
extracted field, is sent anywhere by this library. See
[`docs/LIMITATIONS.md`](docs/LIMITATIONS.md) for what "reliable" currently
means in practice, and [`SECURITY.md`](SECURITY.md) for how to report a
security issue.

## Contributing

See [`CONTRIBUTING.md`](CONTRIBUTING.md) and
[`docs/RELEASING.md`](docs/RELEASING.md).

## License

MIT — see [`LICENSE`](LICENSE).
