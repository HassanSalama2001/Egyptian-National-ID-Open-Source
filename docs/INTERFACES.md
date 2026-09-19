# Interfaces

Every way to call this pipeline — CLI, HTTP API, MCP server, and the
Python SDK — reads the same input and returns the same underlying
[`IDCard`](../src/egyptian_national_id_ocr/models/id_card.py) data. This
page documents exactly what each interface expects in and gives back,
with real (synthetic) example output for every field.

## Contents

- [Shared concepts](#shared-concepts) — read this first
- [Command line (CLI)](#command-line-cli)
- [HTTP API](#http-api)
- [MCP server](#mcp-server)
- [Python SDK](#python-sdk)
- [Field reference](#field-reference)
- [JS/TypeScript client](#jstypescript-client)

---

## Shared concepts

**Input**, everywhere: a JPG or PNG photo/scan of one side of an Egyptian
National ID card — front or back, either is auto-detected. See
[`docs/LIMITATIONS.md`](LIMITATIONS.md) for what scan conditions actually
work today. There is no way to submit "both sides at once"; call the
pipeline twice, once per photo.

**Output**, everywhere: the same [`IDCard`](#field-reference) shape —
`side`, `front` (`null` on a back scan), `back` (`null` on a front scan),
`decoded` (derived from the checksum-validated national ID number),
`confidence`, `status`, `messages`, and `processing_time_ms`. The CLI,
HTTP API and MCP server each wrap or trim this differently (documented
per-interface below), but every field name and value inside it is
identical everywhere.

**`status`** is the first thing to check, before looking at any field:

| value | meaning |
|---|---|
| `success` | The national ID passed its checksum and structural checks, or a corrected digit was confirmed by another field on the card. Trust the result. |
| `low_confidence` | A card was read, but the national ID could not be verified — either the checksum never passed, or a digit was corrected with nothing else on the card confirming it. Treat every value as needing human review. |
| `no_card_detected` | No card-shaped region was found in the image at all. |
| `unreadable_image` | The file couldn't be decoded as an image. |

**`is_expired`** (back only) is a tri-state, not a boolean: `true`,
`false`, or `null`. `null` means "cannot tell" — the expiry date was
unreadable, or (specifically) it degraded to a year+month with no day,
and today happens to fall in that exact expiry month, where the missing
day is exactly what would decide the answer. Never treat `null` as
"not expired".

**`*_english` fields** (`first_name_english`, `address_english`,
`profession_english`, …) are machine transliterations for display and
search — **not** the official Latin spelling on the holder's other
documents, which cannot be derived from the Arabic (محمد is legitimately
Mohamed, Mohammed, Muhammad, or Mohamad depending on the person's own
papers).

**Images in the response** (`card_image`, `field_crops`) are the aligned
card and per-field crops as base64 JPEG data URIs, included by default so
a UI can show "here's what we detected" next to the extracted text. They
dominate the response size — every interface has a way to drop them (see
below) for a server-to-server call or an AI assistant that can't look at
images anyway.

---

## Command line (CLI)

```bash
egy-nid-ocr extract path/to/card.jpg
egy-nid-ocr extract path/to/card.jpg --output json
egy-nid-ocr extract path/to/card.jpg --output json --include-images
egy-nid-ocr doctor
```

| flag | default | effect |
|---|---|---|
| `image_path` (positional) | required | Path to a JPG/PNG file on disk. |
| `--output` / `-o` | `table` | `table` prints a readable Rich table to the terminal; `json` prints the bare `IDCard` object (see below). |
| `--include-images` / `--no-include-images` | **off** | Include base64 `card_image`/`field_crops` in `--output json`. Off by default — a terminal or a pipe into `jq` is the wrong place for a megabyte of base64. |

`egy-nid-ocr doctor` checks the runtime is set up correctly (OpenCV,
PaddleOCR, the trained digit-classifier model, and — as an optional,
non-failing check — a system Tesseract install) and prints what's missing.

### `--output json` shape

The **bare** `IDCard` object — not wrapped in anything. This is the one
place that differs from the HTTP API, which wraps the same object in
`{"data": ..., "image": ...}` (see below). Piping into `jq` works
directly:

```bash
egy-nid-ocr extract card.jpg --output json | jq '.front.national_id'
```

<details>
<summary>Real example — front side (synthetic card, click to expand)</summary>

```json
{
  "side": "front",
  "front": {
    "first_name": "خالد",
    "full_name": "عادل شعبان",
    "address": "٥٦٥ الياسمين عمارات عين شمس البحيرة",
    "national_id": "27310162859511",
    "date_of_birth": "1973-10-16",
    "card_serial_number": "VM3526924",
    "first_name_english": "Khaled",
    "full_name_english": "Adel Shaaban",
    "address_english": "565 El Yasmin Buildings Ain Shams Beheira",
    "field_crops": {},
    "raw_arabic": {
      "first_name": "خالد",
      "full_name": "عادل شعبان",
      "address": "٥٦٥ الياسمين عمارات عين شمس البحيرة",
      "national_id": "27310162859511",
      "birth_date": "1973/10/16",
      "serial_number": "VM3526924"
    }
  },
  "back": null,
  "decoded": {
    "birth_date": "1973-10-16",
    "governorate": "aswan",
    "gender": "male",
    "century": "1900s",
    "sequence": "5951",
    "check_digit": 1
  },
  "confidence": 0.9676155090332031,
  "processing_time_ms": 4334,
  "status": "success",
  "messages": ["National ID extracted and verified successfully."],
  "card_image": null
}
```

`field_crops: {}` and `card_image: null` because this run used
`--include-images` off (the default). `raw_arabic` echoes each
free-text/numeric field's own key — note `serial_number`, not
`card_serial_number` as on `front` itself; the two are the same value
under slightly different internal names.

</details>

<details>
<summary>Real example — back side, same card (click to expand)</summary>

```json
{
  "side": "back",
  "front": null,
  "back": {
    "national_id": "27310162859511",
    "issue_date": "2024/08",
    "profession": "صيدلي –",
    "gender": "male",
    "religion": "muslim",
    "marital_status": "widowed",
    "expiry_date": "2031/08/09",
    "profession_english": "Pharmacist –",
    "is_expired": false,
    "field_crops": {},
    "raw_arabic": {
      "issue_date": "2024/08",
      "national_id": "27310162859511",
      "profession": "صيدلي –",
      "marital_status": "أرمل",
      "religion": "مسلم",
      "gender": "ذ كر",
      "expiry_date": "2031/08/09"
    }
  },
  "decoded": {
    "birth_date": "1973-10-16",
    "governorate": "aswan",
    "gender": "male",
    "century": "1900s",
    "sequence": "5951",
    "check_digit": 1
  },
  "confidence": 0.8775946591581618,
  "processing_time_ms": 5432,
  "status": "success",
  "messages": ["National ID extracted and verified successfully."],
  "card_image": null
}
```

Shown exactly as the pipeline produced it, stray "–" in `profession` and
all — see [`docs/LIMITATIONS.md`](LIMITATIONS.md) for what's a known gap
versus what's solid. `gender`/`religion`/`marital_status` are
closed-vocabulary enums matched from noisy OCR text (`raw_arabic.gender`
is literally `"ذ كر"`, an extra space and all) — that's what they're
*for*.

</details>

---

## HTTP API

```bash
uvicorn src.app:app --reload
```

### `POST /ocr`

**Request:** `multipart/form-data` with one field, `file` (the image).

| query param | default | effect |
|---|---|---|
| `include_images` | `true` | Include `card_image` and every `field_crops` entry. Set `false` for a data-only response — orders of magnitude smaller, the right default for a server-to-server integration. |

```bash
curl -X POST "http://localhost:8000/ocr?include_images=false" \
     -F "file=@card.jpg"
```

**Response**, always:

```json
{ "data": { /* the IDCard object, see below */ }, "image": "data:image/jpeg;base64,..." | null }
```

`data` is the same `IDCard` shape as the CLI's `--output json` (see the
two full examples above — identical field-for-field). `image` is a base64
copy of the *original uploaded photo* (not the aligned card) for the web
UI's side-by-side view; it's `null` whenever `include_images=false`,
independent of whether `data.card_image`/`data.front.field_crops` are
also stripped (they are, by the same flag).

**Error responses** — always `{"detail": "<plain-language message>"}`,
never a raw stack trace:

| status | when |
|---|---|
| `400` | Not an image file, an empty file, or a file that couldn't be decoded as an image. |
| `413` | Upload exceeds the size cap (`NID_MAX_UPLOAD_SIZE_MB`, default 15MB). |
| `500` | An unexpected error during processing (logged server-side; the pipeline's own errors already resolve to a `status` in the 200 response, so this is genuinely unexpected). |

A `4xx`/`5xx` here means the request itself failed — it's not how a
low-confidence *read* is reported. A card that was found but didn't
verify still comes back `200` with `data.status: "low_confidence"`.

### `GET /health`

```json
{ "status": "healthy", "engine": "PaddleOCR" }
```

No auth, no body. Point a container orchestrator's liveness probe at it.

### CORS and upload limits

Configurable via environment variables (`NID_` prefix — see
[`core/config.py`](../src/egyptian_national_id_ocr/core/config.py)):
`NID_CORS_ALLOWED_ORIGINS` (comma-separated, defaults to the `web_ui` dev
server's own origin) and `NID_MAX_UPLOAD_SIZE_MB` (default `15`).

---

## MCP server

```bash
pip install "egyptian-national-id-ocr[mcp]"
egy-nid-ocr-mcp
```

Exposes two tools over stdio — point an MCP client (Claude Desktop, etc.)
at the `egy-nid-ocr-mcp` command. Both use `structured_output=True`, so a
client gets real parsed data, not JSON embedded in a text block.

### `extract_id_card(image_path: str) -> dict`

**Input:** `image_path` — an absolute or `~`-expandable path to a JPG/PNG
file **on the machine running the MCP server**, not a URL and not inline
image bytes. (The server and the calling assistant may be on different
machines — the path must resolve on the server's filesystem.)

**Output:** the same `IDCard` object as everywhere else, with images
always stripped (`without_images()` — same method the HTTP API's
`include_images=false` and the CLI's default both use), as JSON-mode
data (enums as their string values, e.g. `"male"` not `Gender.MALE`). A
path that doesn't exist or can't be decoded returns a small error object
instead of raising:

```json
{ "status": "error", "messages": ["No image file at /no/such/path.jpg"] }
```

### `validate_national_id(national_id: str) -> dict`

Checksum-and-structure-validates a 14-digit number with **no image and
no OCR model load** — useful on its own (e.g. validating a number a user
typed) or as a fast pre-check. Different, smaller shape than
`extract_id_card`:

```json
// valid
{
  "valid": true,
  "national_id": "29001011234564",
  "birth_date": "1990-01-01",
  "governorate": "cairo",
  "gender": "male",
  "century": "1900s"
}
```
```json
// invalid - checksum fails
{ "valid": false, "reason": "Mod-11 check digit does not match." }
```
```json
// invalid - checksum passes but the number can't describe a real person
{
  "valid": false,
  "reason": "Check digit matches, but the number cannot describe a real cardholder (invalid century digit, unknown governorate code, or an impossible birth date)."
}
```

A passing checksum alone is **not** sufficient evidence — about 1 in 11
arbitrary 14-digit strings satisfies Mod-11 by chance, which is why this
also runs the structural checks (century, governorate code, plausible
age) before returning `valid: true`.

---

## Python SDK

```python
import cv2
from egyptian_national_id_ocr.core.pipeline import Pipeline

pipeline = Pipeline()                    # loads OCR models once - keep the instance around
image = cv2.imread("card.jpg")           # BGR, like every cv2.imread
result = pipeline.process_image(image)   # returns an IDCard (pydantic model)

print(result.status, result.confidence)
if result.front:
    print(result.front.national_id, result.front.first_name)

# Strip base64 images before logging, storing, or forwarding to something
# that can't look at images anyway - same method the HTTP API and MCP
# server use internally:
slim = result.without_images()
print(slim.model_dump_json(indent=2))
```

**Input:** a `numpy.ndarray` in BGR order (`cv2.imread`'s native format),
**not** a file path and not raw bytes — decode the file yourself first.
If you're loading from bytes (an upload, a network response), decode via
PIL and apply `ImageOps.exif_transpose` before converting to BGR, the
same way `src/app.py` does — a phone photo's pixels are commonly stored
sideways with an EXIF tag saying how to rotate for display, and
`cv2.imdecode` ignores that tag entirely:

```python
from PIL import Image, ImageOps
import cv2, numpy as np

pil_image = ImageOps.exif_transpose(Image.open("card.jpg"))
image = cv2.cvtColor(np.array(pil_image.convert("RGB")), cv2.COLOR_RGB2BGR)
```

**Output:** an `IDCard` — a real Pydantic model, not a dict. Use
`.model_dump()` / `.model_dump_json()` to get plain data, or access
fields directly (`result.front.national_id`) with full IDE
autocompletion and type checking. See the [field reference](#field-reference)
below for the complete schema, or read
[`models/id_card.py`](../src/egyptian_national_id_ocr/models/id_card.py)
directly — every field has its own docstring.

`Pipeline()` loads the OCR engine at construction time (a few seconds) —
construct one instance and reuse it across images; don't build a new
`Pipeline()` per call.

---

## Field reference

### `IDCard` (top level, every interface)

| field | type | notes |
|---|---|---|
| `side` | `str` | `"front"`, `"back"`, or `"unknown"` |
| `front` | `IDCardFront \| null` | present iff `side == "front"` |
| `back` | `IDCardBack \| null` | present iff `side == "back"` |
| `decoded` | `NationalIDDecoded \| null` | derived from the checksum-validated national ID, not re-read from the card |
| `confidence` | `float` | 0.0–1.0; see `status` first, this is a secondary signal |
| `processing_time_ms` | `int` | wall-clock time for this call |
| `status` | enum | see [Shared concepts](#shared-concepts) |
| `messages` | `str[]` | plain-language, end-user-safe explanations — never a stack trace |
| `card_image` | `str \| null` | base64 JPEG data URI of the aligned card |

### `IDCardFront`

`first_name`, `full_name`, `address`, `national_id`, `date_of_birth`,
`card_serial_number` — each `str | null` — plus `first_name_english`,
`full_name_english`, `address_english` (machine transliterations, see
above), `field_crops` (`dict[str, str]`, per-field base64 crops), and
`raw_arabic` (`dict[str, str]`, the field text before any cross-field
repair — e.g. before a checksum-based digit correction).

### `IDCardBack`

`national_id`, `issue_date` (`YYYY/MM`), `expiry_date` (`YYYY/MM/DD`, or
`YYYY/MM` when the day couldn't be read — see `is_expired` above),
`profession` — each `str | null` — plus `profession_english`, `gender` /
`religion` / `marital_status` (closed-vocabulary enums, see below),
`is_expired` (`bool | null`), `field_crops`, `raw_arabic`.

### `NationalIDDecoded`

Everything the 14-digit number itself encodes, independent of what's
printed elsewhere on the card: `birth_date`, `governorate`, `gender`,
`century` (`"1900s"` or `"2000s"`), `sequence`, `check_digit`.

### Enums

| field | possible values |
|---|---|
| `status` | `success`, `low_confidence`, `no_card_detected`, `unreadable_image` |
| `gender` | `male`, `female`, `unknown` |
| `religion` | `muslim`, `christian`, `jewish`, `other`, `unknown` |
| `marital_status` | `single`, `married`, `divorced`, `widowed`, `unknown` |
| `governorate` | one of Egypt's 27 governorates (snake_case, e.g. `kafr_el_sheikh`) plus `outside_egypt`, `unknown` — full list in [`models/enums.py`](../src/egyptian_national_id_ocr/models/enums.py) |

---

## JS/TypeScript client

**Does not exist yet.** There's a private React demo (`web_ui/`) that
calls the HTTP API directly with `fetch` — no published npm package. If
you need JS/TS, call the [HTTP API](#http-api) directly; its request/
response shape above is the actual contract. See
[`docs/RELEASING.md`](RELEASING.md) for why this isn't advertised as
available.
