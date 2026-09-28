# Known limitations

Measured against a real Egyptian National ID card scanned four ways
(unfiltered, "enhanced", greyscale and "eco"), front and back — eight
images, re-run on every change via `scripts/benchmark_own.py`. Figures
below are from that run, not estimates. The card itself is not in this
repository; see "Reproducing these numbers".

Read this as the honest state of the project. Anything not listed here
has not been measured, which is not the same as working.

## Accuracy

| Field | Unfiltered | Enhanced | Greyscale | Eco |
|---|---|---|---|---|
| `national_id` (front + back) | ✅ | ✅ | ✅ | ✅ |
| `first_name`, `full_name` | ✅ | ✅ | ✅ | ✅ |
| `card_serial_number` | ✅ | ✅ | ✅ | ✅ |
| `address` | ✅ | 1 char off | 1 char off | several chars off |
| `issue_date`, `expiry_date` | ✅ | ✅ | ✅ | ✅ |
| `profession` | 1 char off | ✅ | ✅ | ✅ |
| `gender` | ✅ | ✅ | ✅ (from the ID number) | ✅ |
| `religion`, `marital_status` | ✅ | ✅ | ✅ | ✅ |

The two fields that matter most — the **national ID number** and the
**names** — read correctly on every variant tested. Everything derived
from the national ID (birth date, governorate, gender, century) is
checksum-validated, so it is as reliable as the number itself.

### `address` drifts by a character on filtered scans
One extra character on enhanced/greyscale, several wrong on eco. This is
recogniser noise on a long free-text line, not truncation — the whole
address is found, some glyphs are read wrong. No validation exists that
could catch it, because an address has no checkable structure.

### `profession` is one character off on the unfiltered back
Same length, one glyph different. That scan has the card cut off at
the top edge of the photo, and the recogniser is sensitive to exactly
where the crop falls. Greyscale and eco, which used to lose the last
word, now read it in full.

### `gender` on the greyscale back comes from the ID number
The printed word is not read on that scan. When the national ID is
verified, its 13th digit encodes gender, so that is used instead. A
gender that *was* read is never overwritten — it is the independent
cross-check the ID repair relies on.

### A national ID that cannot exist is not shown
If the number fails its checksum and no repair is possible, and it is
not even a structurally possible ID (wrong length, century, date or
governorate — a third-party scan returned `00000000200000`), the field is
left empty with a message saying so, rather than displayed as data.

## Card cropping

Every field box is a fixed position on the rectified card, so the crop
decides whether anything else can work. `scripts/training/
benchmark_card_crop.py` builds scenes with a known card placement and
measures how far the cropped card's corners land from where they belong
(pass = within 12px on the 1200x750 card). Two seeds, 20 cards each:

| Scene | Pass |
|---|---|
| Card fills the photo, dark / light background | 20/20, 20/20 |
| Photo is exactly the card (must not be trimmed) | 20/20 |
| Card small and sideways on a scanned A4 page | 19-20/20 |
| Angled, perspective, busy background (colour / grey) | 20/20 |
| Upside down | 19-20/20 |
| Negative scan; inverted card on a page | 20/20, 20/20 |

Overall 99%, against 16% for the previous alignment. On real scans the
three filter variants of each side (the same scan, so the card sits in
the same place) crop to within 3px of each other; `tests/
test_own_benchmark.py` checks that.

Weakest remaining case: a **small card on a white page** lands ~5px off
on average (one photo pixel is ~3 card pixels there). A card edge that
is invisible against its background (white card rim on white paper, an
"eco" filter) is placed from the opposite edge and the card's known
shape, which is accurate only when the rest of the outline is clear.

### Synthetic cards are a secondary check, real scans decide
Field boxes are positioned from real scans (the back's are 13px higher
than `align_card` needed, measured as a plateau across four real back
scans). The synthetic generator, `scripts/training/generate_trial_ids.py`,
draws the back from those same boxes (`BACK_FIELDS` plus
`PHYSICAL_FRAME_SHIFT`), mapped onto the template's physical card area.
It used to map them onto the whole template canvas, white border
included, which put every back row somewhere the pipeline does not
read: generated backs read their national ID 17/36 times across six
scene types, against 34/36 after the fix. Generated fronts: 30/36 either
way. `benchmark_card_crop.py --e2e` reports these; it is a crop
benchmark first.

## Speed

| Scan | Front | Back |
|---|---|---|
| Unfiltered | 4.9s | 11.9s |
| Enhanced | 5.5s | 16.9s |
| Greyscale | 6.0s | 2.8s |
| Eco | 5.0s | 11.0s |

Backs are slower when a digit field fails its validator on the first
read, which triggers the per-field offset search across the full
binarization ladder. Every call is capped at 45s.

These are CPU timings on one machine. Treat them as ratios, not promises.

## Scope

- **Egyptian National IDs only.** The layout coordinates, the checksum,
  the governorate table and the field vocabularies are all Egypt-specific.
  Another country's ID will not work — hence the package name.
- **Modern cards.** Older layouts are not handled.
- **One card per image.** It can be tightly cropped, have a background
  margin, or sit small on a scanned page, at any rotation — but every
  pixel of the photo that is not card is resolution the fields don't get.
- **English transliteration is machine-generated.** `first_name_english`
  and friends are for display and search. They are *not* the official
  spelling on the holder's documents, which cannot be derived from the
  Arabic — محمد is legitimately Mohamed, Mohammed, Muhammad or Mohamad.
  Never use these for identity matching.
- **No liveness or forgery detection.** This reads a card; it does not
  tell you the card is genuine.

## Reproducing these numbers

The benchmark card is a real identity document and is not distributed.
`assets/own_benchmark/` is excluded by `.gitignore` in full, and
`tests/test_own_benchmark.py` skips itself when the folder is absent.

To benchmark your own cards, place them in `assets/own_benchmark/` using
the names in `scripts/benchmark_own.py`, then:

```bash
python scripts/benchmark_own.py --write-expected  # record the baseline
python scripts/benchmark_own.py                   # check it still holds
```

Verify the generated `expected.json` by eye before trusting it — it is
whatever the pipeline read, not ground truth.
