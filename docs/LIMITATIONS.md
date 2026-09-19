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
| `issue_date` | ✅ | ✅ | ✅ | ✅ |
| `expiry_date` | ✅ | ✅ | year+month only | ✅ |
| `profession` | ✅ | ✅ | last word missing | last word missing |
| `gender`, `religion`, `marital_status` | ✅ | ✅ | ✅ | ✅ |

The two fields that matter most — the **national ID number** and the
**names** — read correctly on every variant tested. Everything derived
from the national ID (birth date, governorate, gender, century) is
checksum-validated, so it is as reliable as the number itself.

### `address` drifts by a character on filtered scans
One extra character on enhanced/greyscale, several wrong on eco. This is
recogniser noise on a long free-text line, not truncation — the whole
address is found, some glyphs are read wrong. No validation exists that
could catch it, because an address has no checkable structure.

### `profession` loses its last word on greyscale/eco
The detector never proposes the final word as a text region, so nothing
downstream can recover it. Re-reading the region upscaled 3x fixes this
on a full-resolution card image but **not** on a zoomed-out scan, where
the pixels simply are not there. Re-running it everywhere cost fronts
4.4s → 12.6s and fixed nothing, so it is restricted to regions that come
back completely empty.

### `expiry_date` loses its day on greyscale
The digits misread as an impossible month (`2031/81/01`). Rather than
show that or blank the field, the 7-year validity rule recovers the year
and month from the issue date — `2031/08`. **The day is never invented.**
`is_expired` answers normally in every month except the expiry month
itself, where the missing day is exactly what decides it; there it
returns `null`.

## Speed

| Scan | Front | Back |
|---|---|---|
| Unfiltered | 5.3s | 4.2s |
| Enhanced | 5.6s | 9.0s |
| Greyscale | 5.9s | 8.0s |
| Eco | 4.9s | **25.1s** |

The eco back is the outlier: its degraded digits fail their validators,
which triggers the per-field offset search across the full binarization
ladder. It is correct, just slow.

These are CPU timings on one machine. Treat them as ratios, not promises.

## Scope

- **Egyptian National IDs only.** The layout coordinates, the checksum,
  the governorate table and the field vocabularies are all Egypt-specific.
  Another country's ID will not work — hence the package name.
- **Modern cards.** Older layouts are not handled.
- **One card per image**, occupying a reasonable share of the frame.
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
