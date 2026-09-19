# Contributing

Thanks for considering it. This is a small, maintainer-hours-limited
project - please be patient with review times, and know that a slow
response isn't a no.

## Before you start

For anything beyond a small fix (a new feature, a change to the
extraction pipeline's behavior, a new dependency), open an issue first
to discuss the approach. It's a much smaller loss to hear "this won't
land" before you've written the code than after.

## Setup

```bash
git clone https://github.com/HassanSalama2001/National-ID-Open-Source.git
cd National-ID-Open-Source
pip install -e ".[mcp,training]"
```

## Before opening a PR

```bash
python -m pytest tests -q
```

The suite is designed to pass on a clone with no private data: benchmark
and packaging tests that need real card images or a built wheel skip
themselves. If you change anything in `core/`, `ocr/`, or
`postprocessing/`, also run:

```bash
python scripts/training/stress_test_digit_fields.py
python scripts/training/stress_test_card_detection.py
```

These use the seeded synthetic generator (`--seed 42`), so results are
reproducible - if your change moves a number, that's a real effect of
your change, not run-to-run noise. **An accuracy or speed regression on
these blocks the PR**, the same way a failing test would - see
[`docs/RELEASING.md`](docs/RELEASING.md) for why a number moving is
treated as seriously as code breaking.

## How this codebase is meant to be worked on

The comments throughout `core/pipeline.py` and `core/layout_analyzer.py`
aren't incidental - many of them record a specific change that was tried,
measured against real or seeded-synthetic cards, and either kept or
reverted, with the numbers that decided it. That history is there on
purpose: a fix that looks obviously correct in isolation has repeatedly
turned out to help one scan condition while quietly breaking another
(see `docs/LIMITATIONS.md`'s accuracy table - several rows are the result
of exactly that kind of regression being caught before it shipped).

If you're changing extraction behavior:

1. **Measure before changing.** Know what the current behavior actually
   is on real or seeded data, not what you'd expect it to be from
   reading the code.
2. **A/B on identical data.** Compare before/after on the same seeded
   images (or, if you have real cards to test with locally, your own
   `assets/own_benchmark/` - see `scripts/benchmark_own.py`). An
   unseeded comparison can swing several percentage points from run to
   run for reasons that have nothing to do with your change.
3. **Revert if there's no real gain.** A plausible-sounding idea that
   measures as a wash (or a regression on some other case) should be
   reverted, not kept because it seemed like it should have helped.
4. **Say what you measured** in the PR description, not just what you
   changed - the next person reading the code six months from now
   (possibly you) needs the "why", not just the "what".

## Adding a new field or extraction capability

Please include:

- The layout coordinates (calibrated against a template, not eyeballed -
  see `scripts/training/calibrate_template.py`).
- A synthetic test case generated via
  `scripts/training/generate_trial_ids.py`, so the change is verifiable
  without a real ID card.
- An update to `docs/LIMITATIONS.md` if the change affects accuracy or
  speed for any existing field.

## What never belongs in a PR

- **No real ID card images or data**, yours or anyone else's, in any
  form - not as a test fixture, not in a commit message, not in a code
  comment. This has happened before in this repo's history and been
  scrubbed; don't reintroduce it. If your fix needs a real card to
  develop against, test it locally and describe the fix in the PR
  without the image.
- **No cloud/LLM API calls added to the core pipeline.** "Local, no
  cloud APIs" is this project's central design constraint, not a
  preference - see `tests/test_privacy_no_network.py`, which enforces it
  directly rather than just documenting it.

## Code of Conduct

This project follows [`CODE_OF_CONDUCT.md`](CODE_OF_CONDUCT.md).
