# Releasing

One version number governs the Python package; the JavaScript client, when
it exists, tracks it. Every release starts from a green benchmark.

## Before any release

```bash
python -m pytest tests -q          # unit + integration
python scripts/benchmark_own.py    # accuracy + speed on real cards
```

The benchmark must not regress. If a field that was passing now fails,
that is a release blocker, not a changelog footnote. If a field listed in
`KNOWN_GAPS` starts passing, the test fails on purpose — remove the entry
and say so in the changelog.

## Cutting a version

1. Move `[Unreleased]` entries in `CHANGELOG.md` under the new version.
2. Update `docs/LIMITATIONS.md` if any measured number moved. Re-run the
   benchmark and paste the real figures; do not carry old ones forward.
3. Bump `version` in `pyproject.toml`.
4. Tag `vX.Y.Z` and push.

### What counts as which bump

- **Major** — a field is removed or renamed, a response shape changes, or
  a default flips. Anything that breaks a caller.
- **Minor** — new fields, new endpoints/tools, or a measured accuracy
  improvement.
- **Patch** — fixes and speedups that change no interface.

An accuracy change is never a patch, even when no code signature moves:
callers calibrate against the numbers in LIMITATIONS, so those numbers are
part of the interface.

## Python package

```bash
python -m build
python -m twine upload dist/*
```

## JavaScript client

**Not built yet.** There is a React demo in `web_ui/` (private, not
published) but no npm client package. When one is added it should:

- be named to match the Python distribution,
- carry the same version number as the Python release it was tested
  against, so a user can tell at a glance which server it matches,
- and be published only after the benchmark above passes.

Until it exists, do not advertise npm support in the README.
