# Security Policy

## Reporting a vulnerability

Please **do not** open a public GitHub issue for a security vulnerability.

Instead, use GitHub's private
[security advisory form](https://github.com/HassanSalama2001/Egyptian-National-ID-Open-Source/security/advisories/new)
for this repository, or email the maintainer directly (see the
[commit history](https://github.com/HassanSalama2001/Egyptian-National-ID-Open-Source/commits/main)
for a contact address).

Include, where possible:

- What the vulnerability is and its potential impact.
- Steps to reproduce it (a minimal example, not a real ID card).
- Which version/commit you tested against.

You should get an acknowledgement within a few days. This is a
maintainer-hours-limited project, not a company with an SLA - please be
patient, and thank you for reporting responsibly rather than publicly.

## Supported versions

Only the latest released version is supported with security fixes. There
is no long-term-support branch at this stage of the project.

## What "private and local" actually means here

This pipeline's central design claim is that extraction happens entirely
on the machine running it - no card image or extracted field is sent
anywhere by this library. That claim is checked directly by
[`tests/test_privacy_no_network.py`](tests/test_privacy_no_network.py),
which blocks real outbound network connections during a pipeline run and
fails if one is attempted, rather than just asserting the behavior in
prose.

Two things that claim does **not** cover, and you're responsible for if
you deploy this:

- **The HTTP API and web UI** are, if you expose them, whatever you
  expose them as. Nothing in this repo phones home, but a public `/ocr`
  endpoint is still a public endpoint - see
  [`docs/LIMITATIONS.md`](docs/LIMITATIONS.md) for what's already
  hardened (CORS allowlist, upload size cap, a processing-time budget)
  and think about what else your deployment needs (authentication, rate
  limiting, TLS termination) before putting it on the internet.
- **Your own data handling.** If you build something on top of this that
  *does* store card images or extracted fields (the planned Continuous
  Learning feature, for instance, is opt-in and explicitly local-only by
  design - see the project's roadmap) - that's your responsibility to
  secure, not something this library does for you.

## Known limitations that are not security issues

[`docs/LIMITATIONS.md`](docs/LIMITATIONS.md) documents accuracy gaps
(fields that don't read correctly on some scan conditions) and
performance numbers. Those are quality issues, not vulnerabilities -
please file them as regular GitHub issues, not through the security
process above.
