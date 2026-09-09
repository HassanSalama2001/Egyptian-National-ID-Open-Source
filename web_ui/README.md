# Web UI

React + Vite frontend for the National ID OCR pipeline. Deliberately transparent about what the
pipeline actually detected: every field shows the exact cropped region it was read from next to
the extracted value, plus the whole detected/aligned card and an honest status badge
(`success` / `low_confidence` / `no_card_detected` / `unreadable_image`) — nothing is hidden or
inferred client-side.

## Setup

```bash
npm install
cp .env.example .env   # points at the backend; edit if it's not on localhost:8000
npm run dev
```

Needs the backend running separately (`python src/app.py` from the project root, or
`.claude/launch.json`'s `backend` config if using Claude Code's preview tooling).

## Notes

- Values from Arabic fields (`first_name`, `full_name`, `address`) render right-to-left; numeric
  fields don't.
- The "Show raw response" toggle dumps the exact API JSON — useful for debugging without
  reaching for devtools.
- Input contract matches `Pipeline.process_image`'s docstring: this expects a well-framed photo
  (card filling the frame), not an arbitrary photo of a card on a table. "Use your camera instead"
  on the upload screen opens a live guided-crop capture (`CameraCapture.jsx`) — an on-screen card
  outline the user aligns their card to, cropped client-side to that outline before upload (see
  `src/lib/cropMath.js` for the coordinate mapping, and its `cropMath.test.mjs` for the geometry
  checks: `node src/lib/cropMath.test.mjs`). That crop is what makes "guided-crop" work end to end —
  it delivers the backend near-edge-to-edge input, sidestepping `align_card`'s free-form-detection
  bug (see `Pipeline.process_image`'s docstring) rather than requiring a backend fix. File upload
  still works as a fallback (no camera, desktop testing, existing photos).
- Camera capture hasn't been hand-tested against a real webcam yet (the automated tooling used to
  build this has no physical camera) — verified so far: the crop-mapping math in isolation, the
  existing upload path still works unaffected, and the permission-denied/no-camera fallback UI.
  Needs a real hands-on pass on a machine with a webcam before relying on it.
