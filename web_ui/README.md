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
  (card filling the frame), not an arbitrary photo of a card on a table. There's no in-browser
  crop-guide overlay yet — the upload copy just asks for it. Adding a real guided-crop capture UI
  (on-screen card outline, live camera capture) is the natural next step here.
