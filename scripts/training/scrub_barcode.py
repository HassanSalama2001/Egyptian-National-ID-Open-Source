"""
Overwrites the 2D barcode region in assets/Egyptian_ID_Card.jpg and
assets/back_template.jpg with random noise.

Why: the barcode on a real Egyptian ID card encodes the holder's data.
assets/Egyptian_ID_Card.jpg is a stock two-sided card image that predates
this project (in the repo since its initial commit) and its provenance -
whether it's a genuine photograph of a real person's card or a purpose-
built mock-up - could not be established from the file itself (checked:
no EXIF). assets/back_template.jpg was cut from it (confirmed by a
shift-robust cross-correlation of the barcode band: 0.967 against the
stock image's barcode, vs 0.083 - chance level - against a real,
unrelated card).

The pipeline never reads or decodes this barcode (see
core/layout_analyzer.py's BACK_FIELDS - no barcode field exists), so
scrubbing it costs nothing functionally. Run once; the barcode-shaped
region is located automatically (by counting binarized edge-transitions
per row - a dense 2D barcode has far more black/white transitions per
row than card text, emblems, or the hologram graphic) rather than
hand-picked coordinates, so this is re-runnable if either source image
changes.

    python scripts/training/scrub_barcode.py
"""
import sys

import cv2
import numpy as np

RNG_SEED = 42

TARGETS = [
    "assets/Egyptian_ID_Card.jpg",
    "assets/back_template.jpg",
]


def find_barcode_band(gray: np.ndarray) -> tuple[int, int]:
    h, w = gray.shape
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    x0, x1 = int(w * 0.05), int(w * 0.95)
    transitions = np.abs(np.diff(binary[:, x0:x1].astype(np.int16), axis=1)).sum(axis=1) / 255

    kernel = np.ones(5) / 5
    smoothed = np.convolve(transitions, kernel, mode="same")
    threshold = smoothed.max() * 0.4
    band_rows = np.where(smoothed > threshold)[0]
    if len(band_rows) == 0:
        raise RuntimeError("no barcode-like band found")

    gaps = np.where(np.diff(band_rows) > 20)[0]
    runs = np.split(band_rows, gaps + 1)
    best_run = max(runs, key=len)
    pad = int(h * 0.015)
    return max(0, int(best_run.min()) - pad), min(h, int(best_run.max()) + pad)


def scrub(path: str) -> None:
    image = cv2.imread(path)
    if image is None:
        raise SystemExit(f"Could not load {path}")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    y0, y1 = find_barcode_band(gray)

    h, w = image.shape[:2]
    x0, x1 = int(w * 0.05), int(w * 0.95)
    rng = np.random.RandomState(RNG_SEED)
    # Small-block noise, not per-pixel noise - visually approximates a 2D
    # barcode's grid texture (rather than looking like an obviously-edited
    # flat rectangle) without encoding anything real.
    block = 6
    small = rng.randint(0, 2, size=((y1 - y0) // block + 1, (x1 - x0) // block + 1), dtype=np.uint8) * 255
    noise = cv2.resize(small, (x1 - x0, y1 - y0), interpolation=cv2.INTER_NEAREST)
    noise_bgr = cv2.cvtColor(noise, cv2.COLOR_GRAY2BGR)

    image[y0:y1, x0:x1] = noise_bgr
    cv2.imwrite(path, image, [cv2.IMWRITE_JPEG_QUALITY, 95])
    print(f"Scrubbed {path}: barcode band y=({y0},{y1}) x=({x0},{x1})")


def main():
    for path in TARGETS:
        scrub(path)


if __name__ == "__main__":
    main()
