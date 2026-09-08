"""
One-off calibration helper: measures where each field label sits on
assets/front_template.jpg using the same horizontal-projection technique
layout_analyzer.py uses for real-card segmentation, so the coordinates in
generate_trial_ids.py's FRONT_FIELDS aren't hand-guessed.

Re-run this (and update FRONT_FIELDS accordingly) if the template asset
ever changes. Prints label bounding boxes grouped by column; the label's
left edge (min x) is roughly where a value should end (right-aligned),
since values are drawn immediately to the left of their Arabic label.
"""
import cv2
import numpy as np


def find_text_bands(binary_region: np.ndarray, x_offset: int, min_pixels: int = 8, min_height: int = 15):
    h_proj = np.sum(binary_region > 0, axis=1)
    bands = []
    in_band = False
    start = 0
    for y, count in enumerate(h_proj):
        if count > min_pixels and not in_band:
            start, in_band = y, True
        elif count <= min_pixels and in_band:
            bands.append((start, y))
            in_band = False
    if in_band:
        bands.append((start, len(h_proj)))

    results = []
    for y0, y1 in bands:
        if y1 - y0 <= min_height:
            continue
        cols = np.where(np.sum(binary_region[y0:y1, :] > 0, axis=0) > 1)[0]
        if len(cols) == 0:
            continue
        results.append({
            "y_range": (y0, y1),
            "y_center": (y0 + y1) // 2,
            "x_range": (int(cols.min()) + x_offset, int(cols.max()) + x_offset),
        })
    return results


def main():
    template_path = "assets/front_template.jpg"
    img = cv2.imread(template_path)
    if img is None:
        raise SystemExit(f"Could not load {template_path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 90, 255, cv2.THRESH_BINARY_INV)

    print(f"Template: {template_path}  shape={img.shape}")
    print()

    # Right column: first_name, full_name, address, national_id labels
    print("--- Right column (x=1250-2350): name/address/NID labels ---")
    for band in find_text_bands(thresh[:, 1250:2350], x_offset=1250):
        print(f"  y_center={band['y_center']:4d}  y_range={band['y_range']}  "
              f"x_range={band['x_range']}  (value anchor ~ x={band['x_range'][0] - 50})")

    print()
    print("--- Left column (x=0-1200): DOB/serial labels ---")
    for band in find_text_bands(thresh[:, 0:1200], x_offset=0):
        print(f"  y_center={band['y_center']:4d}  y_range={band['y_range']}  "
              f"x_range={band['x_range']}")


if __name__ == "__main__":
    main()
