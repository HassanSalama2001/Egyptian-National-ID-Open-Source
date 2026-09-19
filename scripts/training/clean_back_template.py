"""
Same treatment as clean_front_template.py, applied to assets/back_template.jpg -
erases the caption labels baked into the template (e.g. "الرقم القومي",
"الحالة الاجتماعية") that sit where generate_trial_ids.py draws real
values, which corrupts synthetic back-side benchmarking the same way it
did for the front. This template is a generic mock asset (stock
Tutankhamun hologram graphic, generic Egypt eagle emblem, generic
captions) - not derived from anyone's real ID card - so it's safe to
edit and commit.

Approach: per label region, threshold dark (text) pixels within a
generous bounding box (measured directly off the template - see this
session's scratch crops), then cv2.inpaint over just those pixels so the
surrounding wood-grain/pharaonic-engraving texture stays intact instead
of leaving a flat rectangle.
"""
import sys

import cv2
import numpy as np

TEMPLATE_PATH = "assets/back_template.jpg"
OUTPUT_PATH = sys.argv[1] if len(sys.argv) > 1 else "assets/back_template.jpg"

# Precise boxes (x0, y0, x1, y1), measured directly off the template via
# full-width band crops (not guessed) - see this session's scratch dir.
LABEL_REGIONS = [
    (1420, 110, 2170, 260),   # "الرقم القومي" (national_id label)
    (1750, 260, 2080, 380),   # "المهنة" (profession label)
    (660, 120, 1320, 260),    # "تاريخ الإصدار" (issue_date label)
    (1830, 480, 2220, 660),   # "النوع" (gender label)
    (1420, 480, 1740, 630),   # "الديانة" (religion label)
    (680, 480, 1300, 630),    # "الحالة الاجتماعية" (marital_status label)
    (1340, 700, 2060, 850),   # "تاريخ الانتهاء" (expiry_date label)
]


def main():
    image = cv2.imread(TEMPLATE_PATH)
    if image is None:
        raise SystemExit(f"Could not load {TEMPLATE_PATH}")

    h, w = image.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    for (x0, y0, x1, y1) in LABEL_REGIONS:
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(w, x1), min(h, y1)
        region = image[y0:y1, x0:x1]
        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        thresh = cv2.dilate(thresh, np.ones((3, 3), np.uint8), iterations=3)
        mask[y0:y1, x0:x1] = thresh

    cleaned = cv2.inpaint(image, mask, inpaintRadius=7, flags=cv2.INPAINT_TELEA)
    cv2.imwrite(OUTPUT_PATH, cleaned, [cv2.IMWRITE_JPEG_QUALITY, 95])
    print(f"Saved cleaned template to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
