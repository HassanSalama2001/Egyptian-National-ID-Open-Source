"""
One-time asset-cleanup script: erases the caption labels baked into
assets/front_template.jpg (e.g. "الإسم الاول", "تاريخ الميلاد") - these
were sitting exactly where generate_trial_ids.py now draws real values
(after this project's fields were recalibrated against a real card),
causing rendered values to visually collide with the template's own
label text and corrupting the synthetic benchmark (national_id exact
match dropped to 0/20).

This template is a generic mock asset already in the repo (a plain white
placeholder box, no real photo/personal data) - not derived from anyone's
real ID card, so it's safe to edit and commit.

Approach: a global color-based mask was tried first but the pyramid
silhouette's own shading has near-black, low-saturation pixels in its
darkest creases, indistinguishable from ink by color alone - it left
speckled inpainting artifacts across the graphic. Precise per-label
bounding boxes (measured directly off the template, see crop_labels.py
in this session's scratch dir) avoid that: Otsu-threshold dark pixels
*within* each small box only, so nothing outside a label's own footprint
is ever touched, then cv2.inpaint over just those pixels so the
surrounding texture stays intact instead of leaving a flat rectangle.
"""
import sys

import cv2
import numpy as np

TEMPLATE_PATH = "assets/front_template.jpg"
OUTPUT_PATH = sys.argv[1] if len(sys.argv) > 1 else "assets/front_template.jpg"

# Precise boxes (x0, y0, x1, y1), measured directly off the template by
# cropping and visually inspecting each label - not guessed.
LABEL_REGIONS = [
    (1900, 440, 2500, 600),   # "الإسم الاول" (first_name label)
    (1750, 580, 2500, 740),   # "باقي الإسم" (full_name label)
    (1720, 760, 2360, 910),   # "محل الإقامة" (address label)
    (1700, 1230, 2410, 1390), # "الرقم القومي" (national_id label)
    (190, 1190, 810, 1330),   # "تاريخ الميلاد" (birth_date label)
    (220, 1400, 750, 1550),   # "رقم المصنع" (serial_number label, over the blue border pattern)
    (250, 430, 750, 710),     # "الصورة الشخصية" (photo placeholder caption)
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
        # CLAHE first so a box straddling both the plain background and
        # the blue-border-pattern (serial_number) still gets a clean
        # ink-vs-background split from Otsu, instead of the busier area
        # dominating the histogram.
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        # A few more dilate iterations than the first pass - faint
        # anti-aliased glyph-edge pixels sat just below Otsu's cutoff and
        # survived as light ghosting after inpainting only the solid core.
        thresh = cv2.dilate(thresh, np.ones((3, 3), np.uint8), iterations=3)
        mask[y0:y1, x0:x1] = thresh

    cleaned = cv2.inpaint(image, mask, inpaintRadius=7, flags=cv2.INPAINT_TELEA)
    cv2.imwrite(OUTPUT_PATH, cleaned, [cv2.IMWRITE_JPEG_QUALITY, 95])
    print(f"Saved cleaned template to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
