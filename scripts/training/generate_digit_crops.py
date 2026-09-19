"""
Generates labeled single-digit crops (Arabic-Indic glyphs ٠-٩) for training
a lightweight digit classifier for the NID/serial-number fields.

Unlike Arabic letters, Arabic-Indic digits don't connect/reshape based on
neighbors, so isolated single-digit images are a faithful stand-in for what
character-segmentation will hand the classifier at inference time. Each
crop is rendered at on-card scale then perturbed with augmentations that
approximate real photo conditions (rotation, blur, noise, contrast,
JPEG compression) so the classifier isn't just memorizing clean renders.
"""
import argparse
import random
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

WESTERN_TO_ARABIC_INDIC = str.maketrans("0123456789", "٠١٢٣٤٥٦٧٨٩")
FONT_PATH = "C:/Windows/Fonts/tahomabd.ttf"
# Tried Simplified Arabic Bold here after visually confirming it's a much
# closer glyph-shape match to a real card than Tahoma - and it DID fix
# the persistent '٣' -> '٢' misread. But real-card retesting then showed
# a net-worse result: 3 simultaneous wrong digits instead of 1 (too many
# for the checksum-repair's ambiguity-safe logic to fix at all), because
# this whole family of calligraphic Arabic fonts (also checked Sakkal
# Majalla and Traditional Arabic - same underlying Naskh-style digit
# shapes) has '٢'/'٧' as similar angular hooks and '٥'/'٠' as similar
# round blobs - thinner, more delicate distinguishing strokes than
# Tahoma's bold geometric ones, and so more fragile under real-world
# blur/compression. Reverted to Tahoma as the net-better, known-quantity
# baseline. Properly fixing this needs a bigger, more carefully-curated
# training effort (e.g. real print-and-rescan samples in the correct
# font) than a same-session font swap, not just picking a different font
# file - every calligraphic-style candidate tried shares this weakness.
CANVAS = 64          # output crop size (square)
BASE_FONT_SIZE = 44  # roughly matches on-card digit height after crop+resize


def render_clean_digit(digit: str, font_size: int) -> Image.Image:
    glyph = digit.translate(WESTERN_TO_ARABIC_INDIC)
    font = ImageFont.truetype(FONT_PATH, font_size)
    img = Image.new("L", (CANVAS, CANVAS), color=255)
    draw = ImageDraw.Draw(img)
    bbox = draw.textbbox((0, 0), glyph, font=font)
    w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    x = (CANVAS - w) // 2 - bbox[0]
    y = (CANVAS - h) // 2 - bbox[1]
    draw.text((x, y), glyph, font=font, fill=0)
    return img


def augment(img: Image.Image, rng: random.Random) -> Image.Image:
    # Rotation
    angle = rng.uniform(-6, 6)
    img = img.rotate(angle, resample=Image.BICUBIC, fillcolor=255)

    # Scale jitter (simulates imperfect field-crop sizing)
    scale = rng.uniform(0.85, 1.15)
    new_size = max(8, int(CANVAS * scale))
    img = img.resize((new_size, new_size), Image.LANCZOS)
    canvas = Image.new("L", (CANVAS, CANVAS), color=255)
    offset = ((CANVAS - new_size) // 2, (CANVAS - new_size) // 2)
    canvas.paste(img, offset)
    img = canvas

    # Blur (simulates focus/motion blur from a phone photo)
    if rng.random() < 0.6:
        img = img.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.3, 1.2)))

    arr = np.array(img).astype(np.float32)

    # Brightness/contrast jitter
    contrast = rng.uniform(0.7, 1.3)
    brightness = rng.uniform(-25, 25)
    arr = (arr - 128) * contrast + 128 + brightness
    arr = np.clip(arr, 0, 255)

    # Gaussian noise
    if rng.random() < 0.7:
        noise = np.random.normal(0, rng.uniform(3, 15), arr.shape)
        arr = np.clip(arr + noise, 0, 255)

    # Resolution-loss (downscale-then-upscale) and JPEG-artefact
    # augmentation were added here and REVERTED. The theory was sound -
    # training glyphs are rendered full-size while real back-of-card
    # digits arrive under-sampled - and the isolated stress test agreed,
    # rising 91.9% -> 93.4%. But A/B'd over identical seeded cards at the
    # FIELD level it moved neither target (eco scan-chain back.national_id
    # 6.0% -> 5.4% per-digit, back.expiry_date 43.8% -> 44.8%) while
    # dropping front.birth_date from 100% to 75-83% across every scan
    # condition. Another reminder that the isolated-glyph benchmark does
    # not predict field accuracy - and that back.national_id's real
    # problem is SEGMENTATION (it finds the right digit count only 41.7%
    # of the time), which no amount of classifier training can fix.
    return Image.fromarray(arr.astype(np.uint8))


def to_inference_form(img: Image.Image, rng: random.Random = None) -> Image.Image:
    """
    Puts a training crop through the SAME transformation the inference
    path applies, so the classifier trains on what it will actually be
    asked to classify.

    This closes a train/inference mismatch that had existed since the
    classifier was written: DigitClassifierEngine._find_digit_boxes cuts
    each digit out of a THRESHOLDED image, so at inference every glyph
    arrives as a hard binary mask, isolated and re-centred on a fixed
    canvas. Training, meanwhile, used smooth anti-aliased renders with
    blur and noise still on them - a visibly different distribution.
    Measured consequence: per-digit accuracy sat at ~86% on ordinary
    scans and collapsed to 6% (worse than guessing) on a hard
    black-and-white "eco" scan, because binary input was effectively
    out-of-distribution.

    Mirrors _binarize's Otsu path plus _prepare_crop's isolate/scale/
    centre/invert, so a generated crop is pixel-wise the same kind of
    object the model sees in production.
    """
    arr = np.array(img)

    # When an rng is supplied (training-data generation), also model what
    # a scanner app's black-and-white filter does to the glyph BEFORE our
    # pipeline ever sees it. Real "eco" scans of a card misread exactly
    # the digits whose distinguishing feature is a thin stroke ('٣' read
    # as '٢', '٥' as '٠'), because harsh binarization erodes that stroke
    # away. Training only on cleanly-thresholded glyphs leaves the
    # classifier no experience of it.
    if rng is not None and rng.random() < 0.3:
        _, arr = cv2.threshold(arr, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    denoised = cv2.fastNlMeansDenoising(arr, None, 10, 7, 21)
    _, thresh = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Stroke-thickness jitter, for the same reason - a glyph thinned or
    # fattened by a pixel is a different-looking shape to HOG features,
    # and both happen in real scans depending on the filter used.
    if rng is not None:
        roll = rng.random()
        if roll < 0.25:
            thinned = cv2.erode(thresh, np.ones((2, 2), np.uint8), iterations=1)
            # Erosion can wipe out a light glyph entirely - keep the
            # original rather than emit a blank, mislabelled sample.
            if thinned.any():
                thresh = thinned
        elif roll < 0.45:
            thresh = cv2.dilate(thresh, np.ones((2, 2), np.uint8), iterations=1)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return img
    x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
    glyph = thresh[y:y + h, x:x + w]

    scale = (CANVAS * 0.7) / max(h, w)
    new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
    resized = cv2.resize(glyph, (new_w, new_h), interpolation=cv2.INTER_AREA)

    canvas = np.zeros((CANVAS, CANVAS), dtype=np.uint8)
    x_off = (CANVAS - new_w) // 2
    y_off = (CANVAS - new_h) // 2
    canvas[y_off:y_off + new_h, x_off:x_off + new_w] = resized
    return Image.fromarray(255 - canvas)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-class", type=int, default=800)
    parser.add_argument("--output", default="data/digits")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    np.random.seed(args.seed)
    output_dir = Path(args.output)

    for split, frac in [("train", 0.7), ("val", 0.15), ("test", 0.15)]:
        for digit in "0123456789":
            (output_dir / split / digit).mkdir(parents=True, exist_ok=True)

    from tqdm import tqdm
    for digit in tqdm("0123456789", desc="Classes"):
        n_train = int(args.per_class * 0.7)
        n_val = int(args.per_class * 0.15)
        n_test = args.per_class - n_train - n_val
        counts = {"train": n_train, "val": n_val, "test": n_test}

        idx = 0
        for split, count in counts.items():
            for _ in range(count):
                font_size = int(BASE_FONT_SIZE * rng.uniform(0.9, 1.1))
                clean = render_clean_digit(digit, font_size)
                aug = to_inference_form(augment(clean, rng), rng)
                aug.save(output_dir / split / digit / f"{digit}_{idx:04}.png")
                idx += 1

    print(f"Generated digit crops in {output_dir} "
          f"({args.per_class} per class x 10 classes = {args.per_class*10} total)")


if __name__ == "__main__":
    main()
