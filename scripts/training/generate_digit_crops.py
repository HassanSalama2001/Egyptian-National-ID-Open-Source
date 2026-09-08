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

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

WESTERN_TO_ARABIC_INDIC = str.maketrans("0123456789", "٠١٢٣٤٥٦٧٨٩")
FONT_PATH = "C:/Windows/Fonts/tahomabd.ttf"
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

    return Image.fromarray(arr.astype(np.uint8))


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
                aug = augment(clean, rng)
                aug.save(output_dir / split / digit / f"{digit}_{idx:04}.png")
                idx += 1

    print(f"Generated digit crops in {output_dir} "
          f"({args.per_class} per class x 10 classes = {args.per_class*10} total)")


if __name__ == "__main__":
    main()
