"""
Composites correctly-calibrated synthetic front-card renders onto a
neutral background at a random angle/scale/position, simulating a real
"phone photo of a card on a table" - as opposed to data/synthetic_front/,
which is the card edge-to-edge (no detection needed, just the fixed-
box extraction path).

This exercises the OTHER orientation-handling path: align_card's contour-
based perspective correction, which should handle arbitrary angles (not
just 90-degree multiples - that's what Pipeline.ROTATIONS_TO_TRY is for,
a different, complementary case: no visible card boundary to detect at
all, e.g. an already-cropped edge-to-edge image).

Reuses generate_trial_ids.py's field rendering so the ground truth stays
exactly labeled, then composites onto a plain but varied background.

SCOPE NOTE: running this revealed that assets/front_template.jpg has a
white backing/margin baked into the source pixels (from the original
photographed template), and align_card's contour detector locks onto
that outer white rectangle instead of the actual card content when a
photo has both - a real bug, and one that would likely affect real
photos of a card placed on a white table/paper too. Fixing this properly
means either stripping the margin from the template or making contour
selection prefer colorful content over a plain backing - real work, not
yet done. The product decision (see git log) was to ship with a
guided-crop UI (user aligns the card in an on-screen frame before
submitting, like most banking-app ID scanners) rather than chase
free-form detection - that's what's actually accuracy-tested end to end
(data/synthetic_front/, 98%+). This script and the free-form path stay
in the repo for whoever picks up that harder problem later; the shipped
pipeline does not depend on it.
"""
import argparse
import json
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from generate_trial_ids import generate_sample, PROJECT_ROOT


def random_background(width: int, height: int, rng: random.Random) -> Image.Image:
    """A plain-ish background with some texture/color variation, standing
    in for a table/desk/floor surface - doesn't need to be photorealistic,
    just needs to give align_card's contour detection an actual boundary
    to find (unlike the edge-to-edge synthetic_front samples)."""
    base_color = tuple(rng.randint(40, 200) for _ in range(3))
    bg = Image.new("RGB", (width, height), base_color)
    draw = ImageDraw.Draw(bg)
    # A few random darker/lighter blobs so it's not perfectly flat
    for _ in range(rng.randint(3, 8)):
        x0, y0 = rng.randint(0, width), rng.randint(0, height)
        w, h = rng.randint(50, 300), rng.randint(50, 300)
        shade = tuple(max(0, min(255, c + rng.randint(-30, 30))) for c in base_color)
        draw.ellipse([x0, y0, x0 + w, y0 + h], fill=shade)
    return bg


def composite_at_angle(card_img: Image.Image, background: Image.Image, rng: random.Random) -> Image.Image:
    angle = rng.uniform(-25, 25)
    # Scale the card down to occupy a random fraction of the frame width,
    # like a real photo where the card doesn't fill the whole shot
    scale = rng.uniform(0.5, 0.85)
    target_w = int(background.width * scale)
    target_h = int(card_img.height * (target_w / card_img.width))
    resized = card_img.resize((target_w, target_h), Image.LANCZOS)
    # fillcolor must be explicit and fully transparent (alpha=0) - PIL's
    # rotate() default for RGBA does NOT reliably produce a transparent
    # fill, it can leave the padding opaque white, which then reads as a
    # much stronger/cleaner edge than the actual card boundary to
    # align_card's contour detector than the real card edges do.
    rotated = resized.rotate(angle, expand=True, resample=Image.BICUBIC, fillcolor=(0, 0, 0, 0))

    max_x = max(1, background.width - rotated.width)
    max_y = max(1, background.height - rotated.height)
    pos = (rng.randint(0, max_x), rng.randint(0, max_y))

    canvas = background.copy()
    canvas.paste(rotated, pos, rotated.convert("RGBA") if rotated.mode != "RGBA" else rotated)
    return canvas


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--output", default=str(PROJECT_ROOT / "data" / "realistic_photos"))
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = []

    from tqdm import tqdm
    for i in tqdm(range(1, args.count + 1), desc="Generating realistic photos"):
        # Render the clean card (reusing the calibrated generator) to a temp dir
        tmp_dir = output_dir / "_clean_tmp"
        entry = generate_sample(i, tmp_dir)
        card_img = Image.open(tmp_dir / entry["filename"]).convert("RGBA")

        canvas_w, canvas_h = int(card_img.width * 1.4), int(card_img.height * 1.6)
        background = random_background(canvas_w, canvas_h, rng)
        photo = composite_at_angle(card_img, background, rng).convert("RGB")

        filename = f"photo_{i:04}.jpg"
        photo.save(output_dir / filename, quality=88)
        manifest.append({"filename": filename, "data": entry["data"]})

    with open(output_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    # Clean up the temp clean-card renders
    import shutil
    shutil.rmtree(output_dir / "_clean_tmp", ignore_errors=True)

    print(f"Generated {args.count} realistic photos in {output_dir}")


if __name__ == "__main__":
    main()
