"""
Measures get_card_contour/align_card's success rate across the real-world
conditions this project needs to handle - well-framed vs. zoomed-out
framing, and color vs. black-and-white (many real IDs are scanned/
photocopied, not photographed) - instead of judging the "is card detection
robust now" question from a single anecdotal screenshot.

Generates small batches with generate_realistic_photos.py's own
compositing logic (card on a varied background, random angle/position) at
each of 4 conditions, then reports what fraction of each batch gets a
CORRECT alignment - not just "M is not None".

That distinction matters: an earlier version of this script only checked
whether align_card returned a contour at all, which missed a real bug
where get_card_contour returned a confidently WRONG region (a smaller,
high-contrast internal texture blob that happened to also pass the
aspect-ratio check) instead of failing honestly. "Found A card-shaped
region" and "found THE card" are different claims - this now checks the
second one, via IoU against the known ground-truth placement.

Usage:
    python scripts/training/stress_test_card_detection.py --count 40
"""
import argparse
import random
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from generate_trial_ids import generate_sample
from generate_realistic_photos import random_background, composite_at_angle
from egyptian_national_id_ocr.core.layout_analyzer import LayoutAnalyzer


def iou(box_a: tuple, box_b: tuple) -> float:
    ax0, ay0, ax1, ay1 = box_a
    bx0, by0, bx1, by1 = box_b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    inter = max(0, ix1 - ix0) * max(0, iy1 - iy0)
    area_a = (ax1 - ax0) * (ay1 - ay0)
    area_b = (bx1 - bx0) * (by1 - by0)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def run_condition(label: str, count: int, seed: int, min_scale: float, max_scale: float,
                   grayscale: bool, tmp_dir: Path) -> None:
    rng = random.Random(seed)
    analyzer = LayoutAnalyzer()
    n_correct = 0
    n_wrong_region = 0  # found *something* card-shaped, but not the right place
    n_fallback = 0

    for i in range(1, count + 1):
        entry = generate_sample(i, tmp_dir)
        card_img = Image.open(tmp_dir / entry["filename"]).convert("RGBA")
        canvas_w, canvas_h = int(card_img.width * 1.4), int(card_img.height * 1.6)
        background = random_background(canvas_w, canvas_h, rng)
        photo, gt_bbox = composite_at_angle(card_img, background, rng, min_scale, max_scale)
        photo = photo.convert("RGB")
        if grayscale:
            photo = photo.convert("L").convert("RGB")

        image = cv2.cvtColor(np.array(photo), cv2.COLOR_RGB2BGR)
        _, M = analyzer.align_card(image)
        if M is None:
            n_fallback += 1
            continue

        contour = analyzer.get_card_contour(image)
        if contour is None:
            # M came from the whole-frame aspect-ratio shortcut, not an
            # interior contour - not applicable to this synthetic scene
            # (which always has real background around the card), so
            # treat as a miss for measurement purposes here.
            n_wrong_region += 1
            continue
        x, y, w, h = cv2.boundingRect(contour)
        overlap = iou((x, y, x + w, y + h), gt_bbox)
        if overlap >= 0.5:
            n_correct += 1
        else:
            n_wrong_region += 1

    pct = n_correct / count * 100
    print(f"{label:35s}  correct={n_correct:3d}/{count} ({pct:5.1f}%)  "
          f"wrong_region={n_wrong_region}  no_detection={n_fallback}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=40)
    parser.add_argument("--seed", type=int, default=777)
    args = parser.parse_args()

    tmp_dir = PROJECT_ROOT / "data" / "_stress_test_card_detection_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    print(f"Card-detection stress test ({args.count} samples per condition)")
    print("-" * 80)
    run_condition("Well-framed, color (baseline)", args.count, args.seed, 0.5, 0.85, False, tmp_dir)
    run_condition("Zoomed out, color", args.count, args.seed + 1, 0.15, 0.4, False, tmp_dir)
    run_condition("Well-framed, black & white", args.count, args.seed + 2, 0.5, 0.85, True, tmp_dir)
    run_condition("Zoomed out, black & white", args.count, args.seed + 3, 0.15, 0.4, True, tmp_dir)

    import shutil
    shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
