"""
Measures how precisely the card is cropped, before any OCR runs.

Every field box is a fixed coordinate in the 1200x750 rectified card, so
the only number that matters for preprocessing is: after alignment, how
many pixels is each card corner away from where it should be? That is
exactly how far every field box will be off. This script builds synthetic
scenes with a KNOWN card placement (a homography from the 1200x750 card
to the photo), runs alignment, and reports the mean corner error in that
1200x750 space - including orientation, since a card warped upside down
or sideways has a corner error of hundreds of pixels.

Scenes reproduce the failure types seen on real third-party scans (no
real cards are used or stored):
  tight_dark / tight_light   card fills the photo with a thin background
                             margin - margins shifted every field box
  exact_tight                no margin at all - must NOT be trimmed
  portrait_page              card small and sideways on a white A4 scan
  general / general_gray     angled, perspective, textured background
  upside_down                card rotated ~180 degrees
  inverted_whole             negative scan (whole image inverted)
  inverted_card              dark/inverted card on a white page

Usage:
    python scripts/training/benchmark_card_crop.py --count 12 --seed 7
    python scripts/training/benchmark_card_crop.py --e2e   # also run OCR
"""
import argparse
import random
import sys
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from generate_trial_ids import generate_sample  # noqa: E402

CARD_W, CARD_H = 1200, 750
CANON = np.array([[0, 0], [CARD_W - 1, 0], [CARD_W - 1, CARD_H - 1], [0, CARD_H - 1]],
                 dtype=np.float32)
PASS_PX = 12.0  # mean corner error, in 1200x750 card pixels

CONDITIONS = {
    #                 canvas(w,h)   card width frac  angle        persp  bg        gray  invert
    "tight_dark":    ((1300, 830), (0.90, 0.96), (-2, 2),     0.00, "dark",   False, None),
    "tight_light":   ((1300, 830), (0.90, 0.96), (-2, 2),     0.00, "light",  False, None),
    "exact_tight":   (None,        None,          (0, 0),      0.00, None,     False, None),
    "portrait_page": ((1240, 1754), (0.30, 0.45), (87, 93),   0.01, "white",  True,  None),
    "general":       ((1600, 1200), (0.40, 0.75), (-20, 20),  0.04, "texture", False, None),
    "general_gray":  ((1600, 1200), (0.40, 0.75), (-20, 20),  0.04, "texture", True,  None),
    "upside_down":   ((1600, 1200), (0.45, 0.75), (170, 190), 0.03, "texture", False, None),
    "inverted_whole": ((1300, 830), (0.85, 0.95), (-3, 3),    0.00, "light",  True,  "whole"),
    "inverted_card": ((1600, 1200), (0.45, 0.70), (-8, 8),    0.02, "white",  True,  "card"),
}


def background(kind: str, w: int, h: int, rng: random.Random) -> np.ndarray:
    if kind == "white":
        v = rng.randint(235, 252)
        bg = np.full((h, w, 3), v, np.uint8)
    elif kind == "light":
        bg = np.full((h, w, 3), [rng.randint(190, 235) for _ in range(3)], np.uint8)
    elif kind == "dark":
        bg = np.full((h, w, 3), [rng.randint(15, 70) for _ in range(3)], np.uint8)
    else:  # texture
        bg = np.full((h, w, 3), [rng.randint(40, 200) for _ in range(3)], np.uint8)
        for _ in range(rng.randint(4, 9)):
            c = (rng.randint(0, w), rng.randint(0, h))
            ax = (rng.randint(40, 300), rng.randint(40, 300))
            col = [int(max(0, min(255, int(x) + rng.randint(-35, 35)))) for x in bg[0, 0]]
            cv2.ellipse(bg, c, ax, rng.randint(0, 180), 0, 360, col, -1)
    noise = np.random.default_rng(rng.randint(0, 10**6)).normal(0, 3, bg.shape)
    return np.clip(bg.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def physical_card(render: np.ndarray) -> np.ndarray:
    """Crops a synthetic render to the physical card. The templates carry a
    20-37px (1200x750 units) white border around the card itself; a real
    photo never does, so ground truth must be the card, not the border."""
    g = cv2.cvtColor(render, cv2.COLOR_BGR2GRAY) < 235
    cols = np.where(g.mean(0) > 0.5)[0]
    rows = np.where(g.mean(1) > 0.5)[0]
    return render[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1]


def make_scene(card: np.ndarray, cond: str, rng: random.Random):
    """Returns (photo, H) where H maps 1200x750 card coords -> photo coords."""
    canvas, scale_rng, ang_rng, persp, bg_kind, gray, invert = CONDITIONS[cond]
    card = cv2.resize(physical_card(card), (CARD_W, CARD_H), interpolation=cv2.INTER_AREA)
    if invert == "card":
        card = 255 - card
    if canvas is None:  # exact_tight: the photo IS the card
        photo, H = card.copy(), np.eye(3)
    else:
        cw, ch = canvas
        s = rng.uniform(*scale_rng) * cw / CARD_W
        ang = np.deg2rad(rng.uniform(*ang_rng))
        rot = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])
        pts = (CANON - [CARD_W / 2, CARD_H / 2]) * s
        pts = pts @ rot.T
        jitter = np.array([[rng.uniform(-1, 1), rng.uniform(-1, 1)] for _ in range(4)])
        pts = pts + jitter * persp * CARD_W * s
        span = pts.max(0) - pts.min(0)
        cx = rng.uniform(span[0] / 2 + 2, max(span[0] / 2 + 3, cw - span[0] / 2 - 2))
        cy = rng.uniform(span[1] / 2 + 2, max(span[1] / 2 + 3, ch - span[1] / 2 - 2))
        dst = (pts + [cx, cy]).astype(np.float32)
        H = cv2.getPerspectiveTransform(CANON, dst)
        bg = background(bg_kind, cw, ch, rng)
        warped = cv2.warpPerspective(card, H, (cw, ch))
        mask = cv2.warpPerspective(np.full((CARD_H, CARD_W), 255, np.uint8), H, (cw, ch))
        mask3 = (mask[..., None] / 255.0)
        photo = (warped * mask3 + bg * (1 - mask3)).astype(np.uint8)
    if gray:
        photo = cv2.cvtColor(cv2.cvtColor(photo, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
    if invert == "whole":
        photo = 255 - photo
    ok, buf = cv2.imencode(".jpg", photo, [cv2.IMWRITE_JPEG_QUALITY, rng.randint(70, 92)])
    return cv2.imdecode(buf, cv2.IMREAD_COLOR), H


def corner_error(M, H) -> float:
    """Mean distance of the aligned card's corners from where they belong."""
    if M is None:
        return float("inf")
    mapped = cv2.perspectiveTransform(CANON.reshape(-1, 1, 2), M @ H).reshape(-1, 2)
    return float(np.linalg.norm(mapped - CANON, axis=1).mean())


def align_fn():
    """Returns f(photo) -> (rectified, M_image_to_card). Prefers the new
    preprocessing stage when it exists, else the legacy align_card."""
    try:
        from egyptian_national_id_ocr.core.card_preprocessor import CardPreprocessor
        pre = CardPreprocessor()

        def f(photo):
            cands = pre.prepare(photo)
            return cands[0].image, cands[0].M
        return "card_preprocessor", f
    except ImportError:
        from egyptian_national_id_ocr.core.layout_analyzer import LayoutAnalyzer
        an = LayoutAnalyzer()
        return "legacy align_card", lambda photo: an.align_card(photo)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--count", type=int, default=12)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--e2e", action="store_true", help="also run OCR and check the national ID")
    args = ap.parse_args()

    name, align = align_fn()
    print(f"aligner: {name}   pass = mean corner error <= {PASS_PX:.0f}px (1200x750 space)\n")

    pipeline = None
    if args.e2e:
        from egyptian_national_id_ocr.core.pipeline import Pipeline
        pipeline = Pipeline()

    tmp = Path(tempfile.mkdtemp())
    random.seed(args.seed)
    cards = []
    for i in range(args.count):
        r = generate_sample(i, tmp)
        cards.append((cv2.imread(str(tmp / r["filename"])), cv2.imread(str(tmp / r["back_filename"])),
                      r["data"]["national_id"]))

    totals = []
    for cond in CONDITIONS:
        if args.only and cond not in args.only:
            continue
        rng = random.Random(f"{args.seed}-{cond}")
        errs, nid_ok, times = [], 0, []
        for i, (front, back, nid) in enumerate(cards):
            card = front if i % 2 == 0 else back
            photo, H = make_scene(card, cond, rng)
            t = time.time()
            _, M = align(photo)
            times.append(time.time() - t)
            errs.append(corner_error(M, H))
            if pipeline is not None:
                res = pipeline.process_image(photo)
                side = res.front if res.front else res.back
                nid_ok += int(bool(side) and side.national_id == nid)
        errs = np.array(errs)
        passed = int((errs <= PASS_PX).sum())
        totals.append((passed, len(errs)))
        finite = errs[np.isfinite(errs)]
        med = f"{np.median(finite):7.1f}" if len(finite) else "    n/a"
        line = (f"{cond:15s} pass {passed:2d}/{len(errs)}   median err {med}px   "
                f"no-card {int((~np.isfinite(errs)).sum())}   align {np.mean(times)*1000:5.0f}ms")
        if pipeline is not None:
            line += f"   NID correct {nid_ok}/{len(errs)}"
        print(line, flush=True)
    p = sum(a for a, _ in totals)
    n = sum(b for _, b in totals)
    print(f"\nOVERALL crop pass {p}/{n} ({100 * p / max(n, 1):.0f}%)")


if __name__ == "__main__":
    main()
