"""
Card preprocessing: turns an arbitrary photo or scan into an upright,
polarity-normalised, tightly cropped 1200x750 card image - once, before
any OCR runs.

Every field box downstream is a fixed coordinate in that 1200x750 frame,
so this stage decides whether the rest of the pipeline can work at all:
a crop that is 40px off moves every field box 40px off. Measured with
scripts/training/benchmark_card_crop.py (synthetic scenes with known card
placement) before this module existed, the previous alignment put only
16% of cards within 12px of the right place, and failed outright on
every sideways, upside-down, inverted and thin-margin photo - the same
failures seen on real third-party scans.

Stages, in order:
  1. Find the card's outline (several thresholdings, both polarities).
  2. Fit a straight line to each of its four edges and intersect them,
     giving true corners under perspective and despite rounded corners -
     a rotated rectangle (minAreaRect) cannot represent a card seen
     slightly from the side.
  3. Order the corners by the card's own LONG edge, so a card that is
     sideways in the photo is rotated, not squashed.
  4. Warp from the full-resolution original.
  5. Undo negative/inverted scans.
  6. Decide upright vs. upside down with the printed header/emblem
     templates - a cheap image comparison instead of a full OCR pass per
     rotation.
"""
import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np

from .layout_analyzer import CARD_ASPECT_RATIO, CARD_ASPECT_TOLERANCE, FRAME_ASPECT_TOLERANCE

logger = logging.getLogger(__name__)

CARD_W, CARD_H = 1200, 750
_CANON = np.array([[0, 0], [CARD_W - 1, 0], [CARD_W - 1, CARD_H - 1], [0, CARD_H - 1]],
                  dtype=np.float32)
_ROT180 = np.array([[-1, 0, CARD_W - 1], [0, -1, CARD_H - 1], [0, 0, 1]], dtype=np.float64)


@dataclass
class PreparedCard:
    """One candidate reading of the card, ready for field cropping."""
    image: np.ndarray                 # 1200x750 BGR
    M: Optional[np.ndarray]           # homography original photo -> this image
    aligned: bool                     # False = no card outline found, whole image used
    inverted: bool = False
    notes: List[str] = field(default_factory=list)


class CardPreprocessor:
    WORK_LONG_SIDE = 1400        # detection runs on a downscaled copy
    MIN_AREA_ZOOMED_OUT = 0.03   # card may be small on a full-page scan
    MIN_AREA_TIGHT = 0.55        # frame already card-shaped: only accept the card
    MIN_RECTANGULARITY = 0.85    # contour area / its min-area-rect area
    DARK_CARD_MEDIAN = 105       # below this the card itself is dark: inverted scan
    UPRIGHT_MARGIN = 0.05        # template-score lead needed to flip a card 180
    TEMPLATE_SCALES = (0.6, 0.8, 1.0, 1.25, 1.5)

    def __init__(self, classifier=None):
        if classifier is None:
            from ..detection.side_classifier import SideClassifier
            classifier = SideClassifier()
        self.classifier = classifier

    # ----------------------------------------------------------------- public
    def prepare(self, image: np.ndarray) -> List[PreparedCard]:
        """Returns candidates, most likely upright first (normally exactly
        two: the card and the same card rotated 180 degrees)."""
        h, w = image.shape[:2]
        scale = min(1.0, self.WORK_LONG_SIDE / max(h, w))
        work = cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA) \
            if scale < 1.0 else image

        notes: List[str] = []
        quad, aligned = None, True
        tight = self._is_card_shaped(w, h, FRAME_ASPECT_TOLERANCE)
        if tight:
            # A card-shaped photo is (nearly) cropped already; what is left
            # is at most a band of background. Trimming finds the OUTERMOST
            # edge on each side, which the outline detector cannot promise:
            # when card and background are the same brightness it snaps to
            # a stronger line inside the card (measured 12-15px in).
            trimmed, moved = self._trim_margins(cv2.cvtColor(work, cv2.COLOR_BGR2GRAY))
            if moved:
                quad = trimmed / scale
                notes.append("background margin trimmed")
        if quad is None:
            found = self._find_card_quad(work)
            if found is not None:
                quad = found / scale
                notes.append("card outline detected")
            elif tight:
                quad = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], np.float32)
                notes.append("photo is already cropped to the card")
            else:
                quad = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], np.float32)
                aligned = False
                notes.append("no card outline found - using the whole image")

        quad = self._order_long_edge_first(quad)
        M = cv2.getPerspectiveTransform(quad.astype(np.float32), _CANON)
        # INTER_LINEAR, as align_card uses: the digit classifier was trained
        # on crops warped that way, and cubic's sharper edges and ringing
        # are a different input distribution to it and to PaddleOCR.
        card = cv2.warpPerspective(image, M, (CARD_W, CARD_H), flags=cv2.INTER_LINEAR)

        inverted = False
        gray = cv2.cvtColor(card, cv2.COLOR_BGR2GRAY)
        if np.median(gray) < self.DARK_CARD_MEDIAN:
            card = 255 - card
            inverted = True
            notes.append("inverted (negative) scan corrected")

        flipped = cv2.rotate(card, cv2.ROTATE_180)
        M_flipped = _ROT180 @ M
        up_score, down_score = self._upright_scores(card)
        first = PreparedCard(card, M, aligned, inverted, list(notes))
        second = PreparedCard(flipped, M_flipped, aligned, inverted, notes + ["rotated 180"])
        # Most photos are upright, so only flip on clear evidence - a
        # near-tie (measured 0.46 vs 0.51 on a real upright card) is noise.
        if down_score > up_score + self.UPRIGHT_MARGIN:
            return [second, first]
        return [first, second]

    # ---------------------------------------------------------------- helpers
    @staticmethod
    def _is_card_shaped(w: float, h: float, tol: float) -> bool:
        if w <= 0 or h <= 0:
            return False
        return abs(max(w, h) / min(w, h) - CARD_ASPECT_RATIO) <= tol

    def _masks(self, gray: np.ndarray):
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        yield otsu            # light card on dark background
        yield 255 - otsu      # dark card on light background / paper
        yield cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY_INV, 35, 10)  # uneven lighting
        # Edges catch a card whose brightness matches its background. Scans
        # often give a BROKEN outline (one faint side) - that is fine, the
        # convex-hull checks below and edge snapping cope with it.
        for lo, hi in ((30, 100), (50, 150)):
            edges = cv2.Canny(blur, lo, hi)
            yield cv2.dilate(edges, np.ones((5, 5), np.uint8), iterations=2)

    def _find_card_quad(self, img: np.ndarray) -> Optional[np.ndarray]:
        h, w = img.shape[:2]
        area = float(h * w)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        tight = self._is_card_shaped(w, h, FRAME_ASPECT_TOLERANCE)
        min_area = self.MIN_AREA_TIGHT if tight else self.MIN_AREA_ZOOMED_OUT
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))

        # Every plausible card-shaped region from every thresholding. Areas
        # and rectangularity use the CONVEX HULL: a scan's card outline is
        # often broken on one side, which makes the traced contour itself
        # cover only 40-60% of the card (measured on real scans) while its
        # hull is still the full rectangle. A random blob's hull is not.
        cands = []
        for mask in self._masks(gray):
            closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
            contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in sorted(contours, key=cv2.contourArea, reverse=True)[:4]:
                hull = cv2.convexHull(c)
                ha = cv2.contourArea(hull)
                if ha < min_area * area:
                    continue
                rect = cv2.minAreaRect(hull)
                (rw, rh) = rect[1]
                if min(rw, rh) <= 0 or not self._is_card_shaped(rw, rh, CARD_ASPECT_TOLERANCE):
                    continue
                if ha / (rw * rh) < self.MIN_RECTANGULARITY:
                    continue
                x, y, bw, bh = cv2.boundingRect(hull)
                if sum([x <= 1, y <= 1, x + bw >= w - 2, y + bh >= h - 2]) >= 3:
                    continue  # the image frame itself, not a card inside it
                aspect_err = abs(max(rw, rh) / min(rw, rh) - CARD_ASPECT_RATIO)
                cands.append((ha, aspect_err, rect, hull))
        if not cands:
            return None

        # Candidates overlap heavily (the same card seen by several
        # thresholdings); keep one per distinct region.
        cands.sort(key=lambda t: -t[0])
        distinct = []
        for c in cands:
            (cx, cy), (rw, rh), _ = c[2]
            if any(abs(cx - d[2][0][0]) < 0.03 * max(w, h) and abs(cy - d[2][0][1]) < 0.03 * max(w, h)
                   and abs(c[0] / d[0] - 1) < 0.08 for d in distinct):
                continue
            distinct.append(c)

        # Each region gives two rough shapes, each better in different
        # photos: the outline's own corners follow perspective (a
        # trapezoid); the rotated rectangle survives an outline bloated by
        # a background blob touching the card. Both are snapped to the
        # image's real edges, anything that is no longer card-shaped is
        # dropped, and the rest compete on edge support - the true card
        # boundary is a sharp edge all the way round, a merged outline has
        # at least one side running through plain background.
        scored = []
        for ha, aspect_err, rect, hull in distinct[:6]:
            for rough in (self._hull_quad(hull, rect), cv2.boxPoints(rect).astype(np.float64)):
                q = self._snap_edges(gray, rough)
                ok, q_aspect_err = self._plausible_quad(q, w, h)
                if ok:
                    scored.append((self._edge_support(gray, q), q_aspect_err,
                                   cv2.contourArea(q.astype(np.float32)), q))
        if not scored:
            return None
        top = max(t[0] for t in scored)
        # Near-ties go to the most card-shaped quad: a card lying on a sheet
        # of paper - the sheet's outline is just as sharp as the card's.
        close = [t for t in scored if t[0] >= top - 0.08]
        return self._rebuild_weak_side(gray, min(close, key=lambda t: (round(t[1] / 0.03), -t[2]))[3])

    WEAK_SIDE = 0.55    # a side with less edge support than this is a guess
    STRONG_SIDE = 0.8   # its opposite side must be at least this well supported
    WEAK_SIDE_ASPECT_WEIGHT = 4.0  # edge-support points per unit of aspect-ratio error

    @classmethod
    def _rebuild_weak_side(cls, gray: np.ndarray, quad: np.ndarray) -> np.ndarray:
        """A light card on white paper, or a faint "eco" scan, can have one
        edge that is simply not visible; snapping then settles on whatever
        line is nearest, often slanted, and that corner is a guess. When
        the OPPOSITE side is clearly visible, the card's fixed aspect ratio
        says where the invisible side is: rebuild it parallel to its
        opposite, along the best-supported adjacent side (a card is a
        parallelogram to within the mild perspective of a scan or photo),
        re-snap, and keep whichever version the image supports best.
        Measured on a real eco back scan: the top edge had 18% edge
        support and its left corner sat ~39 card-pixels too high."""
        q = cls._order_clockwise(np.asarray(quad, np.float64))
        fracs = cls._side_support(gray, q)
        H, W = gray.shape[:2]
        options = [q]
        for i in range(4):
            j = (i + 2) % 4
            left_adj, right_adj = (i + 3) % 4, (i + 1) % 4
            if fracs[i] >= cls.WEAK_SIDE or fracs[j] < cls.STRONG_SIDE:
                continue
            guide = left_adj if fracs[left_adj] >= fracs[right_adj] else right_adj
            if fracs[guide] < cls.STRONG_SIDE:
                continue  # no reliable direction to rebuild along
            # Side i runs q[i] -> q[i+1]; opposite side j runs q[i+2] -> q[i+3].
            opp_len = np.linalg.norm(q[(j + 1) % 4] - q[j])
            adj_len = np.linalg.norm(q[guide] - q[(guide + 1) % 4])
            sep = opp_len / CARD_ASPECT_RATIO if opp_len >= adj_len else opp_len * CARD_ASPECT_RATIO
            # Unit vector of the guide side, pointing from side j towards side i.
            if guide == right_adj:      # runs q[i+1] -> q[i+2]
                u = q[(i + 1) % 4] - q[(i + 2) % 4]
            else:                       # runs q[i+3] -> q[i]
                u = q[i] - q[(i + 3) % 4]
            u = u / (np.linalg.norm(u) + 1e-9)
            new = q.copy()
            new[(i + 1) % 4] = q[(i + 2) % 4] + u * sep
            new[i] = q[(i + 3) % 4] + u * sep
            options.append(new)
            options.append(cls._order_clockwise(cls._snap_edges(gray, new).astype(np.float64)))
        # With one edge invisible, edge support alone cannot decide: a
        # re-snap that latches onto a printed line inside the card scores
        # as well as the true (unseen) edge. The card's exact aspect ratio
        # can - on the real eco scan the true outline was 0.003 off it and
        # the inner-line one 0.044 - so here, and only here, it counts.
        def score(o):
            ok, aspect_err = cls._plausible_quad(o, W, H)
            return cls._edge_support(gray, o) - cls.WEAK_SIDE_ASPECT_WEIGHT * aspect_err
        options = [o for o in options if cls._plausible_quad(o, W, H)[0]] or [q]
        return max(options, key=score).astype(np.float32)

    @staticmethod
    def _side_support(gray: np.ndarray, quad: np.ndarray) -> List[float]:
        """For each side, the fraction of points along it sitting on a
        strong edge perpendicular to it."""
        blur = cv2.GaussianBlur(gray, (3, 3), 0).astype(np.float32)
        gx = cv2.Sobel(blur, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(blur, cv2.CV_32F, 0, 1, ksize=3)
        H, W = gray.shape[:2]
        q = np.asarray(quad, np.float64)
        fracs = []
        ts = np.linspace(0.12, 0.88, 40)
        for i in range(4):
            a, b = q[i], q[(i + 1) % 4]
            d = b - a
            L = np.linalg.norm(d)
            if L < 1:
                return [0.0] * 4
            n = np.array([-d[1], d[0]]) / L
            hit = 0
            for t in ts:
                p = a + t * d
                best_r = 0.0
                for o in (-2, -1, 0, 1, 2):
                    x, y = int(round(p[0] + o * n[0])), int(round(p[1] + o * n[1]))
                    if 0 <= x < W and 0 <= y < H:
                        best_r = max(best_r, abs(gx[y, x] * n[0] + gy[y, x] * n[1]))
                hit += best_r > 20
            fracs.append(hit / len(ts))
        return fracs

    @staticmethod
    def _plausible_quad(q: np.ndarray, w: int, h: int) -> Tuple[bool, float]:
        """A photographed card: convex, corners near 90 degrees (moderate
        perspective only), card aspect ratio, inside the photo."""
        q = np.asarray(q, np.float64)
        if not cv2.isContourConvex(q.astype(np.float32).reshape(-1, 1, 2)):
            return False, 9.0
        m = 0.02 * max(w, h)
        if (q[:, 0] < -m).any() or (q[:, 0] > w - 1 + m).any() or                 (q[:, 1] < -m).any() or (q[:, 1] > h - 1 + m).any():
            return False, 9.0
        for i in range(4):
            a, b, c = q[i - 1], q[i], q[(i + 1) % 4]
            u, v = a - b, c - b
            cosang = np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-9)
            if abs(cosang) > 0.34:   # outside ~70-110 degrees
                return False, 9.0
        s = [np.linalg.norm(q[(i + 1) % 4] - q[i]) for i in range(4)]
        e1, e2 = (s[0] + s[2]) / 2, (s[1] + s[3]) / 2
        if min(e1, e2) <= 0:
            return False, 9.0
        err = abs(max(e1, e2) / min(e1, e2) - CARD_ASPECT_RATIO)
        return err <= CARD_ASPECT_TOLERANCE, err

    @classmethod
    def _edge_support(cls, gray: np.ndarray, quad: np.ndarray) -> float:
        """Mean per-side edge support, with the weakest side counted
        twice - a quad with one side running through plain background
        loses to one supported all the way round."""
        fracs = cls._side_support(gray, quad)
        return (sum(fracs) + min(fracs)) / 5.0

    @staticmethod
    def _hull_quad(hull: np.ndarray, rect) -> np.ndarray:
        """Four corners that follow the outline itself. A rotated rectangle
        cannot represent a card seen slightly from the side (a trapezoid) -
        on angled photos that alone put the corners 30-50px off in
        1200x750 terms, too far for edge snapping to recover."""
        peri = cv2.arcLength(hull, True)
        for eps in np.arange(0.01, 0.1, 0.005):
            approx = cv2.approxPolyDP(hull, eps * peri, True)
            if len(approx) == 4:
                return approx.reshape(-1, 2).astype(np.float64)
            if len(approx) < 4:
                break
        return cv2.boxPoints(rect).astype(np.float64)

    TRIM_MAX_FRAC = 0.15   # deepest background band trimmed, of the short side
    TRIM_MIN_GRAD = 25     # weakest border transition accepted (Sobel units)
    TRIM_MAX_STRIP_STD = 14  # a cut-off strip busier than this is card, not background

    @classmethod
    def _trim_margins(cls, gray: np.ndarray) -> Tuple[np.ndarray, int]:
        """For a photo that is card-shaped but may still carry a band of
        background: find each card side as the outermost strong transition,
        fitted as a (possibly tilted) straight line, and intersect the four
        lines. A side is only moved when the strip it cuts off is plain
        background (uniform, untextured) - a card that genuinely fills the
        photo has text and graphics right up to its border, so its strip
        fails that test and that side stays on the frame edge.

        Tilt matters: a card rotated just 2 degrees in a 1300px photo has an
        edge that drifts ~30px from one end to the other, so each side is a
        robustly fitted line, not a single inset.

        Returns the quad and how many sides were moved off the frame edge."""
        H, W = gray.shape[:2]
        blur = cv2.GaussianBlur(gray, (5, 5), 0).astype(np.float32)
        gx = np.abs(cv2.Sobel(blur, cv2.CV_32F, 1, 0, ksize=3))
        gy = np.abs(cv2.Sobel(blur, cv2.CV_32F, 0, 1, ksize=3))
        depth_max = max(4, int(cls.TRIM_MAX_FRAC * min(H, W)))
        n_samples = 48

        def side_line(grad, blur_img, along_len):
            """grad/blur_img are oriented so row 0 is the frame edge and the
            inward direction runs down axis 0; returns (a, b) with the edge
            at depth a + b * t for position t along the side, or None."""
            ts = np.linspace(0.08 * along_len, 0.92 * along_len, n_samples).astype(int)
            pts = []
            for t in ts:
                # The OUTERMOST real transition, by absolute strength: a
                # threshold relative to the band's peak fails whenever the
                # card's photo or text also falls inside the band - those
                # are 5-10x stronger than a light card's faint outer edge.
                prof = grad[:depth_max, t]
                above = np.nonzero(prof >= cls.TRIM_MIN_GRAD)[0]
                if not len(above):
                    continue
                k = int(above[0])
                k += int(np.argmax(prof[k:k + 5]))  # settle on that edge's peak
                pts.append((t, k))
            if len(pts) < 0.5 * n_samples:
                return None
            pts = np.array(pts, np.float64)
            best, best_in = None, None
            rng = np.random.default_rng(0)
            for _ in range(120):
                i, j = rng.choice(len(pts), 2, replace=False)
                if pts[i, 0] == pts[j, 0]:
                    continue
                b = (pts[j, 1] - pts[i, 1]) / (pts[j, 0] - pts[i, 0])
                a = pts[i, 1] - b * pts[i, 0]
                inl = np.abs(pts[:, 1] - (a + b * pts[:, 0])) <= 2.5
                if best_in is None or inl.sum() > best_in.sum():
                    best, best_in = (a, b), inl
            if best is None or best_in.sum() < 0.5 * n_samples:
                return None  # no straight, continuous edge along this side
            b, a = np.polyfit(pts[best_in, 0], pts[best_in, 1], 1)
            depths = a + b * pts[best_in, 0]
            # Under 1% of the short side is the card's own rim, not a margin
            # (a genuinely tight crop otherwise loses ~5px on two sides).
            if np.median(depths) < max(2, 0.01 * min(H, W)) or depths.max() > depth_max - 1                     or depths.min() < 0:
                return None
            strip = [blur_img[:max(1, int(d) - 1), int(t)] for t, d in zip(pts[best_in, 0], depths)]
            strip = np.concatenate(strip)
            if strip.size < 20 or strip.std() > cls.TRIM_MAX_STRIP_STD:
                return None  # the strip holds texture/content: not background
            return a, b

        # Each side expressed as a line in image coords: points (x, y).
        views = {
            "top": (gy, blur),
            "bottom": (gy[::-1], blur[::-1]),
            "left": (gx.T, blur.T),
            "right": (gx.T[::-1], blur.T[::-1]),
        }
        lines = {}
        found = {}
        for side, (g, bimg) in views.items():
            along = W if side in ("top", "bottom") else H
            fit = side_line(g, bimg, along)
            found[side] = fit is not None
            a, b = fit if fit is not None else (0.0, 0.0)
            p0, p1 = np.array([0.0, a]), np.array([float(along - 1), a + b * (along - 1)])
            # (t, depth) -> image (x, y)
            if side == "top":
                conv = lambda q: (q[0], q[1])
            elif side == "bottom":
                conv = lambda q: (q[0], H - 1 - q[1])
            elif side == "left":
                conv = lambda q: (q[1], q[0])
            else:
                conv = lambda q: (W - 1 - q[1], q[0])
            lines[side] = (np.array(conv(p0)), np.array(conv(p1)))

        def cross(l1, l2):
            (p, r), (q, s) = (l1[0], l1[1] - l1[0]), (l2[0], l2[1] - l2[0])
            den = r[0] * s[1] - r[1] * s[0]
            t = ((q - p)[0] * s[1] - (q - p)[1] * s[0]) / den
            return p + t * r

        def corners():
            return np.array([cross(lines["top"], lines["left"]), cross(lines["top"], lines["right"]),
                             cross(lines["bottom"], lines["right"]), cross(lines["bottom"], lines["left"])])

        # One side of a pair can be invisible (card and background the same
        # brightness along it) while the other three are clean. Left on the
        # frame edge, that side stretches the card by the whole margin; the
        # card's fixed aspect ratio says where it must really be.
        quad = corners()
        long_x = W >= H
        for a, b, axis in (("left", "right", 0), ("top", "bottom", 1)):
            if found[a] == found[b]:
                continue
            q = quad
            width = (np.linalg.norm(q[1] - q[0]) + np.linalg.norm(q[2] - q[3])) / 2
            height = (np.linalg.norm(q[3] - q[0]) + np.linalg.norm(q[2] - q[1])) / 2
            span, other = (width, height) if axis == 0 else (height, width)
            is_long = long_x == (axis == 0)
            expected = other * CARD_ASPECT_RATIO if is_long else other / CARD_ASPECT_RATIO
            if abs(span - expected) <= 0.02 * expected:
                continue  # the unmoved side really is on the frame edge
            keep, fix = (b, a) if found[b] else (a, b)
            shift = np.zeros(2)
            shift[axis] = expected if fix in ("right", "bottom") else -expected
            moved_line = (lines[keep][0] + shift, lines[keep][1] + shift)
            limit = (W if axis == 0 else H) - 1
            coords = np.array([moved_line[0][axis], moved_line[1][axis]])
            if (coords < -0.01 * limit).any() or (coords > 1.01 * limit).any():
                # The inference would put the card's edge OUTSIDE the photo.
                # That only follows if the two perpendicular sides really
                # are the card's edges, which an unmoved frame edge does not
                # prove - measured on a real back scan, extrapolating here
                # put the crop ~50px off where three other scans of the same
                # card agreed. Leave the side on the frame edge instead.
                continue
            lines[fix] = moved_line
            found[fix] = True
            quad = corners()
        return quad.astype(np.float32), int(sum(found.values()))

    @classmethod
    def _snap_edges(cls, gray: np.ndarray, box: np.ndarray, reach: Optional[int] = None) -> np.ndarray:
        """Moves each side of a rough box onto the strongest brightness
        transition nearby in the actual image, then intersects the fitted
        lines. Corner accuracy then comes from the photo itself rather than
        from whatever thresholding produced the rough outline (dilated
        edge masks, for instance, sit a few pixels outside the true edge -
        several card-pixels once scaled up to 1200x750)."""
        box = cls._order_clockwise(box)
        blur = cv2.GaussianBlur(gray, (3, 3), 0).astype(np.float32)
        gx = cv2.Sobel(blur, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(blur, cv2.CV_32F, 0, 1, ksize=3)
        H, W = gray.shape[:2]
        centre = box.mean(axis=0)
        short = min(np.linalg.norm(box[1] - box[0]), np.linalg.norm(box[2] - box[1]))
        # Reach is kept short on purpose: the back's barcode edge lies ~5%
        # of the card height inside the card edge and is a far stronger
        # transition than a light card against a light background.
        if reach is None:
            reach = max(4, int(0.03 * short))
        lines = []
        for i in range(4):
            a, b = box[i], box[(i + 1) % 4]
            d = b - a
            length = np.linalg.norm(d)
            if length < 20:
                return box.astype(np.float32)
            u = d / length
            n = np.array([-u[1], u[0]])
            if np.dot((a + b) / 2 - centre, n) < 0:
                n = -n  # n points OUT of the card
            offsets = np.arange(reach, -reach - 1, -1)  # outside -> inside
            pts = []
            for t in np.linspace(0.12, 0.88, 48):
                p = a + t * d
                samples = p[None, :] + offsets[:, None] * n[None, :]
                xs = np.clip(samples[:, 0].round().astype(int), 0, W - 1)
                ys = np.clip(samples[:, 1].round().astype(int), 0, H - 1)
                resp = np.abs(gx[ys, xs] * n[0] + gy[ys, xs] * n[1])
                k = int(np.argmax(resp))
                if resp[k] > 12:
                    # Sub-pixel peak (parabola through the 3 samples): on a
                    # small card one photo pixel is 3+ card pixels.
                    off = 0.0
                    if 0 < k < len(resp) - 1:
                        den = resp[k - 1] - 2 * resp[k] + resp[k + 1]
                        if den < 0:
                            off = 0.5 * (resp[k - 1] - resp[k + 1]) / den
                    pts.append(p + (offsets[k] - off) * n)
            if len(pts) < 12:
                lines.append((a, u))
                continue
            lines.append(cls._ransac_line(np.array(pts, np.float64)))
        corners = []
        for i in range(4):
            p1, d1 = lines[i - 1]
            p2, d2 = lines[i]
            A = np.array([d1, -d2]).T
            if abs(np.linalg.det(A)) < 1e-6:
                return box.astype(np.float32)
            s_, _ = np.linalg.solve(A, p2 - p1)
            corners.append(p1 + s_ * d1)
        corners = np.array(corners)
        diag = np.linalg.norm(box[0] - box[2])
        if np.linalg.norm(corners - box, axis=1).max() > 0.08 * diag:
            return box.astype(np.float32)
        return corners.astype(np.float32)

    @staticmethod
    def _ransac_line(pts: np.ndarray, band: float = 1.5):
        """Line through the largest population of collinear points. Near a
        card edge the per-sample strongest transition is usually the rim,
        but some samples catch printing just inside it (the header text
        runs close to the top edge); a least-squares or Huber fit averages
        the two groups into a TILTED line, putting one corner 10-40 card
        pixels off. RANSAC keeps the dominant group - the rim - only."""
        best_in = None
        rng = np.random.default_rng(0)
        for _ in range(60):
            i, j = rng.choice(len(pts), 2, replace=False)
            d = pts[j] - pts[i]
            L = np.linalg.norm(d)
            if L < 1e-6:
                continue
            n = np.array([-d[1], d[0]]) / L
            inl = np.abs((pts - pts[i]) @ n) <= band
            if best_in is None or inl.sum() > best_in.sum():
                best_in = inl
        use = pts[best_in] if best_in is not None and best_in.sum() >= 6 else pts
        vx, vy, x0, y0 = cv2.fitLine(use.astype(np.float32), cv2.DIST_L2, 0, 0.01, 0.01).ravel()
        return np.array([x0, y0]), np.array([vx, vy])

    @staticmethod
    def _order_clockwise(pts: np.ndarray) -> np.ndarray:
        c = pts.mean(axis=0)
        ang = np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0])
        pts = pts[np.argsort(ang)]            # clockwise in image coordinates
        start = int(np.argmin(pts.sum(axis=1)))  # nearest the top-left
        return np.roll(pts, -start, axis=0)

    @classmethod
    def _order_long_edge_first(cls, quad: np.ndarray) -> np.ndarray:
        q = cls._order_clockwise(np.asarray(quad, np.float64))
        top = np.linalg.norm(q[1] - q[0]) + np.linalg.norm(q[2] - q[3])
        side = np.linalg.norm(q[2] - q[1]) + np.linalg.norm(q[3] - q[0])
        if side > top:  # card is sideways in the photo: rotate, don't squash
            q = np.roll(q, -1, axis=0)
        return q.astype(np.float32)

    def _upright_scores(self, card: np.ndarray) -> Tuple[float, float]:
        """Best header/emblem template match, upright vs. rotated 180. Tried
        at several sizes: the templates were cut from a stock card image
        whose scale does not match a real card warped to 1200x750, which
        made a single-scale match unreliable on real photos."""
        gray = cv2.cvtColor(card, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (CARD_W // 2, CARD_H // 2), interpolation=cv2.INTER_AREA)
        flipped = cv2.rotate(gray, cv2.ROTATE_180)
        score = self.classifier._get_match_score
        up = down = 0.0
        for tpl in (self.classifier.front_template, self.classifier.back_template):
            if tpl is None:
                continue
            for sc in self.TEMPLATE_SCALES:
                t = cv2.resize(tpl, None, fx=sc / 2, fy=sc / 2, interpolation=cv2.INTER_AREA)
                if t.shape[0] >= gray.shape[0] or t.shape[1] >= gray.shape[1] or min(t.shape) < 8:
                    continue
                up = max(up, score(gray, t))
                down = max(down, score(flipped, t))
        return up, down
