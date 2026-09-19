"""
Synthetic Egyptian National ID (front + back) generator.

Renders realistic front- and back-card samples onto the real template
assets with precisely calibrated field positions (measured directly from
the template via row-projection - see scripts/training/calibrate_template.py)
and saves a ground-truth manifest alongside each image pair. This is what
makes the generated data usable for both: a real accuracy benchmark
(compare OCR output against known-correct values, not just "is it 14
chars") and training data for the digit/name recognizers.

Back-side field boxes are tighter than the front's - the back's layout
has competing labels sharing a row (e.g. issue_date and the repeated
national_id), leaving much less room per value than the front's large NID
display gets, so back fields use smaller fonts to match what the real
template's spacing actually allows.

Gender-linked fields (gender word, marital-status word, the NID's own
gender digit) are generated consistently from one underlying choice, not
independently, so the synthetic data doesn't contradict itself.
"""
import json
import os
import random
import string
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

try:
    import arabic_reshaper
    from bidi.algorithm import get_display
except ImportError as e:
    raise SystemExit(
        "Missing training dependencies. Install with: "
        "pip install -e '.[training]'"
    ) from e

from egyptian_national_id_ocr.postprocessing.national_id_parser import (
    GOVERNORATE_CODES,
    compute_nid_check_digit,
)

PROJECT_ROOT = Path(__file__).parent.parent.parent
TEMPLATE_PATH = PROJECT_ROOT / "assets" / "front_template.jpg"
BACK_TEMPLATE_PATH = PROJECT_ROOT / "assets" / "back_template.jpg"

WESTERN_TO_ARABIC_INDIC = str.maketrans("0123456789", "٠١٢٣٤٥٦٧٨٩")

# Field anchor points, calibrated directly against assets/front_template.jpg
# via row-projection (see calibrate_template.py). `anchor` is the (x, y)
# point the RIGHT edge of the value text is aligned to - Arabic is RTL, so
# text is drawn ending at this point and growing leftward, matching where
# real card values sit just to the left of their printed label.
# IMPORTANT: these anchors are re-derived from layout_analyzer.py's
# FRONT_FIELDS (which itself was re-measured from a real card photo via
# scripts/debug_field_grid.py, not the blank template - see that file's
# comment for why). Keeping these two in sync is what makes synthetic
# training/benchmark data representative of what the pipeline actually
# sees on a real card, rather than just self-consistent with itself.
FRONT_FIELDS = {
    "first_name":   {"anchor": (2432, 537), "font_size": 62, "digits": False},
    "full_name":    {"anchor": (2432, 661), "font_size": 62, "digits": False},
    # Real cards wrap the address across 2 lines (building/street, then
    # district - governorate) - a single-line render didn't match, so this
    # field draws multiline (see draw_multiline_value / _render_side).
    # anchor is line 1's position; line 2 follows line_spacing below it.
    "address":      {"anchor": (2432, 827), "font_size": 52, "digits": False, "multiline": True, "line_spacing": 124},
    "national_id":  {"anchor": (2453, 1281), "font_size": 78, "digits": True},
    "date_of_birth":{"anchor": (565, 1240), "font_size": 44, "digits": True},
    # Real card serials are 2 Latin letters + digits (e.g. "AB1234567"),
    # not Arabic-Indic digits - digits=False skips the Arabic-Indic
    # translation draw_value would otherwise apply.
    "serial_number":{"anchor": (747, 1473), "font_size": 52, "digits": False},
}

def _back_fields_from_analyzer() -> dict:
    """Derives the back-side draw anchors directly from the pipeline's own
    crop boxes, instead of hard-coding a second set of coordinates.

    These WERE hard-coded, and drifted: layout_analyzer.BACK_FIELDS was
    re-calibrated against a real card photo, and this table was not
    updated to match. The generator then rendered every back value
    300-600px away from where the pipeline crops - e.g. issue_date drawn
    at x=720 while the pipeline reads x=1141. The resulting synthetic
    backs were unreadable by design, which made the back-side benchmark
    measure this mismatch rather than the pipeline: back.issue_date
    scored 0% even on a clean render, while REAL back cards read
    perfectly.

    Deriving them means the two can no longer disagree. The anchor is the
    box's right edge (draw_value right-aligns, matching Arabic RTL text),
    vertically centred; the font is sized from the box height so the
    rendered text fills the crop the way real printing does, rather than
    sitting tiny inside an over-large box.
    """
    from PIL import Image as _Image

    from egyptian_national_id_ocr.core.layout_analyzer import LayoutAnalyzer

    width, height = _Image.open(BACK_TEMPLATE_PATH).size
    scale_x, scale_y = width / 1200, height / 750

    # Digit fields are drawn at a smaller fraction of box height than text:
    # a date's box is sized for the value plus breathing room, and digits
    # rendered at full box height overflow it.
    height_fraction = {"digits": 0.46, "text": 0.55}
    digit_fields = {"issue_date", "national_id", "expiry_date"}

    fields = {}
    for name, (x, y, box_w, box_h) in LayoutAnalyzer().BACK_FIELDS.items():
        is_digits = name in digit_fields
        fraction = height_fraction["digits" if is_digits else "text"]
        fields[name] = {
            "anchor": (int((x + box_w) * scale_x), int((y + box_h / 2) * scale_y)),
            "font_size": max(18, int(box_h * scale_y * fraction)),
            "digits": is_digits,
        }
    return fields


BACK_FIELDS = _back_fields_from_analyzer()

# Bundled (assets/fonts/, Amiri - OFL licensed, see assets/fonts/OFL.txt),
# not an OS font path. This used to hardcode "C:/Windows/Fonts/tahoma*.ttf" -
# which not only doesn't exist on Linux/Mac (this generator is now part of
# the automated test suite via tests/conftest.py's synthetic_card_sample
# fixture, and CI's first real run on a Linux runner failed outright on
# this), it also meant "seeded generation is reproducible" was only true
# ON WINDOWS: two contributors on different OSes, or the same seed run
# locally vs. in CI, were never guaranteed to render identical pixels,
# since nothing pinned WHICH font file Tahoma even resolved to. Bundling
# the font makes both problems the same fix: one specific file, present
# and identical everywhere this repository is cloned.
FONT_DIR = PROJECT_ROOT / "assets" / "fonts"
FONT_PATH = str(FONT_DIR / "Amiri-Bold.ttf")
FONT_PATH_FALLBACK = str(FONT_DIR / "Amiri-Regular.ttf")
# A per-field-type font split (Simplified Arabic Bold for digits only)
# was tried and reverted - see generate_digit_crops.py's FONT_PATH for
# the full story (a closer glyph-shape match to a real card, but with its
# own '٢'/'٧' and '٥'/'٠' confusions that made real-card accuracy net
# worse). Back to a single font for everything.
DIGIT_FONT_PATH = FONT_PATH
DIGIT_FONT_PATH_FALLBACK = FONT_PATH_FALLBACK

NAMES = ["محمد", "أحمد", "محمود", "عمر", "إبراهيم", "علي", "محسن", "خالد", "ياسين", "عبد الله",
         "مصطفى", "حسن", "حسين", "كريم", "طارق", "وليد", "سامي", "رامي", "عادل", "فؤاد"]
SURNAMES = ["حمزة", "إسماعيل", "زكي", "كامل", "حسني", "سليم", "رزق", "منصور", "جاد", "بكر",
            "عبد الرحمن", "الشريف", "النجار", "حسن", "درويش", "فهمي", "عثمان", "شعبان"]
GOVERNORATE_NAMES = ["الجيزة", "القاهرة", "الإسكندرية", "الدقهلية", "المنيا", "أسيوط", "الشرقية",
                      "الغربية", "البحيرة", "المنوفية", "بني سويف", "الفيوم"]
STREET_WORDS = ["شارع", "ميدان", "حي"]
# Address line 1 (building/street) parts - real cards show a number, then
# an area name, then a building-type word (e.g. "٤٢٩ البنفسج عمارات").
AREA_NAMES = ["البنفسج", "الياسمين", "النرجس", "الفردوس", "الأندلس", "الزهراء", "الشروق"]
BUILDING_TYPES = ["عمارات", "فيلات", "أبراج"]
# Address line 2 (district - governorate).
DISTRICTS = ["التجمع الاول", "التجمع الثالث", "مدينة نصر", "المعادي", "الزمالك",
             "المهندسين", "حدائق القبة", "عين شمس"]
# Real cards can show either a short job word or a full degree/qualification
# title (e.g. "بكالوريوس فى علوم الحاسب") - both appear in practice, and the
# longer phrases are what actually stress-test the crop box width.
OCCUPATIONS = ["مهندس", "طبيب", "محاسب", "مدرس", "موظف", "طالب", "محامي", "صيدلي", "تاجر", "سائق",
               "بكالوريوس علوم", "بكالوريوس تجارة", "بكالوريوس هندسة", "بكالوريوس فى علوم الحاسب",
               "دبلوم فني", "ربة منزل"]

# Real option sets for the back's closed-vocabulary fields, matching
# postprocessing/enum_matcher.py's tables. Gendered forms picked to agree
# with the same underlying gender choice as the NID's own gender digit -
# see generate_id_data().
RELIGIONS_MALE = ["مسلم", "مسيحي"]
RELIGIONS_FEMALE = ["مسلمة", "مسيحية"]
MARITAL_MALE = ["أعزب", "متزوج", "مطلق", "أرمل"]
MARITAL_FEMALE = ["عزباء", "متزوجة", "مطلقة", "أرملة"]


def _load_font(size: int, digits: bool = False) -> ImageFont.FreeTypeFont:
    if digits:
        path = DIGIT_FONT_PATH if os.path.exists(DIGIT_FONT_PATH) else DIGIT_FONT_PATH_FALLBACK
    else:
        path = FONT_PATH if os.path.exists(FONT_PATH) else FONT_PATH_FALLBACK
    return ImageFont.truetype(path, size)


def generate_id_data() -> dict:
    first = random.choice(NAMES)
    middle = random.choice(NAMES)
    last = random.choice(SURNAMES)

    century = random.choice(["2", "3"])
    # Egyptian IDs are issued at 16, so a 2000s-century cardholder can only
    # have been born up to 16 years ago - this used to run to the current
    # year, generating "cardholders" aged 2, which no real card describes.
    # That produced benchmark samples the pipeline is right to reject (see
    # national_id_parser.MIN_PLAUSIBLE_AGE_YEARS) and would have quietly
    # understated its accuracy.
    from datetime import date as _date
    newest_2000s_year = (_date.today().year - 16) % 100
    year = (
        f"{random.randint(50, 99):02}"
        if century == "2"
        else f"{random.randint(0, newest_2000s_year):02}"
    )
    month = f"{random.randint(1, 12):02}"
    day = f"{random.randint(1, 28):02}"
    gov_code = random.choice(list(GOVERNORATE_CODES.keys()))
    seq = f"{random.randint(1, 999):03}"
    gender_digit = random.choice(["1", "3", "5", "7", "9"]) if random.random() < 0.5 \
        else random.choice(["0", "2", "4", "6", "8"])

    first13 = f"{century}{year}{month}{day}{gov_code}{seq}{gender_digit}"
    check_digit = compute_nid_check_digit(first13)
    nid = f"{first13}{check_digit}"

    full_year = (1900 if century == "2" else 2000) + int(year)

    # Odd = male, even = female (real NID convention) - keep gendered word
    # forms consistent with this rather than picking them independently.
    is_male = int(gender_digit) % 2 != 0

    issue_year = random.randint(2018, 2024)
    issue_month = random.randint(1, 12)
    issue_day = random.randint(1, 28)
    expiry_year = issue_year + 7  # standard adult-card validity period

    address_line1 = (
        f"{str(random.randint(1, 999)).translate(WESTERN_TO_ARABIC_INDIC)} "
        f"{random.choice(AREA_NAMES)} {random.choice(BUILDING_TYPES)}"
    )
    address_line2 = f"{random.choice(DISTRICTS)} - {random.choice(GOVERNORATE_NAMES)}"

    return {
        "first_name": first,
        "full_name": f"{middle} {last}",
        # Ground truth for the whole (2-line) address, space-joined - this
        # matches what OCR actually produces: PaddleOCR detects each line
        # separately and pipeline.py's _process_crop joins multi-line
        # results with a space (see its read_layout sort+join). The split
        # lines themselves (for rendering) live in _address_lines below.
        "address": f"{address_line1} {address_line2}",
        "_address_lines": [address_line1, address_line2],
        "national_id": nid,
        # Real cards print this as YYYY/MM/DD (confirmed against a real
        # card), not DD/MM/YYYY.
        "date_of_birth": f"{full_year:04}/{month}/{day}",
        # Real card serials are 2 Latin letters + 7 digits (e.g.
        # "AB1234567"), not Arabic-Indic digits.
        "serial_number": f"{''.join(random.choices(string.ascii_uppercase, k=2))}{random.randint(1000000, 9999999)}",
        # Back-side fields
        "profession": random.choice(OCCUPATIONS),
        "gender": "ذكر" if is_male else "أنثى",
        "religion": random.choice(RELIGIONS_MALE if is_male else RELIGIONS_FEMALE),
        "marital_status": random.choice(MARITAL_MALE if is_male else MARITAL_FEMALE),
        # Real cards print issue_date as YYYY/MM (no day) and expiry_date
        # as the full YYYY/MM/DD - not the DD/MM/YYYY originally assumed.
        "issue_date": f"{issue_year}/{issue_month:02}",
        "expiry_date": f"{expiry_year}/{issue_month:02}/{issue_day:02}",
    }


def draw_value(draw: ImageDraw.ImageDraw, text: str, anchor: tuple, font: ImageFont.FreeTypeFont,
               is_digits: bool, color=(15, 15, 15)) -> tuple:
    """
    Draws `text` right-aligned to `anchor`. Digit fields are rendered using
    Arabic-Indic glyphs (٠-٩), matching real Egyptian ID cards - training a
    recognizer on Western digits would teach it the wrong glyph shapes
    entirely. Returns the drawn bounding box (x0, y0, x1, y1) for the
    ground-truth manifest.
    """
    if is_digits:
        display_text = text.translate(WESTERN_TO_ARABIC_INDIC)
        # Digit strings still flow left-to-right visually on the card even
        # though individual glyphs are Arabic-Indic; no bidi reshaping needed.
    else:
        reshaped = arabic_reshaper.reshape(text)
        display_text = get_display(reshaped)

    bbox = draw.textbbox((0, 0), display_text, font=font)
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x0 = anchor[0] - w
    y0 = anchor[1] - h // 2 - bbox[1]
    draw.text((x0, y0), display_text, font=font, fill=color)
    return (x0, y0 + bbox[1], x0 + w, y0 + bbox[1] + h)


def draw_multiline_value(draw: ImageDraw.ImageDraw, lines: list, anchor: tuple, font: ImageFont.FreeTypeFont,
                          is_digits: bool, line_spacing: int, color=(15, 15, 15)) -> tuple:
    """Draws each line right-aligned to the same anchor[0], starting at
    anchor[1] and stepping down by line_spacing per line (matches how real
    cards right-align a multi-line field's lines to a shared edge). Returns
    the bounding box spanning all lines, for the ground-truth manifest."""
    box = None
    for i, text in enumerate(lines):
        line_anchor = (anchor[0], anchor[1] + i * line_spacing)
        x0, y0, x1, y1 = draw_value(draw, text, line_anchor, font, is_digits, color)
        box = (x0, y0, x1, y1) if box is None else (
            min(box[0], x0), min(box[1], y0), max(box[2], x1), max(box[3], y1)
        )
    return box


def _render_side(template_path: Path, fields: dict, data: dict) -> tuple:
    template = Image.open(template_path).convert("RGB")
    draw = ImageDraw.Draw(template)
    boxes = {}
    for field_name, cfg in fields.items():
        font = _load_font(cfg["font_size"], digits=cfg["digits"])
        if cfg.get("multiline"):
            boxes[field_name] = draw_multiline_value(
                draw, data[f"_{field_name}_lines"], cfg["anchor"], font, cfg["digits"], cfg["line_spacing"]
            )
        else:
            boxes[field_name] = draw_value(
                draw, data[field_name], cfg["anchor"], font, cfg["digits"]
            )
    return template, boxes


def generate_sample(index: int, output_dir: Path, data: dict = None) -> dict:
    """Renders both sides of one synthetic card from a single shared `data`
    dict (generated if not passed in), so front and back ground truth -
    including the repeated national_id and the gender-linked fields - stay
    consistent with each other rather than being generated independently."""
    if data is None:
        data = generate_id_data()

    output_dir.mkdir(parents=True, exist_ok=True)

    front_img, front_boxes = _render_side(TEMPLATE_PATH, FRONT_FIELDS, data)
    filename = f"sample_{index:04}.jpg"
    front_img.save(output_dir / filename, quality=92)

    back_img, back_boxes = _render_side(BACK_TEMPLATE_PATH, BACK_FIELDS, data)
    back_filename = f"sample_{index:04}_back.jpg"
    back_img.save(output_dir / back_filename, quality=92)

    return {
        "filename": filename,
        "back_filename": back_filename,
        "data": data,
        "boxes": front_boxes,
        "back_boxes": back_boxes,
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=500)
    parser.add_argument("--output", default=str(PROJECT_ROOT / "data" / "synthetic_front"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    output_dir = Path(args.output)
    manifest = []

    from tqdm import tqdm
    for i in tqdm(range(1, args.count + 1), desc="Generating samples"):
        manifest.append(generate_sample(i, output_dir))

    with open(output_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f"Generated {args.count} labeled samples in {output_dir}")
    print(f"Manifest: {output_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
