"""
Synthetic Egyptian National ID (front side) generator.

Renders realistic front-card samples onto the real template asset with
precisely calibrated field positions (measured directly from the template
via row-projection - see scripts/training/calibrate_template.py) and saves
a ground-truth manifest alongside each image. This is what makes the
generated data usable for both: a real accuracy benchmark (compare OCR
output against known-correct values, not just "is it 14 chars") and
training data for the digit/name recognizers.

Only front-side fields are generated: pipeline.py does not yet implement
back-side extraction (no `elif side == BACK` branch), so back-side ground
truth would currently be unused.
"""
import json
import os
import random
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

from national_id_ocr.postprocessing.national_id_parser import (
    GOVERNORATE_CODES,
    compute_nid_check_digit,
)

PROJECT_ROOT = Path(__file__).parent.parent.parent
TEMPLATE_PATH = PROJECT_ROOT / "assets" / "front_template.jpg"

WESTERN_TO_ARABIC_INDIC = str.maketrans("0123456789", "٠١٢٣٤٥٦٧٨٩")

# Field anchor points, calibrated directly against assets/front_template.jpg
# via row-projection (see calibrate_template.py). `anchor` is the (x, y)
# point the RIGHT edge of the value text is aligned to - Arabic is RTL, so
# text is drawn ending at this point and growing leftward, matching where
# real card values sit just to the left of their printed label.
FRONT_FIELDS = {
    "first_name":   {"anchor": (1900, 539), "font_size": 62, "digits": False},
    "full_name":    {"anchor": (1750, 662), "font_size": 62, "digits": False},
    "address":      {"anchor": (1700, 841), "font_size": 58, "digits": False},
    "national_id":  {"anchor": (1680, 1303), "font_size": 78, "digits": True},
    "date_of_birth":{"anchor": (340, 1267), "font_size": 44, "digits": True},
    "serial_number":{"anchor": (350, 1477), "font_size": 52, "digits": True},
}

FONT_PATH = "C:/Windows/Fonts/tahomabd.ttf"
FONT_PATH_FALLBACK = "C:/Windows/Fonts/tahoma.ttf"

NAMES = ["محمد", "أحمد", "محمود", "عمر", "إبراهيم", "علي", "محسن", "خالد", "ياسين", "عبد الله",
         "مصطفى", "حسن", "حسين", "كريم", "طارق", "وليد", "سامي", "رامي", "عادل", "فؤاد"]
SURNAMES = ["حمزة", "إسماعيل", "زكي", "كامل", "حسني", "سليم", "رزق", "منصور", "جاد", "بكر",
            "عبد الرحمن", "الشريف", "النجار", "حسن", "درويش", "فهمي", "عثمان", "شعبان"]
GOVERNORATE_NAMES = ["الجيزة", "القاهرة", "الإسكندرية", "الدقهلية", "المنيا", "أسيوط", "الشرقية",
                      "الغربية", "البحيرة", "المنوفية", "بني سويف", "الفيوم"]
STREET_WORDS = ["شارع", "ميدان", "حي"]


def _load_font(size: int) -> ImageFont.FreeTypeFont:
    path = FONT_PATH if os.path.exists(FONT_PATH) else FONT_PATH_FALLBACK
    return ImageFont.truetype(path, size)


def generate_id_data() -> dict:
    first = random.choice(NAMES)
    middle = random.choice(NAMES)
    last = random.choice(SURNAMES)

    century = random.choice(["2", "3"])
    year = f"{random.randint(50, 99):02}" if century == "2" else f"{random.randint(0, 24):02}"
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

    return {
        "first_name": first,
        "full_name": f"{middle} {last}",
        "address": f"{random.choice(STREET_WORDS)} {str(random.randint(1, 100)).translate(WESTERN_TO_ARABIC_INDIC)}، {random.choice(GOVERNORATE_NAMES)}",
        "national_id": nid,
        "date_of_birth": f"{day}/{month}/{full_year:04}",
        "serial_number": f"{random.randint(10000000, 99999999)}",
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


def generate_sample(index: int, output_dir: Path) -> dict:
    template = Image.open(TEMPLATE_PATH).convert("RGB")
    draw = ImageDraw.Draw(template)

    data = generate_id_data()
    boxes = {}
    for field_name, cfg in FRONT_FIELDS.items():
        font = _load_font(cfg["font_size"])
        boxes[field_name] = draw_value(
            draw, data[field_name], cfg["anchor"], font, cfg["digits"]
        )

    filename = f"sample_{index:04}.jpg"
    output_dir.mkdir(parents=True, exist_ok=True)
    template.save(output_dir / filename, quality=92)

    return {"filename": filename, "data": data, "boxes": boxes}


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
