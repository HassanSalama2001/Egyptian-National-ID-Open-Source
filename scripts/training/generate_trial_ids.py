import os
import random
import arabic_reshaper
from bidi.algorithm import get_display
from PIL import Image, ImageDraw, ImageFont
import numpy as np

# Final Refined Coordinates (1200x1500 template)
# X=920 is the right-side anchor (near labels)
FRONT_LAYOUT = {
    "first_name": (920, 165),
    "full_name": (920, 230),
    "address": (920, 300),
    "national_id": (920, 440),
    "serial_number": (140, 450),
}

# Back Card (Y starts after ~750)
BACK_LAYOUT = {
    "national_id": (850, 815),
    "occupation": (850, 860),
    "gender": (930, 955),
    "religion": (750, 955),
    "social_status": (550, 955),
}

# Sample Egyptian Data
NAMES = ["محمد", "أحمد", "محمود", "عمر", "إبراهيم", "علي", "محسن", "خالد", "ياسين", "عبد الله"]
SURNAMES = ["حمزة", "إسماعيل", "زكي", "كامل", "حسني", "سليم", "رزق", "منصور", "جاد", "بكر"]
GOVERNORATES = ["الجيزة", "القاهرة", "الإسكندرية", "الدقهلية", "المنيا", "أسيوط"]
OCCUPATIONS = ["مهندس", "طبيب", "محاسب", "طالب", "موظف", "مدرس"]

def generate_id_data():
    first = random.choice(NAMES)
    middle = random.choice(NAMES)
    last = random.choice(SURNAMES)
    
    # Construct NID: Century(1) YY(2) MM(2) DD(2) Gov(2) Sequence(3) Gender(1) Check(1)
    century = random.choice(["2", "3"])
    year = f"{random.randint(50, 99):02}" if century == "2" else f"{random.randint(0, 24):02}"
    month = f"{random.randint(1, 12):02}"
    day = f"{random.randint(1, 28):02}"
    gov = f"{random.randint(1, 27):02}"
    seq = f"{random.randint(1, 999):03}"
    gender = random.choice(["1", "2"]) # 1=Male, 2=Female
    # Basic checksum placeholder
    nid = f"{century}{year}{month}{day}{gov}{seq}{gender}3" 
    
    return {
        "first_name": first,
        "full_name": f"{middle} {last}",
        "address": f"شارع {random.randint(1, 100)}، {random.choice(GOVERNORATES)}",
        "national_id": nid,
        "serial_number": f"KR{random.randint(1000000, 9999999)}",
        "occupation": random.choice(OCCUPATIONS),
        "gender": "ذكر" if gender == "1" else "أنثى",
        "religion": "مسلم",
        "social_status": "أعزب" if gender == "1" else "آنسة",
    }

def draw_arabic_text(draw, text, position, font, color=(20, 20, 20)):
    # Reshape and handle Bidi (RTL)
    reshaped_text = arabic_reshaper.reshape(text)
    bidi_text = get_display(reshaped_text)
    
    # Draw (Right-aligned logic: position provided is the anchor point for the END of the text)
    bbox = draw.textbbox((0, 0), bidi_text, font=font)
    w = bbox[2] - bbox[0]
    draw.text((position[0] - w, position[1]), bidi_text, font=font, fill=color)

def generate_sample(index, output_dir):
    template = Image.open("assets/Egyptian_ID_Card.jpg")
    draw = ImageDraw.Draw(template)
    
    # Fonts
    try:
        font_name = ImageFont.truetype("C:/Windows/Fonts/arialbd.ttf", 36)
        font_nid = ImageFont.truetype("C:/Windows/Fonts/arialbd.ttf", 48)
        font_serial = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 28)
    except:
        # Fallback to default if arial not found
        font_name = ImageFont.load_default()
        font_nid = ImageFont.load_default()
        font_serial = ImageFont.load_default()

    data = generate_id_data()
    
    # Draw Front
    draw_arabic_text(draw, data["first_name"], FRONT_LAYOUT["first_name"], font_name)
    draw_arabic_text(draw, data["full_name"], FRONT_LAYOUT["full_name"], font_name)
    draw_arabic_text(draw, data["address"], FRONT_LAYOUT["address"], font_name)
    draw_arabic_text(draw, data["national_id"], FRONT_LAYOUT["national_id"], font_nid)
    draw.text(FRONT_LAYOUT["serial_number"], data["serial_number"], font=font_serial, fill=(40, 40, 40))

    # Draw Back
    draw_arabic_text(draw, data["national_id"], BACK_LAYOUT["national_id"], font_nid)
    draw_arabic_text(draw, data["occupation"], BACK_LAYOUT["occupation"], font_name)
    draw_arabic_text(draw, data["gender"], BACK_LAYOUT["gender"], font_name)
    draw_arabic_text(draw, data["religion"], BACK_LAYOUT["religion"], font_name)
    draw_arabic_text(draw, data["social_status"], BACK_LAYOUT["social_status"], font_name)

    # Save
    os.makedirs(output_dir, exist_ok=True)
    filename = f"sample_id_{index:02}.jpg"
    template.save(os.path.join(output_dir, filename))
    return filename

if __name__ == "__main__":
    output_dir = "scripts/training/samples"
    print(f"Generating 10 samples in {output_dir}...")
    for i in range(1, 11):
        name = generate_sample(i, output_dir)
        print(f"Generated {name}")
