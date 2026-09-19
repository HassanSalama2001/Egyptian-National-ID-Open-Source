import cv2
import easyocr
import os
import sys

# Add src to path
sys.path.append("src")

from egyptian_national_id_ocr.postprocessing.national_id_parser import find_nid_in_text
from egyptian_national_id_ocr.ocr.easyocr_engine import EasyOCREngine
from egyptian_national_id_ocr.ocr.base import OCRPreprocessor

def debug_image(image_path):
    if not os.path.exists(image_path):
        print(f"File not found: {image_path}")
        return

    print(f"Debugging: {image_path}")
    img = cv2.imread(image_path)
    
    # Run through preprocessor
    preprocessor = OCRPreprocessor()
    pre_img = preprocessor.preprocess(img)
    cv2.imwrite("scratch/pre_debug.png", pre_img)
    print("Saved preprocessed image to scratch/pre_debug.png")

    # Run EasyOCR
    reader = easyocr.Reader(['ar', 'en'], gpu=False)
    results = reader.readtext(pre_img, detail=0)
    full_text = " ".join(results)
    
    print("\n--- RAW OCR TEXT ---")
    # Handle Arabic printing
    import sys
    sys.stdout.reconfigure(encoding='utf-8')
    print(full_text)
    print("--------------------")

    nid = find_nid_in_text(full_text)
    print(f"\nExtracted NID: {nid}")

if __name__ == "__main__":
    target = "assets/dataset/ID186.png"
    debug_image(target)
