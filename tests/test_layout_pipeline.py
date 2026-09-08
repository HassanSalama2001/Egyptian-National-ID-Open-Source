import cv2
import json
import numpy as np
from national_id_ocr.core.pipeline import Pipeline
from national_id_ocr.ocr.easyocr_engine import EasyOCREngine

def test_pipeline(img_path):
    print(f"Testing pipeline on: {img_path}")
    img = cv2.imread(img_path)
    if img is None:
        print("Error: Could not load image")
        return

    # Initialize hardened engine
    engine = EasyOCREngine(gpu=False)
    pipeline = Pipeline(ocr_engine=engine)
    
    # Process
    result = pipeline.process_image(img)
    
    # Save a JSON file for the user first (safer than printing on Windows)
    with open("debug_pipeline_result.json", "w", encoding="utf-8") as f:
        json.dump(result.model_dump(), f, indent=2, ensure_ascii=False, default=str)
    
    # Display Results (Might still fail on some consoles but we have the file)
    try:
        print(json.dumps(result.model_dump(), indent=2, ensure_ascii=False, default=str))
    except UnicodeEncodeError:
        print("Success! (Console cannot display Arabic, see debug_pipeline_result.json)")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--img", default="assets/dataset/ID0.png")
    args = parser.parse_args()
    
    test_pipeline(args.img)
