import cv2
import os
import sys
import json
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from national_id_ocr.core.pipeline import Pipeline
from national_id_ocr.ocr.easyocr_engine import EasyOCREngine

def test_full_pipeline():
    dataset_dir = Path("assets/dataset")
    output_dir = Path("output/test_results")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    ocr_engine = EasyOCREngine()
    pipeline = Pipeline(ocr_engine=ocr_engine)
    
    # Test on first 5 front images
    sample_files = [f for f in os.listdir(dataset_dir) if f.startswith("ID") and not f.startswith("IDB") and f.endswith(".png")][:5]
    
    all_results = {}
    
    for filename in sample_files:
        img_path = dataset_dir / filename
        image = cv2.imread(str(img_path))
        if image is None: continue
        
        print(f"Processing {filename}...")
        
        result = pipeline.process_image(image)
        
        res_dict = {
            "side": result.side,
            "confidence": result.confidence,
            "processing_time_ms": result.processing_time_ms,
            "front": {
                "first_name": result.front.first_name,
                "full_name": result.front.full_name,
                "address": result.front.address,
                "national_id": result.front.national_id,
                "date_of_birth": result.front.date_of_birth,
                "card_serial_number": result.front.card_serial_number
            } if result.front else None
        }
        
        all_results[filename] = res_dict
        
    with open(output_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=4)
    
    print(f"Results saved to {output_dir / 'results.json'}")

if __name__ == "__main__":
    test_full_pipeline()
