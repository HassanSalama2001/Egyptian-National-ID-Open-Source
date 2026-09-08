import os
import time
import random
import cv2
import numpy as np
from pathlib import Path
from national_id_ocr.core.pipeline import Pipeline

def run_benchmark(dataset_dir: str, num_samples: int = 50):
    from national_id_ocr.ocr.easyocr_engine import EasyOCREngine
    
    # Initialize Pipeline with EasyOCR for host-compatibility
    engine = EasyOCREngine(gpu=False)
    pipeline = Pipeline(ocr_engine=engine)
    
    dataset_path = Path(dataset_dir)
    images = list(dataset_path.glob("*.p*")) + list(dataset_path.glob("*.j*")) # png, jpg, jpeg
    
    if len(images) > num_samples:
        test_samples = random.sample(images, num_samples)
    else:
        test_samples = images
        
    results = {
        "total": 0,
        "detection_success": 0,
        "side_correct": 0,
        "nid_extracted": 0,
        "total_time": 0.0,
        "failures": []
    }
    
    print(f"STARTING Benchmark on {len(test_samples)} samples...")
    print("-" * 50)
    
    for img_path in test_samples:
        results["total"] += 1
        start_time = time.time()
        
        try:
            # Load Image
            image = cv2.imread(str(img_path))
            if image is None:
                raise ValueError("Could not load image")
                
            # Process image (Internal pipeline steps for debugging)
            rectified_card = pipeline.detector.detect(image)
            if rectified_card is not None:
                cv2.imwrite("debug_rectified.jpg", rectified_card)
                side_cls = pipeline.classifier.classify(rectified_card)
                if side_cls.value == "front":
                    rois = pipeline.front_segmenter.segment(rectified_card)
                    if "national_id" in rois:
                        cv2.imwrite("debug_roi_nid.jpg", rois["national_id"])
                elif side_cls.value == "back":
                    rois = pipeline.back_segmenter.segment(rectified_card)
                    if "national_id" in rois:
                        cv2.imwrite("debug_roi_nid.jpg", rois["national_id"])
            
            # Process image
            card_id = pipeline.process_image(image)
            elapsed = time.time() - start_time
            
            # Debug: Save what Tesseract found in full page if fallback was triggered
            # (We need to reach into the pipeline or just run it here)
            debug_text = pipeline.ocr_engine.extract_text(image, lang="ara", psm=3)
            with open("debug_full_ocr.txt", "w", encoding="utf-8") as f:
                f.write(debug_text)
            results["total_time"] += elapsed
            
            # Logic Analysis
            # Detection is successful if we at least classified a side (requires rectified image)
            is_detected = card_id.side != "unknown"
            
            nid = None
            if card_id.front and card_id.front.national_id:
                nid = card_id.front.national_id
            elif card_id.back and card_id.back.national_id:
                nid = card_id.back.national_id
                
            has_nid = nid is not None and len(nid) == 14
            
            # Print extracted fields for visibility
            if card_id.front:
                print(f"   Fields: NID={card_id.front.national_id}, Name={card_id.front.first_name}")
            elif card_id.back:
                print(f"   Fields: NID={card_id.back.national_id}")
            
            # 2. Side Inference (Inferred from filename for this specific dataset)
            # Front: ID*.png, Back: IDB*.png
            is_back_truth = "IDB" in img_path.name or "back" in img_path.name.lower()
            is_side_correct = (card_id.side == "back") == is_back_truth
            
            if is_detected: results["detection_success"] += 1
            if is_side_correct: results["side_correct"] += 1
            if has_nid: results["nid_extracted"] += 1
            
            status = "OK" if has_nid else "WARN"
            print(f"[{results['total']:02d}] {status} {img_path.name} ({elapsed:.2f}s) - Side: {card_id.side}")
            
        except Exception as e:
            results["failures"].append({"path": str(img_path), "error": str(e)})
            print(f"[{results['total']:02d}] FAIL {img_path.name} - Failed: {str(e)[:50]}")
            
    # Calculate averages
    avg_time = results["total_time"] / results["total"] if results["total"] > 0 else 0
    
    print("-" * 50)
    print(f"RESULTS SUMMARY:")
    print(f"   - Total Samples: {results['total']}")
    print(f"   - Detection Success: {results['detection_success']} ({results['detection_success']/results['total']*100:.1f}%)")
    print(f"   - Side Classification Accuracy: {results['side_correct']} ({results['side_correct']/results['total']*100:.1f}%)")
    print(f"   - NID Extraction Rate: {results['nid_extracted']} ({results['nid_extracted']/results['total']*100:.1f}%)")
    print(f"   - Avg Latency: {avg_time:.3f}s")
    
if __name__ == "__main__":
    dataset_dir = "assets/dataset"
    if os.path.exists(dataset_dir):
        run_benchmark(dataset_dir, num_samples=30)
    else:
        print(f"Error: Dataset directory {dataset_dir} not found.")
