import os
import cv2
import time
import logging
import argparse
from tqdm import tqdm
from national_id_ocr.core.pipeline import Pipeline
from national_id_ocr.ocr.easyocr_engine import EasyOCREngine

# Disable logging for cleaner output
logging.getLogger('national_id_ocr').setLevel(logging.ERROR)

def run_benchmark(dataset_path, sample_size=50):
    print(f"Starting Accuracy Overhaul Benchmark...")
    print(f"Dataset: {dataset_path}")
    print(f"Sampling: {sample_size} images")
    print("-" * 50)

    # Initialize Pipeline with EasyOCR
    engine = EasyOCREngine(gpu=False)
    pipeline = Pipeline(ocr_engine=engine)

    # Get image list
    all_images = [f for f in os.listdir(dataset_path) if f.endswith(('.png', '.jpg', '.jpeg'))]
    if sample_size > 0:
        import random
        random.seed(42)
        images = random.sample(all_images, min(sample_size, len(all_images)))
    else:
        images = all_images

    stats = {
        "total": len(images),
        "nid_extracted": 0,
        "side_detected": 0,
        "total_time_ms": 0,
        "errors": 0
    }

    results = []

    for img_name in tqdm(images, desc="Processing"):
        img_path = os.path.join(dataset_path, img_name)
        image = cv2.imread(img_path)
        
        if image is None:
            stats["errors"] += 1
            continue

        try:
            start_time = time.time()
            result = pipeline.process_image(image)
            latency = (time.time() - start_time) * 1000
            
            stats["total_time_ms"] += latency
            
            nid = None
            if result.front and result.front.national_id:
                nid = result.front.national_id
            elif result.back and result.back.national_id:
                nid = result.back.national_id
                
            success = nid is not None and len(nid) == 14
            if success:
                stats["nid_extracted"] += 1
            
            if result.side != "unknown":
                stats["side_detected"] += 1
                
            results.append({
                "name": img_name,
                "success": success,
                "nid": nid,
                "side": result.side,
                "latency": latency
            })
            
        except Exception as e:
            stats["errors"] += 1

    # Print Report
    print("-" * 50)
    print("Benchmark Complete!")
    print(f"National ID Extraction Rate: {(stats['nid_extracted']/stats['total'])*100:.2f}%")
    print(f"Side Detection Rate: {(stats['side_detected']/stats['total'])*100:.2f}%")
    print(f"Avg Latency: {stats['total_time_ms']/stats['total']:.2f} ms")
    print(f"Errors: {stats['errors']}")
    print("-" * 50)

    # Print some successful examples
    print("\nSample Extractions:")
    successes = [r for r in results if r['success']][:5]
    for s in successes:
        print(f"  - {s['name']}: {s['nid']} ({s['side']})")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="assets/dataset")
    parser.add_argument("--sample", type=int, default=20)
    args = parser.parse_args()
    
    run_benchmark(args.dataset, args.sample)
