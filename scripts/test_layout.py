import cv2
import os
import sys
import numpy as np
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from national_id_ocr.core.layout_analyzer import LayoutAnalyzer

def test_layout_on_dataset():
    dataset_dir = Path("assets/dataset")
    output_dir = Path("output/debug_layout")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    analyzer = LayoutAnalyzer()
    
    # Test on first 5 front images
    sample_files = [f for f in os.listdir(dataset_dir) if f.startswith("ID") and not f.startswith("IDB") and f.endswith(".png")][:5]
    
    for filename in sample_files:
        img_path = dataset_dir / filename
        image = cv2.imread(str(img_path))
        if image is None: continue
        
        print(f"Processing {filename}...")
        
        # 1. Align
        rectified, M = analyzer.align_card(image)
        
        # 2. Draw boxes on rectified image
        debug_img = rectified.copy()
        for field_name, [x, y, w, h] in analyzer.FRONT_FIELDS.items():
            cv2.rectangle(debug_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(debug_img, field_name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # Save result
        out_path = output_dir / f"boxed_{filename}"
        cv2.imwrite(str(out_path), debug_img)
        print(f"Saved to {out_path}")

if __name__ == "__main__":
    test_layout_on_dataset()
