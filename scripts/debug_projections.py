import cv2
import numpy as np
import os
from pathlib import Path

def test_projections():
    dataset_dir = Path("assets/dataset")
    output_dir = Path("output/debug_projections")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process ID0.png and ID10.png
    for filename in ["ID0.png", "ID10.png"]:
        img_path = dataset_dir / filename
        image = cv2.imread(str(img_path))
        if image is None: continue
        
        # Warp to 1200x750 (simulate LayoutAnalyzer)
        # For simplicity in this script, just resize
        warped = cv2.resize(image, (1200, 750))
        
        gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        
        # Binary threshold (inverted)
        _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
        
        # 1. Right Region Projection (for names/address)
        right_region = thresh[100:550, 500:1150]
        h_proj = np.sum(right_region, axis=1)
        
        # Find peaks in projection
        peaks = []
        in_peak = False
        start = 0
        for i, val in enumerate(h_proj):
            if val > (np.max(h_proj) * 0.1) and not in_peak:
                in_peak = True
                start = i
            elif val < (np.max(h_proj) * 0.1) and in_peak:
                in_peak = False
                peaks.append((start + 100, i + 100)) # Add offset
        
        debug_img = warped.copy()
        for i, (y1, y2) in enumerate(peaks):
            cv2.rectangle(debug_img, (500, y1), (1150, y2), (0, 255, 0), 2)
            cv2.putText(debug_img, f"Peak {i}", (1160, y1 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            
        cv2.imwrite(str(output_dir / f"proj_{filename}"), debug_img)
        print(f"Detected {len(peaks)} peaks in {filename}")

if __name__ == "__main__":
    test_projections()
