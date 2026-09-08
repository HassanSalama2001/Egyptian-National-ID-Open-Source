import cv2
import numpy as np
import os
from pathlib import Path

def visualize_text_blobs():
    dataset_dir = Path("assets/dataset")
    output_dir = Path("output/debug_blobs")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process first few front images
    sample_files = [f for f in os.listdir(dataset_dir) if f.startswith("ID") and not f.startswith("IDB") and f.endswith(".png")][:3]
    
    for filename in sample_files:
        img_path = dataset_dir / filename
        image = cv2.imread(str(img_path))
        if image is None: continue
        
        # 1. Grayscale & Blur
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (7, 7), 0)
        
        # 2. Adaptive Thresholding to get text
        thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 10)
        
        # 3. Dilate horizontally to merge characters/words
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 5))
        dilated = cv2.dilate(thresh, kernel, iterations=2)
        
        # 4. Find contours of blobs
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        debug_img = image.copy()
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            # Filter by size to ignore noise/borders
            if w > 50 and h > 20 and w < 1000 and h < 200:
                cv2.rectangle(debug_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        cv2.imwrite(str(output_dir / f"blobs_{filename}"), debug_img)
        print(f"Saved blobs for {filename}")

if __name__ == "__main__":
    visualize_text_blobs()
