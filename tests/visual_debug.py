import cv2
import os
import numpy as np
from egyptian_national_id_ocr.ocr.base import OCRPreprocessor
from egyptian_national_id_ocr.segmentation.front_segmenter import FrontSegmenter
from egyptian_national_id_ocr.detection.card_detector import CardDetector

def debug_preprocess(img_path):
    img = cv2.imread(img_path)
    if img is None:
        print("Failed to load image")
        return
        
    detector = CardDetector()
    rectified = detector.detect(img)
    cv2.imwrite("debug_rectified.png", rectified)
    
    segmenter = FrontSegmenter()
    rois = segmenter.segment(rectified)
    
    preprocessor = OCRPreprocessor()
    
    os.makedirs("debug_rois", exist_ok=True)
    
    for field, roi in rois.items():
        cv2.imwrite(f"debug_rois/{field}_raw.png", roi)
        processed = preprocessor.preprocess(roi, mode="neural")
        cv2.imwrite(f"debug_rois/{field}_processed.png", processed)
        print(f"Saved {field} (raw: {roi.shape}, processed: {processed.shape})")

if __name__ == "__main__":
    debug_preprocess("assets/dataset/ID0.png")
