import cv2
import sys
import os
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from national_id_ocr.detection.card_detector import CardDetector
from national_id_ocr.core.exceptions import NationalIDOCRError

def test_detection():
    detector = CardDetector()
    assets = ["front.jpg", "back.jpg"]
    
    for asset in assets:
        img_path = Path(__file__).parent.parent / "assets" / asset
        print(f"Testing detection on {asset}...")
        
        image = cv2.imread(str(img_path))
        if image is None:
            print(f"Error: Could not load {asset}")
            continue
            
        try:
            detected = detector.detect(image)
            output_path = f"detected_{asset}"
            cv2.imwrite(output_path, detected)
            print(f"Success! Saved to {output_path}")
        except NationalIDOCRError as e:
            print(f"Failed to detect {asset}: {e}")

if __name__ == "__main__":
    test_detection()
