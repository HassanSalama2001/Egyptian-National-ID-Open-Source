import cv2
import easyocr
import sys
from pathlib import Path

def test_easyocr(image_path):
    print(f"Testing EasyOCR on: {image_path}")
    image = cv2.imread(str(image_path))
    if image is None:
        print("Error: Could not load image")
        return

    # Convert BGR to RGB
    rgb = image[:, :, ::-1]
    
    print("Initializing Reader...")
    reader = easyocr.Reader(['ar', 'en'], gpu=False)
    
    print("Extracting text...")
    results = reader.readtext(rgb, detail=0)
    
    import sys
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    
    print("--- RESULTS ---")
    for res in results:
        # Convert to standard numbers for easier reading in logs if needed
        from egyptian_national_id_ocr.postprocessing.numeral_converter import normalize_arabic_numerals
        norm = normalize_arabic_numerals(res)
        print(f"Original: {res} | Normalized: {norm}")
    print("---------------")

if __name__ == "__main__":
    dataset_dir = Path("assets/dataset")
    images = list(dataset_dir.glob("*.png"))
    if not images:
        print("No images found in assets/dataset")
    else:
        test_easyocr(images[0])
