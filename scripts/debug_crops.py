import cv2
import os
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from national_id_ocr.core.layout_analyzer import LayoutAnalyzer

def debug_crops():
    dataset_dir = Path("assets/dataset")
    output_dir = Path("output/debug_crops")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    analyzer = LayoutAnalyzer()
    
    # Test on ID0.png
    filename = "ID0.png"
    img_path = dataset_dir / filename
    image = cv2.imread(str(img_path))
    if image is None: return
    
    rectified, M = analyzer.align_card(image)
    cv2.imwrite(str(output_dir / "rectified_ID0.png"), rectified)
    
    crops = analyzer.get_field_crops(rectified)
    for name, crop in crops.items():
        cv2.imwrite(str(output_dir / f"{name}_ID0.png"), crop)

if __name__ == "__main__":
    debug_crops()
