import cv2
import os
from pathlib import Path

def generate_templates():
    project_root = Path(__file__).parent.parent
    assets_dir = project_root / "assets"
    templates_dir = project_root / "src" / "egyptian_national_id_ocr" / "detection" / "templates"
    
    os.makedirs(templates_dir, exist_ok=True)
    
    # Use Egyptian_ID_Card.jpg as the source for high-quality templates
    template_src = cv2.imread(str(assets_dir / "Egyptian_ID_Card.jpg"))
    if template_src is None:
        print("Error: Could not find template source.")
        return

    # 1. Front Side Header: "بطاقة تحقيق الشخصية"
    # Approximate region in the template image (top center)
    front_header = template_src[40:120, 480:800]
    cv2.imwrite(str(templates_dir / "front_header.jpg"), front_header)
    
    # 2. Back Side Eagle: Eagle emblem
    # Approximate region in the template image (top right of back side)
    # The back side starts around y=500 in the Egyptian_ID_Card.jpg
    back_eagle = template_src[550:700, 820:960]
    cv2.imwrite(str(templates_dir / "back_eagle.jpg"), back_eagle)
    
    print(f"Templates generated in {templates_dir}")

if __name__ == "__main__":
    generate_templates()
