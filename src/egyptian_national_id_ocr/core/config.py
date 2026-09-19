import shutil
from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path
import os
import platform

def find_tesseract_path() -> str:
    """Attempts to find the tesseract executable on the system."""
    # 1. Check if it's already in the PATH
    path = shutil.which("tesseract")
    if path:
        return path
        
    # 2. Check common Windows paths
    if platform.system() == "Windows":
        common_paths = [
            r"C:\Program Files\Tesseract-OCR\tesseract.exe",
            r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
            os.path.join(os.environ.get("LOCALAPPDATA", ""), r"Tesseract-OCR\tesseract.exe")
        ]
        for p in common_paths:
            if os.path.exists(p):
                return p
                
    # 3. Default to "tesseract" and hope for the best
    return "tesseract"

class Settings(BaseSettings):
    # API Settings
    API_TITLE: str = "Egyptian National ID OCR"
    API_VERSION: str = "0.1.0"
    DEBUG: bool = False
    
    # OCR Settings
    TESSERACT_CMD: str = find_tesseract_path()
    DEFAULT_OCR_ENGINE: str = "tesseract"
    ENABLE_EASYOCR_FALLBACK: bool = True
    MIN_CONFIDENCE_THRESHOLD: float = 0.5
    
    # Image Settings
    TARGET_WIDTH: int = 1000
    TARGET_HEIGHT: int = 630  # Standard ID ratio approx 1.58:1
    
    # Directories
    BASE_DIR: Path = Path(__file__).parent.parent.parent.parent
    ASSETS_DIR: Path = BASE_DIR / "assets"
    
    model_config = SettingsConfigDict(env_prefix="NID_")

settings = Settings()
