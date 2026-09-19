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

    # Comma-separated list of origins the HTTP API accepts CORS requests
    # from. Defaults to the web_ui dev server (see web_ui/vite.config -
    # port 5173) rather than "*" - a wildcard origin combined with
    # allow_credentials=True (needed for cookie/auth-header based
    # deployments) lets ANY website's JavaScript call this API using the
    # visiting browser's own credentials, not just the intended frontend.
    # Override with NID_CORS_ALLOWED_ORIGINS for a deployed frontend's
    # real origin.
    CORS_ALLOWED_ORIGINS: str = "http://localhost:5173,http://127.0.0.1:5173"

    # Hard cap on one /ocr upload, enforced by reading in chunks and
    # aborting as soon as the running total crosses this (see app.py) -
    # not a check performed after buffering the whole body, which would
    # do nothing to bound memory for an attacker who just sends more
    # bytes than that. 15MB comfortably covers a real phone photo (a few
    # MB) with headroom, while still bounding a public endpoint's
    # worst-case memory use per request.
    MAX_UPLOAD_SIZE_MB: int = 15

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

    @property
    def cors_allowed_origins_list(self) -> list[str]:
        return [origin.strip() for origin in self.CORS_ALLOWED_ORIGINS.split(",") if origin.strip()]

settings = Settings()
