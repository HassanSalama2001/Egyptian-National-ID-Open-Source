from fastapi import FastAPI, File, UploadFile, HTTPException
import cv2
import numpy as np
import io
from .core.pipeline import Pipeline
from .core.config import settings
from .models.id_card import IDCard

from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title=settings.API_TITLE, version=settings.API_VERSION)

# Configure CORS for standalone frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify the actual origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
pipeline = Pipeline(tesseract_cmd=settings.TESSERACT_CMD)

@app.post("/api/v1/extract", response_model=IDCard)
async def extract_nid(file: UploadFile = File(...)):
    """
    Extract data from an Egyptian National ID card image.
    Supports JPG, PNG, etc.
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
        
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if image is None:
        raise HTTPException(status_code=400, detail="Could not decode image")
        
    result = pipeline.process_image(image)
    return result

@app.get("/api/v1/health")
async def health_check():
    return {"status": "healthy", "version": settings.API_VERSION}
