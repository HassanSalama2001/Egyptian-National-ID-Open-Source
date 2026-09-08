import os
import time
import cv2
import numpy as np
import base64
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
from national_id_ocr.core.pipeline import Pipeline
from national_id_ocr.ocr.paddle_ocr_engine import PaddleOCREngine

app = FastAPI(title="Egyptian National ID OCR API")

# Configure CORS for React development server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify the actual origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize OCR Engine once - PaddleOCR (mobile det+rec), CPU-only,
# faster and more accurate than EasyOCR for this task (see
# ocr/paddle_ocr_engine.py)
engine = PaddleOCREngine()
pipeline = Pipeline(ocr_engine=engine)

@app.post("/ocr")
async def process_id_card(file: UploadFile = File(...)):
    start_time = time.time()
    
    # Read image
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if image is None:
        raise HTTPException(status_code=400, detail="Invalid image file")
        
    try:
        # Process through pipeline
        id_card = pipeline.process_image(image)
        
        # Add latency info
        id_card.processing_time_ms = int((time.time() - start_time) * 1000)
        
        # Convert original image to base64 to show in UI
        # We could also use the rectified image if side was detected
        _, buffer = cv2.imencode('.jpg', image)
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        
        return {
            "data": id_card.model_dump(),
            "image": f"data:image/jpeg;base64,{img_base64}"
        }
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy", "engine": "EasyOCR"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
