import io
import logging
import os
import time
import cv2
import numpy as np
import base64
from PIL import Image, ImageOps
from fastapi import FastAPI, File, Query, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
from egyptian_national_id_ocr.core.pipeline import Pipeline
from egyptian_national_id_ocr.ocr.paddle_ocr_engine import PaddleOCREngine

logger = logging.getLogger(__name__)

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
async def process_id_card(
    file: UploadFile = File(...),
    include_images: bool = Query(
        True,
        description=(
            "Whether to include base64 image payloads (the aligned card "
            "image, the per-field crops, and a copy of the uploaded photo). "
            "These power the 'show your work' UI but dominate the response "
            "size - set false for a data-only response that is orders of "
            "magnitude smaller."
        ),
    ),
):
    start_time = time.time()

    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="That doesn't look like an image file. Please upload a JPG or PNG photo of the ID card.",
        )

    contents = await file.read()
    if not contents:
        raise HTTPException(
            status_code=400,
            detail="The uploaded file is empty. Please choose a photo and try again.",
        )

    # Decode via PIL (not cv2.imdecode) specifically to apply exif_transpose:
    # phone photos commonly store pixels in one orientation plus an EXIF
    # Orientation tag saying how to rotate/mirror for display. cv2.imdecode
    # ignores that tag entirely and loads the raw pixels, so a real phone
    # photo could be processed sideways or mirrored even though it displays
    # upright everywhere else - align_card's rotation retry only tries pure
    # rotations, not mirrors, so a mirrored EXIF orientation would never
    # self-correct. exif_transpose() normalizes both cases before any of
    # our own processing sees the image.
    try:
        pil_image = Image.open(io.BytesIO(contents))
        pil_image = ImageOps.exif_transpose(pil_image)
        image = cv2.cvtColor(np.array(pil_image.convert("RGB")), cv2.COLOR_RGB2BGR)
    except Exception:
        image = None

    if image is None:
        raise HTTPException(
            status_code=400,
            detail="We couldn't read this file as an image. Please make sure it's a valid JPG or PNG and try again.",
        )

    try:
        # Process through pipeline
        id_card = pipeline.process_image(image)
        id_card.processing_time_ms = int((time.time() - start_time) * 1000)

        if not include_images:
            # Data-only response: no aligned card image, no field crops,
            # and no echo of the uploaded photo (that echo is the single
            # largest item in the payload).
            return {"data": id_card.without_images().model_dump(), "image": None}

        # Convert original image to base64 to show in UI
        _, buffer = cv2.imencode('.jpg', image)
        img_base64 = base64.b64encode(buffer).decode('utf-8')

        return {
            "data": id_card.model_dump(),
            "image": f"data:image/jpeg;base64,{img_base64}"
        }
    except Exception:
        # Pipeline.process_image already catches its own internal errors
        # and returns a result with an honest status/message - this is a
        # last-resort net for anything truly unexpected. Never leak the
        # raw exception message to the client.
        logger.exception("Unexpected error while processing an ID card image")
        raise HTTPException(
            status_code=500,
            detail="Something went wrong while processing this image. Please try again in a moment.",
        )

@app.get("/health")
async def health_check():
    return {"status": "healthy", "engine": "PaddleOCR"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
