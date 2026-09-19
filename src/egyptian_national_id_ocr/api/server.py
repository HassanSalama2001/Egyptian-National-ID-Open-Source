import logging

from fastapi import FastAPI, File, UploadFile, HTTPException
import cv2
import numpy as np
import io
from PIL import Image, ImageOps
from .core.pipeline import Pipeline
from .core.config import settings
from .models.id_card import IDCard

logger = logging.getLogger(__name__)

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
    Supports JPG, PNG, etc. Expects a well-framed photo (see
    Pipeline.process_image's docstring) - designed for a guided-crop UI
    where the user aligns the card to an on-screen frame before
    submitting, not an arbitrary unconstrained photo.
    """
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

    # PIL + exif_transpose, not cv2.imdecode: cv2 ignores EXIF Orientation,
    # so a real phone photo (pixels stored sideways/mirrored plus a "rotate
    # for display" tag) would be processed in the wrong orientation even
    # though it displays upright everywhere else - see src/app.py for the
    # full explanation of this same fix.
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
        return pipeline.process_image(image)
    except Exception:
        # Pipeline.process_image already catches its own internal errors
        # and returns a result with an honest status/message instead of
        # raising - this is a last-resort net for anything truly
        # unexpected, so a caller never sees a raw stack trace.
        logger.exception("Unexpected error while processing an ID card image")
        raise HTTPException(
            status_code=500,
            detail="Something went wrong while processing this image. Please try again in a moment.",
        )

@app.get("/api/v1/health")
async def health_check():
    return {"status": "healthy", "version": settings.API_VERSION}
