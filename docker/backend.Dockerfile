# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Install system dependencies for OpenCV and Tesseract
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    tesseract-ocr \
    tesseract-ocr-ara \
    tesseract-ocr-eng \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# Copy the requirements file into the container
COPY pyproject.toml .
RUN pip install --no-cache-dir .

# Copy the rest of the application code
COPY src /app/src
COPY assets /app/assets

# Expose the port the app runs on
EXPOSE 8000

# Run the application
CMD ["uvicorn", "egyptian_national_id_ocr.api.server:app", "--host", "0.0.0.0", "--port", "8000"]
