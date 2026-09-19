class NationalIDOCRError(Exception):
    """Base exception for all errors in the National ID OCR pipeline."""
    pass

class CardNotFoundError(NationalIDOCRError):
    """Raised when the ID card cannot be detected in the image."""
    pass

class DetectionError(NationalIDOCRError):
    """Raised when there is an error during the card detection or perspective transform phase."""
    pass

class SegmentationError(NationalIDOCRError):
    """Raised when specific ROI regions cannot be extracted from the card."""
    pass

class OCRError(NationalIDOCRError):
    """Raised when the OCR engine fails to process an image."""
    pass

class ValidationError(NationalIDOCRError):
    """Raised when the extracted data fails validation rules."""
    pass

class PreprocessingError(NationalIDOCRError):
    """Raised when image preprocessing fails."""
    pass
