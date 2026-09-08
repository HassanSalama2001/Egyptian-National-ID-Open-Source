import re

def normalize_arabic_numerals(text: str) -> str:
    """
    Converts Eastern Arabic numerals (٠١٢٣٤٥٦٧٨٩) to standard (0123456789).
    """
    arabic_map = {
        '٠': '0', '١': '1', '٢': '2', '٣': '3', '٤': '4',
        '٥': '5', '٦': '6', '٧': '7', '٨': '8', '٩': '9'
    }
    for arabic, western in arabic_map.items():
        text = text.replace(arabic, western)
    return text

def clean_ocr_text(text: str) -> str:
    """
    Cleans OCR output by removing unwanted characters and normalizing whitespace.
    """
    # Remove characters that are likely OCR artifacts (non-alphanumeric/punctuation)
    # Keep Arabic characters, numbers, and basic punctuation
    text = re.sub(r'[^\w\s\d\u0600-\u06FF/-]', '', text)
    
    # Normalize whitespace
    text = " ".join(text.split())
    return text.strip()
