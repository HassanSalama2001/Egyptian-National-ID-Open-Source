from datetime import date
from typing import Optional
from ..models.id_card import NationalIDDecoded
from ..models.enums import Gender, Governorate

GOVERNORATE_CODES = {
    "01": Governorate.CAIRO,
    "02": Governorate.ALEXANDRIA,
    "03": Governorate.PORT_SAID,
    "04": Governorate.SUEZ,
    "11": Governorate.DAMIETTA,
    "12": Governorate.DAKAHLIA,
    "13": Governorate.SHARQUIA,
    "14": Governorate.QALYUBIA,
    "15": Governorate.KAFR_EL_SHEIKH,
    "16": Governorate.GHARBIA,
    "17": Governorate.MONUFIA,
    "18": Governorate.BEHEIRA,
    "19": Governorate.ISMAILIA,
    "21": Governorate.GIZA,
    "22": Governorate.BENI_SUEF,
    "23": Governorate.FAYOUM,
    "24": Governorate.MINYA,
    "25": Governorate.ASSIUT,
    "26": Governorate.SOHAG,
    "27": Governorate.QENA,
    "28": Governorate.ASWAN,
    "29": Governorate.LUXOR,
    "31": Governorate.RED_SEA,
    "32": Governorate.NEW_VALLEY,
    "33": Governorate.MATROUH,
    "34": Governorate.NORTH_SINAI,
    "35": Governorate.SOUTH_SINAI,
    "88": Governorate.OUTSIDE_EGYPT,
}

def decode_national_id(nid: str) -> Optional[NationalIDDecoded]:
    """
    Decodes the 14-digit Egyptian National ID.
    Format: C YY MM DD BB SSS G K
    C: Century (2=1900-1999, 3=2000-2099)
    YYMMDD: Date of Birth
    BB: Birth Governorate Code
    SSS: Sequence number
    G: Gender (Odd=Male, Even=Female)
    K: Check Digit
    """
    if not nid or len(nid) != 14 or not nid.isdigit():
        return None
        
    try:
        century_code = int(nid[0])
        century = "1900s" if century_code == 2 else "2000s" if century_code == 3 else "Unknown"
        
        year = int(nid[1:3])
        full_year = (1900 if century_code == 2 else 2000) + year
        month = int(nid[3:5])
        day = int(nid[5:7])
        
        birth_date = date(full_year, month, day)
        
        gov_code = nid[7:9]
        governorate = GOVERNORATE_CODES.get(gov_code, Governorate.UNKNOWN)
        
        gender_digit = int(nid[12])
        gender = Gender.MALE if gender_digit % 2 != 0 else Gender.FEMALE
        
        return NationalIDDecoded(
            birth_date=birth_date,
            governorate=governorate,
            gender=gender,
            century=century,
            sequence=nid[9:13],
            check_digit=int(nid[13])
        )
    except Exception:
        return None

def validate_nid_checksum(nid: str) -> bool:
    """
    Validates the 14th digit of the Egyptian National ID using the Mod-11 algorithm.
    Weights: [2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4] (Standard cyclic)
    """
    if not nid or len(nid) != 14 or not nid.isdigit():
        return False
    
    weights = [2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4]
    digits = [int(d) for d in nid]
    
    checksum = 0
    for i in range(13):
        checksum += digits[i] * weights[i]
    
    calculated_digit = checksum % 11
    if calculated_digit == 10:
        calculated_digit = 0
        
    return calculated_digit == digits[13]

def repair_nid(noisy_nid: str) -> Optional[str]:
    """
    Attempts to fix single-digit OCR errors in a 14-digit NID string
    by iterating through common confusion characters and validating against the checksum.
    """
    if not noisy_nid: return None
    
    # 1. Basic purification
    import re
    cleaned = "".join(re.findall(r'[\dA-Z|!OISGZTBA]', noisy_nid.upper()))
    if len(cleaned) < 14: return None
    
    # 2. Try the top 14 chars
    candidate = fuzzy_digit_repair(cleaned[:14])
    if validate_nid_checksum(candidate):
        return candidate
    
    # 3. OCR Confusion Brute Force (Single character swap)
    confusions = {
        '0': ['O', 'D', 'Q', 'U'],
        '1': ['I', 'l', '|', '!', '7'],
        '2': ['Z'],
        '3': [],
        '4': ['A'],
        '5': ['S'],
        '6': ['G', 'b'],
        '7': ['T', '1'],
        '8': ['B'],
        '9': ['g', 'q']
    }
    
    # Inverse mapping for repairs
    reverse_conf = {}
    for digit, chars in confusions.items():
        for char in chars:
            reverse_conf[char] = digit
            
    # Try swapping one potentially confused character at a time
    for i in range(len(candidate)):
        original_char = cleaned[i]
        if original_char in reverse_conf:
            # We already tried the primary mapping in fuzzy_digit_repair
            # But maybe it was something else? No, fuzzy_digit_repair is deterministic.
            pass
            
        # Let's try every digit at this position if it's suspicious
        # But even better: cycle through 0-9 for each position if the checksum fails
        # This is fast for only 14 positions.
        original_digit = candidate[i]
        for d in "0123456789":
            if d == original_digit: continue
            
            test_nid = candidate[:i] + d + candidate[i+1:]
            if validate_nid_checksum(test_nid):
                # Extra validation: does it make sense decoded?
                if decode_national_id(test_nid):
                    return test_nid
                    
    return None

def fuzzy_digit_repair(text: str) -> str:
    """
    Common OCR confusion mapping:
    - O, D, Q, U -> 0
    - I, l, |, ! -> 1
    - S -> 5
    - G, b -> 6
    - B -> 8
    - Z -> 2
    - T, t -> 7
    - A -> 4
    """
    mapping = {
        'O': '0', 'D': '0', 'Q': '0', 'U': '0',
        'I': '1', 'l': '1', '|': '1', '!': '1',
        'S': '5', 'G': '6', 'b': '6', 'B': '8',
        'Z': '2', 'T': '7', 't': '7', 'A': '4'
    }
    repaired = ""
    for char in text.upper():
        if char.isdigit():
            repaired += char
        elif char in mapping:
            repaired += mapping[char]
    return repaired

def find_nid_in_text(text: str) -> Optional[str]:
    """
    Searches for a 14-digit National ID pattern in a text block.
    Uses fuzzy repair to salvage noisy text.
    """
    import re
    from .numeral_converter import normalize_arabic_numerals
    
    # 1. Normalize numerals
    text = normalize_arabic_numerals(text)
    
    # 2. Extract potential numeric sequences including common confusion characters
    # We look for any sequence of 14 characters that could be a NID after repair
    # First, handle the obvious ones:
    digits_only = re.sub(r'[^\d]', '', text)
    match = re.search(r'[23]\d{13}', digits_only)
    if match:
        return match.group(0)
        
    # 3. Fuzzy match: Look for alphanumeric strings that "look" like a NID
    # We strip spaces and punctuation first
    cleaned = re.sub(r'[\s.,\-_]', '', text)
    # Check for any 14-char sequence that starts with 2, 3 or equivalent
    # (e.g. 'Z' for '2', 'S' for '5' if it's the 1st digit? No, first digit is 2 or 3)
    
    # We'll try to find all sequences of 13-15 chars and repair them
    potential_matches = re.findall(r'[23Z][\dA-Z]{13,14}', cleaned.upper())
    for pot in potential_matches:
        # Try direct repair first
        repaired = repair_nid(pot)
        if repaired:
            return repaired
            
        # Fallback to fuzzy repair if checksum fails but it looks valid otherwise
        simple_repaired = fuzzy_digit_repair(pot)
        if len(simple_repaired) == 14 and simple_repaired[0] in '23':
            if decode_national_id(simple_repaired):
                return simple_repaired
                
    return None
