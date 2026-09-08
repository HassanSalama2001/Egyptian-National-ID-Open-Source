import pytest
from national_id_ocr.postprocessing.national_id_parser import validate_nid_checksum, repair_nid

def test_checksum_valid():
    # Example valid IDs are hard to come by, but let's test one that passes the Mod-11 with cyclic weights
    # Assuming cyclic weights [2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4]
    # Let's construct one that sum % 11 == 1
    # 2 9 0 0 1 0 1 1 2 3 4 5 6 -> Checksum calc
    # Sum = (2*2) + (9*3) + (0*4) + (0*5) + (1*6) + (0*7) + (1*8) + (1*9) + (2*10) + (3*1) + (4*2) + (5*3) + (6*4)
    # Sum = 4 + 27 + 0 + 0 + 6 + 0 + 8 + 9 + 20 + 3 + 8 + 15 + 24 = 124
    # 124 % 11 = 3
    valid_id = "29001011234563" 
    assert validate_nid_checksum(valid_id) == True

def test_repair_single_digit_error():
    # Change '3' to 'S'
    noisy_id = "2900101123456S"
    repaired = repair_nid(noisy_id)
    assert repaired == "29001011234563"

def test_repair_common_confusion():
    # Change '0' to 'O'
    noisy_id = "29OO1011234563"
    repaired = repair_nid(noisy_id)
    assert repaired == "29001011234563"

if __name__ == "__main__":
    # If running manually, we can print results
    print(f"Repairing 2900101123456S -> {repair_nid('2900101123456S')}")
