import pytest
from egyptian_national_id_ocr.postprocessing.national_id_parser import (
    validate_nid_checksum,
    repair_nid,
    decode_national_id,
    is_structurally_valid_nid,
)

def test_checksum_valid():
    # Weights/formula verified against a real, physically-confirmed
    # Egyptian ID this session (see NID_CHECKSUM_WEIGHTS in
    # national_id_parser.py for the full story - the previous weights
    # here were a self-consistent guess that had never been checked
    # against a real card, and failed one when it finally was).
    # first_13 = "2900101123456", check digit computed as 4.
    valid_id = "29001011234564"
    assert validate_nid_checksum(valid_id) == True

def test_repair_single_digit_error():
    # Change '4' to 'S'
    noisy_id = "2900101123456S"
    repaired = repair_nid(noisy_id)
    assert repaired == "29001011234564"

def test_repair_common_confusion():
    # Change '0' to 'O'
    noisy_id = "29OO1011234564"
    repaired = repair_nid(noisy_id)
    assert repaired == "29001011234564"

def test_checksum_valid_but_impossible_nid_is_rejected():
    # Taken from a real scan that the pipeline reported as "extracted and
    # verified successfully" at 93% confidence. It genuinely passes
    # Mod-11 - a checksum over 14 digits accepts roughly 1 in 11 random
    # strings - but it cannot describe a real person: century digit '0'
    # (only 2 and 3 are issued) and governorate code '40' (not a real
    # code). The checksum must not be the only gate.
    impossible = "00212234006107"
    assert validate_nid_checksum(impossible) is True
    assert is_structurally_valid_nid(impossible) is False
    assert decode_national_id(impossible) is None
    assert repair_nid(impossible) is None


def test_invalid_century_does_not_fabricate_a_birth_year():
    # The year used to be computed as `1900 if code == 2 else 2000`, so
    # any unrecognized century code silently produced a confident 2000s
    # birth date rather than admitting the year was unknowable.
    assert decode_national_id("00212234006107") is None


def test_unknown_governorate_is_rejected():
    # Structurally fine except for governorate code '40'.
    from egyptian_national_id_ocr.postprocessing.national_id_parser import compute_nid_check_digit
    first_13 = "2900101400123"
    nid = first_13 + str(compute_nid_check_digit(first_13))
    assert validate_nid_checksum(nid) is True
    assert is_structurally_valid_nid(nid) is False


def test_structurally_valid_nid_still_accepted():
    # The known-good ID from test_checksum_valid must survive the new gate.
    assert is_structurally_valid_nid("29001011234564") is True


def test_nid_implying_someone_too_young_is_rejected():
    # Egyptian IDs are issued at 16, so a decoded birth date implying a
    # child means the digits were misread - a real scan produced exactly
    # this kind of reading.
    from datetime import date
    from egyptian_national_id_ocr.postprocessing.national_id_parser import compute_nid_check_digit
    # century(1) + yy(2) + mm(2) + dd(2) + governorate(2) + seq(3) + gender(1)
    recent_year = date.today().year - 2
    first_13 = f"3{recent_year % 100:02d}" + "0101" + "01" + "123" + "4"
    nid = first_13 + str(compute_nid_check_digit(first_13))
    assert validate_nid_checksum(nid) is True
    assert is_structurally_valid_nid(nid) is False


def test_implausible_ocr_birth_dates_rejected():
    from datetime import date
    from egyptian_national_id_ocr.postprocessing.national_id_parser import is_plausible_birth_date
    # Readings actually produced by real scans of the date-of-birth field.
    assert is_plausible_birth_date("4136-11-51") is False   # not a date at all
    assert is_plausible_birth_date("") is False
    # A parseable date that still can't be a cardholder's birth date.
    assert is_plausible_birth_date(f"{date.today().year - 2}-03-04") is False
    assert is_plausible_birth_date("2001-12-25") is True


if __name__ == "__main__":
    # If running manually, we can print results
    print(f"Repairing 2900101123456S -> {repair_nid('2900101123456S')}")
