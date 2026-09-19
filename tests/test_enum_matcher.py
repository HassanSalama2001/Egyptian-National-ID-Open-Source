from egyptian_national_id_ocr.postprocessing.enum_matcher import (
    match_enum, GENDER_VALUES, RELIGION_VALUES, MARITAL_STATUS_VALUES,
)
from egyptian_national_id_ocr.models.enums import Gender, Religion, MaritalStatus


def test_exact_matches():
    assert match_enum("ذكر", GENDER_VALUES)[0] == Gender.MALE
    assert match_enum("أنثى", GENDER_VALUES)[0] == Gender.FEMALE
    assert match_enum("مسلم", RELIGION_VALUES)[0] == Religion.MUSLIM
    assert match_enum("مسيحي", RELIGION_VALUES)[0] == Religion.CHRISTIAN


def test_exact_match_has_full_confidence():
    _, confidence = match_enum("ذكر", GENDER_VALUES)
    assert confidence == 1.0


def test_tolerates_single_character_ocr_noise():
    # 'ذكر' with one character swapped for a common OCR confusion
    value, confidence = match_enum("دكر", GENDER_VALUES)
    assert value == Gender.MALE
    assert 0.0 < confidence < 1.0


def test_gendered_marital_status_forms_resolve_to_same_member():
    assert match_enum("أعزب", MARITAL_STATUS_VALUES)[0] == MaritalStatus.SINGLE
    assert match_enum("عزباء", MARITAL_STATUS_VALUES)[0] == MaritalStatus.SINGLE
    assert match_enum("متزوج", MARITAL_STATUS_VALUES)[0] == MaritalStatus.MARRIED
    assert match_enum("متزوجة", MARITAL_STATUS_VALUES)[0] == MaritalStatus.MARRIED


def test_empty_input_returns_none():
    assert match_enum("", GENDER_VALUES) == (None, 0.0)
    assert match_enum(None, GENDER_VALUES) == (None, 0.0)


def test_unrelated_text_returns_none():
    value, confidence = match_enum("شارع الجمهورية", GENDER_VALUES)
    assert value is None
    assert confidence == 0.0
