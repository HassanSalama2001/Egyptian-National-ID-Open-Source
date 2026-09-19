"""
Covers the two rules that let a degraded back scan still answer usefully:
an expiry whose digits are unreadable degrades to the year+month the
7-year validity rule fixes (never to an invented day), and the
expired/not-expired answer stays honest about the one month where the
missing day is what decides it.
"""
from datetime import date

import pytest

from egyptian_national_id_ocr.core.pipeline import Pipeline


class TestUnreadableExpiryDegradesToYearMonth:
    def test_impossible_month_falls_back_to_issue_plus_validity(self):
        # A real greyscale scan read expiry as "2031/81/01" - there is no
        # month 81. Issue date 2024/08 fixes expiry at 2031/08.
        assert Pipeline._repair_expiry_year("2024/08", "2031/81/01") == "2031/08"

    def test_fallback_never_invents_a_day(self):
        recovered = Pipeline._repair_expiry_year("2024/08", "2031/81/01")
        assert recovered.count("/") == 1, "a day must not be fabricated"

    def test_impossible_day_also_falls_back(self):
        assert Pipeline._repair_expiry_year("2019/03", "2026/03/45") == "2026/03"

    def test_still_blanks_when_there_is_no_issue_date_to_derive_from(self):
        assert Pipeline._repair_expiry_year("", "2031/81/01") == ""

    def test_readable_expiry_is_left_alone(self):
        assert Pipeline._repair_expiry_year("2024/08", "2031/08/12") == "2031/08/12"


class TestExpiryWithoutADay:
    def test_month_before_expiry_month_is_not_expired(self):
        today = date.today()
        future = f"{today.year + 3:04d}/{today.month:02d}"
        assert Pipeline._card_is_expired(future) is False

    def test_month_after_expiry_month_is_expired(self):
        today = date.today()
        past = f"{today.year - 3:04d}/{today.month:02d}"
        assert Pipeline._card_is_expired(past) is True

    def test_current_month_is_unknown_not_guessed(self):
        # Inside the expiry month the answer depends on the day, which is
        # exactly what this form does not have - so neither True nor False.
        today = date.today()
        assert Pipeline._card_is_expired(f"{today.year:04d}/{today.month:02d}") is None

    def test_full_date_still_compares_on_the_day(self):
        today = date.today()
        assert Pipeline._card_is_expired(f"{today.year - 1:04d}/01/01") is True
        assert Pipeline._card_is_expired(f"{today.year + 1:04d}/12/31") is False

    def test_garbage_is_unknown(self):
        assert Pipeline._card_is_expired("2031/81") is None
        assert Pipeline._card_is_expired("") is None


class TestFreeTextPassMerge:
    """The rescan pass may only ADD to the raw pass, never replace it."""

    def test_fills_a_field_the_raw_pass_left_empty(self):
        merged = Pipeline._merge_free_text_passes({}, {"marital_status": ("أعزب", 0.9)})
        assert merged["marital_status"][0] == "أعزب"

    def test_extends_a_reading_it_fully_contains(self):
        merged = Pipeline._merge_free_text_passes(
            {"profession": ("بكالوريوس فى علوم", 0.94)},
            {"profession": ("بكالوريوس فى علوم الحاسب", 0.92)},
        )
        assert merged["profession"][0] == "بكالوريوس فى علوم الحاسب"

    def test_refuses_a_longer_reading_that_disagrees(self):
        merged = Pipeline._merge_free_text_passes(
            {"profession": ("مهندس", 0.94)},
            {"profession": ("محاسب قانونى", 0.99)},
        )
        assert merged["profession"][0] == "مهندس"

    def test_refuses_a_shorter_reading(self):
        # Measured: the normalized pass truncated this field to one word.
        merged = Pipeline._merge_free_text_passes(
            {"profession": ("بكالوريوس فى علوم", 0.94)},
            {"profession": ("بكالوريوس", 0.99)},
        )
        assert merged["profession"][0] == "بكالوريوس فى علوم"

    def test_ignores_an_empty_rescan_result(self):
        merged = Pipeline._merge_free_text_passes(
            {"profession": ("مهندس", 0.9)}, {"profession": ("", 0.0)}
        )
        assert merged["profession"][0] == "مهندس"
