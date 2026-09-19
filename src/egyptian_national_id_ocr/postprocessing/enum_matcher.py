"""
Fuzzy-matches OCR'd Arabic text for closed-vocabulary back-side fields
(gender, religion, marital status) against their known valid values,
instead of trusting free-text OCR output directly. Same "constrain to a
valid answer" idea as national_id_parser's checksum validation - OCR noise
on a short, known-vocabulary word is far more likely to still be closest to
the correct value than a random string, so snapping to the nearest known
option is more robust than passing the raw OCR text through.

Marital status has gendered forms on real cards (أعزب for a single man,
عزباء for a single woman, etc.) - both map to the same MaritalStatus member.
"""
from typing import Dict, Optional, Tuple

from .text_utils import levenshtein
from ..models.enums import Gender, Religion, MaritalStatus

GENDER_VALUES: Dict[str, Gender] = {
    "ذكر": Gender.MALE,
    "أنثى": Gender.FEMALE,
}

RELIGION_VALUES: Dict[str, Religion] = {
    "مسلم": Religion.MUSLIM,
    "مسلمة": Religion.MUSLIM,
    "مسيحي": Religion.CHRISTIAN,
    "مسيحية": Religion.CHRISTIAN,
    "يهودي": Religion.JEWISH,
    "يهودية": Religion.JEWISH,
}

MARITAL_STATUS_VALUES: Dict[str, MaritalStatus] = {
    "أعزب": MaritalStatus.SINGLE,
    "عزباء": MaritalStatus.SINGLE,
    "آنسة": MaritalStatus.SINGLE,
    "متزوج": MaritalStatus.MARRIED,
    "متزوجة": MaritalStatus.MARRIED,
    "مطلق": MaritalStatus.DIVORCED,
    "مطلقة": MaritalStatus.DIVORCED,
    "أرمل": MaritalStatus.WIDOWED,
    "أرملة": MaritalStatus.WIDOWED,
}


def match_enum(text: str, table: Dict[str, object], max_distance_ratio: float = 0.5) -> Tuple[Optional[object], float]:
    """Finds the closest known value to `text` by Levenshtein distance.

    Returns (enum_member, confidence) where confidence is 1.0 for an exact
    match and decreases with edit distance; (None, 0.0) if the input is
    empty, too different from every candidate, or too CLOSE to call
    between two candidates that map to different enum members.

    Raising max_distance_ratio alone doesn't do what it looks like it
    does for short words: for a 4-character candidate like "مسلم",
    max_distance_ratio*4 rounds down to the same integer distance cutoff
    for anything from ~0.26 to ~0.49 (confirmed: bumping 0.4 -> 0.45 was a
    no-op against a real card's OCR result that needed one more point of
    tolerance to match). 0.5 is the actual next integer-distance step up
    for a 4-char word - but 0.5 is *exactly* the closest ratio between
    two genuinely different answers anywhere in these tables (a 5-vs-6
    character pair, distance 3), so allowing it everywhere would risk
    that specific pair being accepted as a match for each other.

    Instead of picking a ratio that has to compromise between "generous
    enough for a short word" and "strict enough not to confuse two real
    answers", this checks each condition on the axis it actually belongs
    to: the ratio cutoff controls absolute plausibility ("is this
    reasonably close to *some* answer at all"), and a separate margin
    check - the best match must beat the best *differently-valued*
    candidate by at least 1 edit - handles distinguishing between two
    real answers, however close either one's ratio is to the cutoff.
    """
    text = (text or "").strip()
    if not text:
        return None, 0.0

    # Best distance per distinct ENUM VALUE (not per candidate string) -
    # multiple spellings (gendered forms) map to the same member and
    # shouldn't compete against each other for "best" vs "second-best".
    best_distance_per_value: Dict[object, int] = {}
    best_candidate_per_value: Dict[object, str] = {}
    for candidate, value in table.items():
        dist = levenshtein(text, candidate)
        if value not in best_distance_per_value or dist < best_distance_per_value[value]:
            best_distance_per_value[value] = dist
            best_candidate_per_value[value] = candidate

    if not best_distance_per_value:
        return None, 0.0

    ranked = sorted(best_distance_per_value.items(), key=lambda kv: kv[1])
    best_value, best_distance = ranked[0]
    best_candidate = best_candidate_per_value[best_value]

    if best_distance > max_distance_ratio * max(len(best_candidate), 1):
        return None, 0.0

    if len(ranked) > 1:
        _, second_distance = ranked[1]
        if second_distance - best_distance < 1:
            return None, 0.0  # too close to call between two real answers

    confidence = max(0.0, 1.0 - best_distance / max(len(best_candidate), 1))
    return best_value, confidence
