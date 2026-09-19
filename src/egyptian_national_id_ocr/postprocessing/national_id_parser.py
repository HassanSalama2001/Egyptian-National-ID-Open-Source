import logging
from datetime import date
from typing import Optional
from ..models.id_card import NationalIDDecoded
from ..models.enums import Gender, Governorate

logger = logging.getLogger(__name__)

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
        # Only 2 (1900s) and 3 (2000s) are real, issued century codes.
        # This used to fall through to "Unknown" for anything else while
        # STILL decoding the rest - and, worse, the year calculation below
        # used `1900 if code == 2 else 2000`, so an invalid century code
        # silently produced a 2000s birth year out of nothing. Confirmed
        # on a real scan: a checksum-valid-but-wrong candidate starting
        # with '0' decoded to a confident, entirely fabricated 2002 birth
        # date, was reported as "verified successfully" at 93%
        # confidence, and the "Unknown" century sitting right next to it
        # in the UI was the only hint anything was wrong. A century we
        # can't identify means we cannot know the birth year at all, so
        # there is nothing honest to return here.
        if century_code not in (2, 3):
            return None
        century = "1900s" if century_code == 2 else "2000s"

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
            birth_date=birth_date.isoformat(),
            governorate=governorate,
            gender=gender,
            century=century,
            sequence=nid[9:13],
            check_digit=int(nid[13])
        )
    except Exception as e:
        logger.debug(f"decode_national_id failed for {nid!r}: {e}")
        return None

# This project's checksum was, for its entire life until now, using the
# wrong weights ([2,3,4,5,6,7,8,9,10,1,2,3,4], a generic cyclic Mod-11
# guess) and the wrong final step (checksum % 11 directly). It had never
# been checked against a real government-issued ID - the project's own
# test only constructed a self-consistent example under its own (wrong)
# formula, and the synthetic generator used the same formula to produce
# its "ground truth", so every check up to this point was only agreeing
# with itself. Caught when a real user's confirmed-correct real card
# number failed this checksum; verified these weights and formula
# against that real number, cross-checked against an independently
# published Egyptian ID validator (github.com/MohamedAAbdallah/
# Egyptian-ID-Validator-Py) - both give the correct check digit for that
# real card, unlike the old formula.
NID_CHECKSUM_WEIGHTS = [2, 7, 6, 5, 4, 3, 2, 7, 6, 5, 4, 3, 2]


def compute_nid_check_digit(first_13_digits: str) -> int:
    """
    Computes the 14th (check) digit for the first 13 digits of an Egyptian
    National ID using the real Mod-11 algorithm (see NID_CHECKSUM_WEIGHTS
    for how this was verified). Shared by validate_nid_checksum
    (verification) and the synthetic data generator (so generated samples
    carry a real, verifiable checksum instead of a placeholder digit).
    """
    if len(first_13_digits) != 13 or not first_13_digits.isdigit():
        raise ValueError("Expected exactly 13 digits")

    digits = [int(d) for d in first_13_digits]
    checksum = sum(d * w for d, w in zip(digits, NID_CHECKSUM_WEIGHTS))
    # Note the direction: 11 - (checksum % 11), not checksum % 11 directly
    # - this, not just the weights, was also wrong in the old formula.
    calculated_digit = 11 - (checksum % 11)
    if calculated_digit == 10:
        return 0
    if calculated_digit == 11:
        return 1
    return calculated_digit


def validate_nid_checksum(nid: str) -> bool:
    """
    Validates the 14th digit of the Egyptian National ID using the real
    Mod-11 algorithm - see NID_CHECKSUM_WEIGHTS for how this was verified.
    """
    if not nid or len(nid) != 14 or not nid.isdigit():
        return False

    return compute_nid_check_digit(nid[:13]) == int(nid[13])


# Nobody holding a currently-valid Egyptian ID was born more than this
# many years ago - a generous bound, used only to reject decoded birth
# dates that are obviously impossible rather than to judge real people.
MAX_PLAUSIBLE_AGE_YEARS = 120

# Egyptian National IDs are issued at 16; nobody younger holds one, so a
# decoded birth date implying a younger person means the digits are
# misread. Set to 15 rather than 16 deliberately - one year of slack so a
# genuine edge case can never be rejected, while still ruling out the
# large majority of impossible readings. Confirmed useful on a real scan
# whose misread national_id decoded to a birth date that had not happened
# yet.
MIN_PLAUSIBLE_AGE_YEARS = 15

# (what the classifier READ, what was actually PRINTED) - Arabic-Indic
# glyph pairs it is repeatedly confirmed to confuse on real cards, both
# cases where the distinguishing feature is a thin stroke that harsh
# binarization erodes away:
#   '٣' (3) read as '٢' (2) - the extra tooth on the 3 is lost
#   '٥' (5) read as '٠' (0) - the 5's tail collapses into a round blob
# These are what makes a repair "evidence-backed" in repair_nid_detailed:
# something specific points at that digit being wrong, as opposed to
# searching every position blindly.
#
# ORDERED, most-confirmed first, and searched one tier at a time - NOT
# as one combined set. Pooling them cost real accuracy: a greyscale scan
# whose only error was a single '٣'->'٢' went unrepaired, because
# allowing '٠'->'٥' substitutions at the same time produced a second
# checksum-valid candidate and the ambiguity guard (correctly) refused to
# guess between them. Trying the better-evidenced pair alone first
# restores its full recovery rate while keeping the weaker rule available
# when it is the only explanation on offer.
KNOWN_DIGIT_CONFUSIONS = (("2", "3"), ("0", "5"))


def is_structurally_valid_nid(nid: str) -> bool:
    """
    Checks that a 14-digit NID is *possible*, not merely checksum-valid.

    A Mod-11 checksum is a single linear equation over 14 digits, so
    roughly 1 in 11 arbitrary digit strings satisfies it by chance. Any
    search that tries multiple candidate readings (repair_nid does
    exactly that) will therefore manufacture checksum-valid garbage given
    enough attempts - so the checksum cannot be the only gate. Confirmed
    on a real scan: an OCR misread passed the checksum outright and was
    presented as "National ID extracted and verified successfully" at 93%
    confidence, even though its own decoded output said Unknown century,
    Unknown governorate, and the wrong gender.

    This is the second, independent gate: the fields the NID encodes have
    to describe a person who could actually exist.
    """
    decoded = decode_national_id(nid)
    if decoded is None:
        # Invalid century code, or a date like month 13 / day 32.
        return False

    # An unrecognized governorate code means these digits aren't a real
    # birthplace - GOVERNORATE_CODES covers every Egyptian governorate
    # plus 88 (born outside Egypt), so there is no legitimate code left
    # over for this to reject.
    if decoded.governorate == Governorate.UNKNOWN:
        return False

    try:
        birth = date.fromisoformat(decoded.birth_date)
    except ValueError:
        return False

    today = date.today()
    if birth > today:
        return False
    age = today.year - birth.year
    if age > MAX_PLAUSIBLE_AGE_YEARS or age < MIN_PLAUSIBLE_AGE_YEARS:
        return False

    return True


def is_plausible_birth_date(iso_date: str) -> bool:
    """
    Whether an independently-read (OCR'd) date of birth could belong to
    someone holding an Egyptian National ID - same age bounds the decoded
    NID is held to.

    Used to decide whether the date-of-birth field is trustworthy enough
    to (a) cross-check a repaired NID against and (b) show to the user.
    Real scans produced readings like "4136/11/51" and "3001/11/05" for
    this field; feeding those into the NID cross-check as if they were
    real would be worse than having no cross-check at all, and displaying
    them is just noise.
    """
    try:
        birth = date.fromisoformat(iso_date)
    except (ValueError, TypeError):
        return False

    today = date.today()
    if birth > today:
        return False
    age = today.year - birth.year
    return MIN_PLAUSIBLE_AGE_YEARS <= age <= MAX_PLAUSIBLE_AGE_YEARS

class NIDRepairResult:
    """
    Outcome of repair_nid_detailed. `repaired` and `corroborated` exist so
    the pipeline can tell a number we simply READ and verified from one we
    CHANGED digits on to make it verify - see repair_nid_detailed for why
    those two deserve very different confidence.
    """
    __slots__ = ("nid", "repaired", "corroborated")

    def __init__(self, nid: Optional[str], repaired: bool = False, corroborated: bool = False):
        self.nid = nid
        self.repaired = repaired
        self.corroborated = corroborated


def _candidate_is_acceptable(nid: str) -> bool:
    """Checksum + structural plausibility, with no reference to any
    independently-read field - see _candidate_matches_corroboration for
    why those two questions are kept separate."""
    return validate_nid_checksum(nid) and is_structurally_valid_nid(nid)


def _candidate_matches_corroboration(
    nid: str,
    expected_birth_date: Optional[str] = None,
    expected_gender: Optional[Gender] = None,
) -> bool:
    """Whether a candidate also agrees with the independently-read fields.

    Deliberately separate from _candidate_is_acceptable: agreement is a
    reason to TRUST a candidate more, never a reason to discard one that
    is otherwise valid. Confirmed on a real greyscale scan where the
    date-of-birth field misread as a different but entirely plausible
    date - folding that into the acceptance test made the wrong date veto
    the correct national ID, so a card that had previously verified
    stopped verifying. The reading we cross-check against can itself be
    wrong; it gets a vote, not a veto.

    Returns False when there is nothing to check against. "No field
    disagreed with this" is not the same claim as "another field
    confirmed it" - conflating them reported a repair nobody had
    corroborated as fully verified, which is the exact overstatement the
    corroborated flag exists to prevent.
    """
    if not expected_birth_date and expected_gender is None:
        return False

    decoded = decode_national_id(nid)
    if decoded is None:
        return False
    if expected_birth_date and decoded.birth_date != expected_birth_date:
        return False
    if expected_gender is not None and decoded.gender != expected_gender:
        return False
    return True


def repair_nid_detailed(
    noisy_nid: str,
    expected_birth_date: Optional[str] = None,
    expected_gender: Optional[Gender] = None,
) -> NIDRepairResult:
    """
    Attempts to fix single-digit OCR errors in a 14-digit NID string,
    validating candidates against the checksum, structural plausibility,
    and - when available - independently-read fields from the card.

    WHY CORROBORATION IS REQUIRED FOR THE GENERAL CASE: a Mod-11 checksum
    is one linear equation, so for almost any wrong digit there exists
    some *other* single-digit change at a *different* position that also
    satisfies it. Brute-forcing every position therefore doesn't "repair"
    a number so much as find a different number that happens to validate.
    Measured over 3000 synthetic IDs per condition:

        errors   recovered   WRONG ANSWER   declined
          1        28.2%         9.8%         62.0%
          2         0.0%        20.2%         79.8%
          3         0.0%        20.1%         79.9%

    i.e. with 2+ OCR errors it can never be right, yet still returned a
    confidently wrong ID a fifth of the time - and a "repaired" ID
    populates the decoded block, which reads to the user as verified.
    Requiring the candidate's decoded birth date to match the separately-
    OCR'd date-of-birth field changes those numbers to:

        errors   recovered   WRONG ANSWER   declined
          1        55.3%         0.0%         44.7%
          2         0.0%         3.0%         97.0%
          3         0.0%         5.0%         95.0%

    - roughly double the recoveries AND no false repairs at one error,
    because a wrong candidate almost never also produces the right birth
    date. This is why date_of_birth is read from the card rather than
    only derived from the NID: it is the independent signal that makes
    repairing the NID safe.

    Without any corroborating signal, only evidence-backed edits are
    tried: positions the OCR itself flagged as ambiguous (a letter rather
    than a digit) and the digit classifier's one repeatedly-confirmed
    Arabic-Indic confusion ('٣' read as '٢'). The unrestricted "any digit
    at any position" search is NOT run uncorroborated - that is the pass
    that produced the wrong answers above.
    """
    if not noisy_nid:
        return NIDRepairResult(None)

    # 1. Basic purification
    import re
    cleaned = "".join(re.findall(r'[\dA-Z|!OISGZTBA]', noisy_nid.upper()))
    if len(cleaned) < 14:
        return NIDRepairResult(None)

    has_corroboration = bool(expected_birth_date) or expected_gender is not None
    candidate = fuzzy_digit_repair(cleaned[:14])
    if len(candidate) < 14:
        return NIDRepairResult(None)
    candidate = candidate[:14]

    # 2. The number as read, unchanged. This is the only path that can
    # claim the ID was verified rather than reconstructed. A checksum-
    # and structure-valid reading stands on its own here even if the
    # date-of-birth field disagrees with it - that field is itself an OCR
    # reading and is the likelier of the two to be wrong.
    if _candidate_is_acceptable(candidate):
        return NIDRepairResult(
            candidate,
            repaired=False,
            corroborated=_candidate_matches_corroboration(
                candidate, expected_birth_date, expected_gender
            ),
        )

    # 3. Single-digit repairs. Positions where OCR returned a non-digit,
    # plus the classifier's known confusions (KNOWN_DIGIT_CONFUSIONS), are
    # "evidence-backed": something specific points at that position being
    # wrong. Every other position is only searched when an independent
    # field can confirm the result.
    suspicious_positions = {i for i, ch in enumerate(cleaned[:14]) if not ch.isdigit()}

    # Evidence tiers, strongest first: positions OCR itself flagged as
    # ambiguous (it returned a letter, not a digit), then each known
    # glyph confusion in order of how well confirmed it is. Kept separate
    # rather than pooled - see KNOWN_DIGIT_CONFUSIONS.
    evidence_tiers = [set() for _ in range(1 + len(KNOWN_DIGIT_CONFUSIONS))]
    speculative = set()
    corroborated_candidates = set()
    for i in range(14):
        for d in "0123456789":
            if d == candidate[i]:
                continue
            test_nid = candidate[:i] + d + candidate[i + 1:]
            if not _candidate_is_acceptable(test_nid):
                continue
            if _candidate_matches_corroboration(test_nid, expected_birth_date, expected_gender):
                corroborated_candidates.add(test_nid)

            if i in suspicious_positions:
                evidence_tiers[0].add(test_nid)
                continue
            for tier, pair in enumerate(KNOWN_DIGIT_CONFUSIONS, start=1):
                if (candidate[i], d) == pair:
                    evidence_tiers[tier].add(test_nid)
                    break
            else:
                speculative.add(test_nid)

    if has_corroboration:
        # With an independent field to check against, every position can
        # be searched - if exactly one candidate satisfies checksum,
        # structure AND that field, it's the answer.
        if len(corroborated_candidates) == 1:
            return NIDRepairResult(corroborated_candidates.pop(), repaired=True, corroborated=True)

    # Otherwise fall back to a unique edit that something specific
    # pointed at. Measured against its actual failure mode (the
    # classifier reading '٣' as '٢') this recovers 84.5% with 0.0%
    # wrong - it encodes a real, observed confusion rather than
    # searching blindly.
    #
    # This runs for the corroborated case too, and deliberately AFTER
    # the search above: requiring union-uniqueness alone dropped that
    # same 84.5% to 47.4%, i.e. having MORE information made the result
    # worse, which is never the right trade. Corroboration should only
    # ever add capability, so it opens up the wider search without
    # closing off the narrow rule that already worked.
    # Take the first tier that offers exactly one explanation. A weaker
    # rule never gets to create ambiguity for a stronger one.
    fix = None
    for tier in evidence_tiers:
        if len(tier) == 1:
            fix = tier.pop()
            break

    if fix is not None:
        return NIDRepairResult(
            fix,
            repaired=True,
            # Corroborated only if the cross-checked field actually agrees
            # with THIS candidate. When it disagrees the fix still stands
            # (a misread date must not veto a valid ID - see
            # _candidate_matches_corroboration) but it is reported as
            # unconfirmed, so confidence and status reflect that honestly
            # rather than borrowing trust the evidence doesn't support.
            corroborated=_candidate_matches_corroboration(
                fix, expected_birth_date, expected_gender
            ),
        )

    # Ambiguous, or only speculative fixes with nothing to confirm them -
    # surfacing "needs review" beats inventing a plausible wrong ID.
    return NIDRepairResult(None)


def repair_nid(
    noisy_nid: str,
    expected_birth_date: Optional[str] = None,
    expected_gender: Optional[Gender] = None,
) -> Optional[str]:
    """Convenience wrapper returning just the repaired NID - see
    repair_nid_detailed when you need to know whether digits were changed
    or whether an independent field confirmed the result."""
    return repair_nid_detailed(noisy_nid, expected_birth_date, expected_gender).nid


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
            if is_structurally_valid_nid(simple_repaired):
                return simple_repaired
                
    return None
