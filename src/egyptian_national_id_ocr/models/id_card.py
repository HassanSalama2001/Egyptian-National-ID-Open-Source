from datetime import date
from typing import Optional
from pydantic import BaseModel, Field
from .enums import Gender, Religion, MaritalStatus, Governorate, ExtractionStatus

class NationalIDDecoded(BaseModel):
    birth_date: Optional[str] = None
    governorate: Optional[Governorate] = Governorate.UNKNOWN
    gender: Optional[Gender] = Gender.UNKNOWN
    century: Optional[str] = None
    sequence: Optional[str] = None
    check_digit: Optional[int] = None

# Wording reused by every *_english field below. These are generated
# renderings, not official spellings: "محمد" is written Mohamed,
# Mohammed, Muhammad AND Mohamad on real Egyptian documents, and which
# one is correct for a given person is whatever their own papers say -
# it cannot be derived from the Arabic. See
# postprocessing/transliteration.py.
_ENGLISH_FIELD_CAVEAT = (
    " Machine transliteration for display/search - NOT the official "
    "spelling on the holder's documents, which cannot be derived from "
    "the Arabic."
)


class IDCardFront(BaseModel):
    first_name: Optional[str] = Field(None, description="First name in Arabic")
    full_name: Optional[str] = Field(None, description="Rest of the name in Arabic")
    address: Optional[str] = Field(None, description="Address in Arabic")
    national_id: Optional[str] = Field(None, description="14-digit National ID number")
    date_of_birth: Optional[str] = Field(None, description="Date of birth")
    card_serial_number: Optional[str] = Field(None, description="Physical card serial number (bottom left)")

    # Latin-script renderings of the Arabic fields above.
    first_name_english: Optional[str] = Field(None, description="First name in Latin script." + _ENGLISH_FIELD_CAVEAT)
    full_name_english: Optional[str] = Field(None, description="Rest of the name in Latin script." + _ENGLISH_FIELD_CAVEAT)
    address_english: Optional[str] = Field(None, description="Address in Latin script." + _ENGLISH_FIELD_CAVEAT)

    # Visual verification
    field_crops: dict[str, str] = Field(default_factory=dict, description="Base64 encoded snippet of each field area")
    raw_arabic: dict[str, str] = Field(default_factory=dict)

class IDCardBack(BaseModel):
    national_id: Optional[str] = None
    issue_date: Optional[str] = None  # Often format YYYY/MM
    profession: Optional[str] = None
    # gender/religion/marital_status are already English: they are
    # closed-vocabulary enums ("male", "muslim", "single") produced by
    # fuzzy-matching the Arabic against known values, so they need no
    # separate *_english counterpart. Only free text does.
    gender: Optional[Gender] = Gender.UNKNOWN
    religion: Optional[Religion] = Religion.UNKNOWN
    marital_status: Optional[MaritalStatus] = MaritalStatus.UNKNOWN
    expiry_date: Optional[str] = None

    profession_english: Optional[str] = Field(None, description="Profession in Latin script." + _ENGLISH_FIELD_CAVEAT)

    # None when expiry_date could not be read or validated - "we don't
    # know" is a different answer from "not expired", and a caller
    # deciding whether to accept this document needs to tell them apart.
    is_expired: Optional[bool] = Field(
        None,
        description=(
            "Whether the card has expired as of today, compared to the "
            "day (not just the month) of expiry_date. None if expiry_date "
            "is missing or failed validation."
        ),
    )

    # Visual verification
    field_crops: dict[str, str] = Field(default_factory=dict)
    raw_arabic: dict[str, str] = Field(default_factory=dict)

class IDCard(BaseModel):
    side: str = "unknown"
    front: Optional[IDCardFront] = None
    back: Optional[IDCardBack] = None
    decoded: Optional[NationalIDDecoded] = None
    confidence: float = 0.0
    processing_time_ms: int = 0
    status: ExtractionStatus = ExtractionStatus.NO_CARD_DETECTED
    # Plain-language, non-technical messages a UI can show directly to an
    # end user - never a stack trace or internal field/variable name.
    messages: list[str] = Field(default_factory=list)
    # Base64 JPEG of the rectified whole-card image (post alignment/
    # perspective-correction), for a UI to show "here's the card we
    # detected" alongside the individual field_crops - the same
    # transparency idea, one level up.
    card_image: Optional[str] = Field(None, description="Base64-encoded rectified card image")

    def without_images(self) -> "IDCard":
        """Returns a copy with every base64 image payload removed.

        The card image and per-field crops exist for the "show your work"
        UI, but they dominate the response size - dropping them is the
        difference between a payload measured in megabytes and one
        measured in kilobytes. Callers that only want the extracted data
        (a server-to-server integration, or an AI assistant that cannot
        look at an image anyway) should use this.

        A method on the model rather than logic in the API layer, so the
        HTTP API, the Python SDK and the MCP server all strip exactly the
        same set of fields.
        """
        copy = self.model_copy(deep=True)
        copy.card_image = None
        if copy.front is not None:
            copy.front.field_crops = {}
        if copy.back is not None:
            copy.back.field_crops = {}
        return copy
