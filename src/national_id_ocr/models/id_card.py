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

class IDCardFront(BaseModel):
    first_name: Optional[str] = Field(None, description="First name in Arabic")
    full_name: Optional[str] = Field(None, description="Rest of the name in Arabic")
    address: Optional[str] = Field(None, description="Address in Arabic")
    national_id: Optional[str] = Field(None, description="14-digit National ID number")
    date_of_birth: Optional[str] = Field(None, description="Date of birth")
    card_serial_number: Optional[str] = Field(None, description="Physical card serial number (bottom left)")
    
    # Visual verification
    field_crops: dict[str, str] = Field(default_factory=dict, description="Base64 encoded snippet of each field area")
    raw_arabic: dict[str, str] = Field(default_factory=dict)

class IDCardBack(BaseModel):
    national_id: Optional[str] = None
    issue_date: Optional[str] = None  # Often format YYYY/MM
    profession: Optional[str] = None
    gender: Optional[Gender] = Gender.UNKNOWN
    religion: Optional[Religion] = Religion.UNKNOWN
    marital_status: Optional[MaritalStatus] = MaritalStatus.UNKNOWN
    expiry_date: Optional[str] = None

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
