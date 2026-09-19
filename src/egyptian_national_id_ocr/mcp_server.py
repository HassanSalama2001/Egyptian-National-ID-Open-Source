"""
MCP (Model Context Protocol) server exposing this pipeline as tools an AI
assistant can call.

Run it directly (stdio transport, which is what local MCP clients use):

    egy-nid-ocr-mcp

Or point a client at it explicitly, e.g. in Claude Desktop's config:

    {
      "mcpServers": {
        "egyptian-national-id-ocr": {
          "command": "egy-nid-ocr-mcp"
        }
      }
    }

DESIGN NOTES (the non-obvious parts):

* The pipeline is built lazily, on the first extraction, NOT at import.
  Constructing it loads PaddleOCR models and takes seconds; MCP clients
  start their servers at launch and can time out waiting. Importing this
  module must stay cheap.

* validate_national_id deliberately touches none of that, so asking
  "is this number valid?" costs milliseconds instead of a model load.

* Tool results strip the base64 card/field images the HTTP API returns
  for its web UI (via IDCard.without_images, the same path the API's
  include_images=false uses). A model cannot look at them, and they would
  consume an enormous amount of its context for nothing.

* These tools return personal data read off an identity document. That is
  the whole point of the pipeline, but it is worth being deliberate about:
  anything a tool returns here enters the calling assistant's context.
"""
import logging
from pathlib import Path
from typing import Any

from mcp.server.mcpserver import MCPServer
from mcp.types import ToolAnnotations

logger = logging.getLogger(__name__)

mcp = MCPServer(
    name="egyptian-national-id-ocr",
    instructions=(
        "Extracts and validates data from EGYPTIAN national ID cards. "
        "It does not work for other countries' identity documents - the "
        "field positions, card templates, checksum, and digit classifier "
        "are all specific to the Egyptian card."
    ),
)

_pipeline = None


def _get_pipeline():
    """Builds the pipeline on first use - see this module's design notes."""
    global _pipeline
    if _pipeline is None:
        from .core.pipeline import Pipeline

        logger.info("Loading OCR pipeline (first call only)...")
        _pipeline = Pipeline()
    return _pipeline


def _load_image(path: Path):
    """Decodes an image the same way the HTTP API does, honouring the EXIF
    orientation tag - a phone photo commonly stores its pixels sideways
    plus a tag saying how to rotate for display, and ignoring that tag
    would feed the pipeline a rotated card."""
    import cv2
    import numpy as np
    from PIL import Image, ImageOps

    pil_image = ImageOps.exif_transpose(Image.open(path))
    return cv2.cvtColor(np.array(pil_image.convert("RGB")), cv2.COLOR_RGB2BGR)




# structured_output + a precise return annotation give the client a real
# output schema and parsed data, instead of JSON it has to parse back out
# of a text block. A bare `dict` annotation is not specific enough for the
# SDK to generate that schema - `dict[str, Any]` is.
@mcp.tool(
    title="Extract Egyptian National ID card",
    annotations=ToolAnnotations(readOnlyHint=True, openWorldHint=False),
    structured_output=True,
)
def extract_id_card(image_path: str) -> dict[str, Any]:
    """Read the fields from a photo or scan of an Egyptian national ID card.

    Works on either side of the card and detects which side it is. The
    front yields name, address, national ID number, date of birth and card
    serial; the back yields national ID, issue and expiry dates,
    profession, gender, religion and marital status.

    The returned `status` says how much to trust the result:
      - "success": the national ID passed its checksum and structural
        checks, or a corrected digit was confirmed by another field.
      - "low_confidence": a card was read but the national ID could not be
        verified, OR a digit was corrected with nothing confirming it.
        Treat the values as needing human review rather than as facts.
      - "no_card_detected" / "unreadable_image": nothing usable was found.

    Args:
        image_path: Path to a JPG or PNG image file on this machine.
    """
    path = Path(image_path).expanduser()
    if not path.is_file():
        return {
            "status": "error",
            "messages": [f"No image file at {path}"],
        }

    try:
        image = _load_image(path)
    except Exception as exc:
        return {
            "status": "unreadable_image",
            "messages": [f"Could not read {path.name} as an image: {exc}"],
        }

    result = _get_pipeline().process_image(image)
    # without_images() is shared with the HTTP API's include_images=false
    # path, so both strip exactly the same fields.
    return result.without_images().model_dump(mode="json")


@mcp.tool(
    title="Validate Egyptian national ID number",
    annotations=ToolAnnotations(readOnlyHint=True, openWorldHint=False),
    structured_output=True,
)
def validate_national_id(national_id: str) -> dict[str, Any]:
    """Check a 14-digit Egyptian national ID number and decode what it encodes.

    Validates the Mod-11 check digit AND that the number describes someone
    who could exist - a valid checksum alone is not enough, since roughly
    1 in 11 arbitrary 14-digit strings satisfies it by chance.

    On success also returns the birth date, birth governorate, gender and
    century encoded in the number itself. Needs no image and loads no OCR
    models, so it is cheap to call.

    Args:
        national_id: The 14-digit number, e.g. "29001011234564".
    """
    from .postprocessing.national_id_parser import (
        decode_national_id,
        is_structurally_valid_nid,
        validate_nid_checksum,
    )

    digits = "".join(ch for ch in national_id if ch.isdigit())
    if len(digits) != 14:
        return {
            "valid": False,
            "reason": f"Expected 14 digits, got {len(digits)}.",
        }

    checksum_ok = validate_nid_checksum(digits)
    structurally_valid = is_structurally_valid_nid(digits)
    if not checksum_ok:
        return {"valid": False, "reason": "Mod-11 check digit does not match."}
    if not structurally_valid:
        return {
            "valid": False,
            "reason": (
                "Check digit matches, but the number cannot describe a real "
                "cardholder (invalid century digit, unknown governorate "
                "code, or an impossible birth date)."
            ),
        }

    decoded = decode_national_id(digits)
    return {
        "valid": True,
        "national_id": digits,
        "birth_date": decoded.birth_date,
        "governorate": decoded.governorate.value,
        "gender": decoded.gender.value,
        "century": decoded.century,
    }


def main() -> None:
    """Console-script entry point - runs over stdio, the transport local
    MCP clients use to talk to a server they launched as a subprocess."""
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
