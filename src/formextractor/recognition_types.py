"""Recognition area types and the character sets for constrained fields."""

from typing import Literal

AreaKind = Literal["text", "number", "date"]
AREA_KINDS = ("text", "number", "date")
AREA_CHARACTERS = {
    "number": "0123456789.,+- ",
    "date": "0123456789./- ",
}
