"""Shared output contracts for OCR results and application callers."""

import json
import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from .errors import ExtractionError


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class FormField(StrictModel):
    number: int | str | None
    name: str = Field(min_length=1)
    value: str | None

    @field_validator("name")
    @classmethod
    def nonblank_name(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Field name must not be blank")
        return value


class FormTable(StrictModel):
    name: str | None
    columns: list[str]
    rows: list[list[str | None]]

    @model_validator(mode="after")
    def rectangular(self) -> "FormTable":
        width = len(self.columns)
        if width == 0 and self.rows:
            width = len(self.rows[0])
        if any(len(row) != width for row in self.rows):
            raise ValueError("All table rows must match the column count")
        return self


class PageExtraction(StrictModel):
    document_type: str | None
    fields: list[FormField]
    tables: list[FormTable]


class PageResult(PageExtraction):
    text: str | None = None
    page: int = Field(ge=1)
    image_width: int = Field(gt=0)
    image_height: int = Field(gt=0)
    ocr: dict[str, Any] | None = None


class DocumentResult(StrictModel):
    source: str
    mode: Literal["paddle"] = "paddle"
    model: str
    pages: list[PageResult]


def _unique_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON property")
        result[key] = value
    return result


def parse_extraction(response: str) -> PageExtraction:
    """Accept JSON or a single JSON fence; never salvage a partial result."""
    value = response.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*\n?(.*?)\n?```", value, flags=re.DOTALL)
    if fenced:
        value = fenced.group(1)
    try:
        data = json.loads(value, object_pairs_hook=_unique_keys)
        return PageExtraction.model_validate(data)
    except (ValueError, ValidationError) as exc:
        # Do not echo document contents in diagnostics.
        raise ExtractionError("Model output is not valid fields-and-tables JSON") from exc
