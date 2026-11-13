"""Numbered, page-specific recognition templates and tabular batch results."""

import fnmatch
import math
import os
from collections.abc import Callable
from contextlib import closing
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from .documents import IMAGE_EXTENSIONS, iter_pages
from .errors import ExtractionError
from .paddle import PaddleLayout
from .schema import StrictModel


class RecognitionArea(StrictModel):
    number: int = Field(ge=1)
    page: int = Field(ge=1)
    x1: float = Field(ge=0, le=1, allow_inf_nan=False)
    y1: float = Field(ge=0, le=1, allow_inf_nan=False)
    x2: float = Field(ge=0, le=1, allow_inf_nan=False)
    y2: float = Field(ge=0, le=1, allow_inf_nan=False)

    @model_validator(mode="after")
    def positive_size(self) -> "RecognitionArea":
        if self.x2 <= self.x1 or self.y2 <= self.y1:
            raise ValueError("Recognition areas must have positive width and height")
        return self

    def pixel_box(self, width: int, height: int) -> tuple[int, int, int, int]:
        return (
            math.floor(self.x1 * width),
            math.floor(self.y1 * height),
            math.ceil(self.x2 * width),
            math.ceil(self.y2 * height),
        )


class BatchRow(StrictModel):
    file: str
    values: list[str | None]
    error: str | None = None


class BatchResult(StrictModel):
    source: str
    mode: Literal["areas"] = "areas"
    model: str = "PP-StructureV3"
    language: str
    file_mask: str
    areas: list[RecognitionArea]
    columns: list[int]
    rows: list[BatchRow]

    @model_validator(mode="after")
    def rectangular(self) -> "BatchResult":
        numbers = sorted(area.number for area in self.areas)
        if not numbers or len(set(numbers)) != len(numbers) or self.columns != numbers:
            raise ValueError("Columns must contain the unique area numbers in ascending order")
        if any(len(row.values) != len(numbers) for row in self.rows):
            raise ValueError("Every file row must contain one value per recognition area")
        return self


def source_key(source: Path) -> str:
    return os.path.normcase(str(source.resolve()))


def read_areas(value) -> list[RecognitionArea]:
    """Recover valid regions from settings without letting stale data break import."""
    result = {}
    for item in value if isinstance(value, list) else []:
        try:
            area = RecognitionArea.model_validate(item)
        except ValueError:
            continue
        result.setdefault(area.number, area)
    return sorted(result.values(), key=lambda area: area.number)


def matching_files(source: Path, mask: str) -> list[Path]:
    patterns = [pattern.strip() for pattern in mask.split(";") if pattern.strip()]
    if not patterns or any(any(char in pattern for char in "/\\:") for pattern in patterns):
        raise ExtractionError("Use filename masks such as *.pdf or *.png;*.jpg, without folders.")
    return sorted(
        (
            path.resolve()
            for path in source.resolve().parent.iterdir()
            if path.is_file()
            and path.suffix.lower() in IMAGE_EXTENSIONS | {".pdf"}
            and any(
                fnmatch.fnmatchcase(path.name.casefold(), pattern.casefold())
                for pattern in patterns
            )
        ),
        key=lambda path: (path.name.casefold(), path.name),
    )


def recognize_batch(
    source: str,
    sources: list[str],
    areas: list[RecognitionArea],
    mask: str,
    *,
    language: str = "en",
    layout=None,
    on_file: Callable[[str], None] | None = None,
) -> BatchResult:
    """Apply the template's relative page rectangles to each file, in stable order."""
    if not areas:
        raise ExtractionError("Add at least one recognition area before recognizing.")
    if not sources:
        raise ExtractionError("No supported files match the file mask.")
    ordered = sorted(areas, key=lambda area: area.number)
    if len({area.number for area in ordered}) != len(ordered):
        raise ExtractionError("Recognition area numbers must be unique")
    if layout is None:
        layout = PaddleLayout(lang=language)
    required_pages = {area.page for area in ordered}
    rows = []
    for filename in sources:
        if on_file is not None:
            on_file(filename)
        row = BatchRow(file=filename, values=[None] * len(ordered))
        seen = set()
        try:
            with closing(iter_pages(Path(filename))) as pages:
                for page in pages:
                    seen.add(page.number)
                    for index, area in enumerate(ordered):
                        if area.page != page.number:
                            continue
                        crop = page.image.crop(area.pixel_box(page.image.width, page.image.height))
                        try:
                            row.values[index] = layout.analyze(crop).markdown.strip()
                        finally:
                            crop.close()
                    if page.number >= max(required_pages):
                        break
            missing = sorted(required_pages - seen)
            if missing:
                row.error = "Missing template pages: " + ", ".join(map(str, missing))
        except Exception as exc:
            row.error = str(exc)
        rows.append(row)
    return BatchResult(
        source=source,
        language=language,
        file_mask=mask,
        areas=ordered,
        columns=[area.number for area in ordered],
        rows=rows,
    )


def batch_text(result: BatchResult, tr) -> str:
    """An editable Markdown table: one file per row and numbered areas per column."""

    def cell(value: str | None) -> str:
        return (
            (value or "")
            .replace("\\", "\\\\")
            .replace("|", "\\|")
            .replace("\r\n", "\n")
            .replace("\n", "\\n")
            .replace("\r", "\\n")
            .replace("\t", "\\t")
        )

    columns = [tr("file")] + [tr("area_number", number=number) for number in result.columns]
    errors = any(row.error for row in result.rows)
    if errors:
        columns.append(tr("batch_error"))
    table = [columns, ["---"] * len(columns)]
    for row in result.rows:
        values = [Path(row.file).name, *row.values]
        if errors:
            values.append(row.error)
        table.append(values)
    escaped = [[cell(value) for value in row] for row in table]
    widths = [max(len(row[index]) for row in escaped) for index in range(len(columns))]
    escaped[1] = ["-" * width for width in widths]
    return (
        "\n".join(
            "| "
            + " | ".join(value.ljust(width) for value, width in zip(row, widths, strict=True))
            + " |"
            for row in escaped
        )
        + "\n"
    )
