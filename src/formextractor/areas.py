"""Numbered, page-specific recognition templates and tabular batch results."""

import fnmatch
import math
import os
from collections.abc import Callable
from contextlib import closing
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from .area_matching import AreaTemplate, PageMatcher
from .documents import IMAGE_EXTENSIONS, iter_pages
from .errors import ExtractionError
from .paddle import PaddleLayout, PaddleTextDetector
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
    text_detector=None,
    on_file: Callable[[str], None] | None = None,
    on_row: Callable[[BatchRow], None] | None = None,
) -> BatchResult:
    """Locate, orient, and recognize each template area, publishing partial rows."""
    if not areas:
        raise ExtractionError("Add at least one recognition area before recognizing.")
    if not sources:
        raise ExtractionError("No supported files match the file mask.")
    ordered = sorted(areas, key=lambda area: area.number)
    if len({area.number for area in ordered}) != len(ordered):
        raise ExtractionError("Recognition area numbers must be unique")
    if layout is None:
        layout = PaddleLayout(lang=language)
    if text_detector is None:
        text_detector = PaddleTextDetector()
    required_pages = {area.page for area in ordered}
    templates = {}
    template_errors = {}
    try:
        with closing(iter_pages(Path(source))) as pages:
            for page in pages:
                text_regions = (
                    text_detector.detect(page.image) if page.number in required_pages else ()
                )
                for area in ordered:
                    if area.page == page.number:
                        templates[area.number] = AreaTemplate(
                            page.image, area, text_regions=text_regions
                        )
                if page.number >= max(required_pages):
                    break
    except Exception as exc:
        template_errors = {area.number: str(exc) for area in ordered}
    rows = []
    for filename in sources:
        if on_file is not None:
            on_file(filename)
        row = BatchRow(file=filename, values=[None] * len(ordered))
        errors = []
        seen = set()
        try:
            with closing(iter_pages(Path(filename))) as pages:
                for page in pages:
                    seen.add(page.number)
                    if page.number not in required_pages:
                        continue
                    matcher = PageMatcher(page.image, text_regions=text_detector.detect(page.image))
                    for index, area in enumerate(ordered):
                        if area.page != page.number:
                            continue
                        try:
                            if area.number in template_errors:
                                raise ExtractionError(template_errors[area.number])
                            template = templates.get(area.number)
                            if template is None:
                                raise ExtractionError(f"Missing template page: {area.page}")
                            with matcher.crop(template) as crop:
                                row.values[index] = layout.analyze(crop).markdown.strip()
                        except Exception as exc:
                            errors.append(f"Area {area.number}: {exc}")
                            row.error = "; ".join(errors)
                        if on_row is not None:
                            on_row(row.model_copy(deep=True))
                    if page.number >= max(required_pages):
                        break
            missing = sorted(required_pages - seen)
            if missing:
                errors.append("Missing template pages: " + ", ".join(map(str, missing)))
        except Exception as exc:
            errors.append(str(exc))
        row.error = "; ".join(errors) or None
        rows.append(row)
        if on_row is not None:
            on_row(row.model_copy(deep=True))
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
