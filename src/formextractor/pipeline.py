"""Shared orchestration for the CLI and embedding in other Python applications."""

import logging
import re
from contextlib import closing
from pathlib import Path
from typing import Protocol

from PIL import Image

from .documents import iter_pages
from .errors import ExtractionError
from .paddle import LayoutResult, PaddleLayout
from .schema import DocumentResult, FormField, PageResult

LOGGER = logging.getLogger(__name__)


class LayoutAnalyzer(Protocol):
    def analyze(self, image: Image.Image) -> LayoutResult: ...


def extract_document(
    source: str | Path,
    *,
    pages: str | None = None,
    dpi: int = 200,
    max_pixels: int = 4_000_000,
    rotation: int = 0,
    layout: LayoutAnalyzer | None = None,
) -> DocumentResult:
    source = Path(source)
    if layout is None:
        layout = PaddleLayout()
    results = []
    with closing(
        iter_pages(source, pages=pages, dpi=dpi, max_pixels=max_pixels, rotation=rotation)
    ) as document_pages:
        for page in document_pages:
            LOGGER.info("Recognizing page %d with PaddleOCR", page.number)
            try:
                evidence = layout.analyze(page.image)
                results.append(
                    PageResult(
                        document_type=None,
                        fields=extract_explicit_fields(evidence.markdown),
                        tables=[],
                        text=evidence.markdown,
                        page=page.number,
                        image_width=page.image.width,
                        image_height=page.image.height,
                        ocr={"markdown": evidence.markdown, "raw": evidence.data},
                    )
                )
            except ExtractionError as exc:
                raise ExtractionError(f"Page {page.number}: {exc}") from exc
    return DocumentResult(source=str(source), model="PP-StructureV3", pages=results)


def extract_explicit_fields(text: str) -> list[FormField]:
    """Only split explicit label:value lines; keep all other text in page.text."""
    fields = []
    for line in text.splitlines():
        # Tables and headings remain in the original OCR output. Do not turn
        # HTML attributes, URLs, times, or Markdown tables into form fields.
        line = line.strip()
        if any(token in line for token in ("<", ">", "|", "://")) or line.startswith("#"):
            continue
        name, separator, value = line.partition(":")
        if not separator or not any(character.isalpha() for character in name):
            continue
        match = re.fullmatch(r"(\d+)[.)]\s+(.+)", name.strip())
        number, name = (match[1], match[2]) if match else (None, name.strip())
        fields.append(FormField(number=number, name=name, value=value.strip()))
    return fields
