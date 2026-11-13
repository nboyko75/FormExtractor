"""Render one page at a time and normalize image orientation and transparency."""

import math
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import pymupdf
from PIL import Image, ImageOps, UnidentifiedImageError

from .errors import ExtractionError

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}


@dataclass
class DocumentPage:
    number: int
    image: Image.Image


def parse_pages(spec: str | None, count: int) -> list[int]:
    """Return sorted, unique, one-based page numbers; validate before inference."""
    if count < 1:
        raise ExtractionError("Document contains no pages")
    if spec is None:
        return list(range(1, count + 1))
    selected: set[int] = set()
    try:
        for part in spec.split(","):
            bounds = part.strip().split("-")
            if len(bounds) not in (1, 2):
                raise ValueError
            start = int(bounds[0])
            end = int(bounds[-1])
            if not 1 <= start <= end <= count:
                raise ValueError
            selected.update(range(start, end + 1))
    except ValueError as exc:
        raise ExtractionError(f"Pages must be within 1-{count}, e.g. 1,3-5") from exc
    return sorted(selected)


def _normalize(image: Image.Image, max_pixels: int, rotation: int) -> Image.Image:
    oriented = ImageOps.exif_transpose(image)
    rgba = oriented.convert("RGBA")
    background = Image.new("RGBA", rgba.size, "white")
    background.alpha_composite(rgba)
    result = background.convert("RGB")
    if rotation:
        result = result.rotate(-rotation, expand=True)
    area = result.width * result.height
    if area > max_pixels:
        scale = math.sqrt(max_pixels / area)
        result = result.resize(
            (max(1, int(result.width * scale)), max(1, int(result.height * scale))),
            Image.Resampling.LANCZOS,
        )
    return result


def iter_pages(
    source: Path,
    *,
    pages: str | None = None,
    dpi: int = 200,
    max_pixels: int = 4_000_000,
    rotation: int = 0,
) -> Iterator[DocumentPage]:
    if dpi <= 0 or max_pixels <= 0:
        raise ExtractionError("DPI and max pixels must be positive")
    if rotation not in (0, 90, 180, 270):
        raise ExtractionError("Rotation must be 0, 90, 180, or 270 degrees clockwise")
    source = Path(source)
    if not source.is_file():
        raise ExtractionError(f"Input file does not exist: {source}")
    suffix = source.suffix.lower()
    if suffix == ".pdf":
        try:
            document = pymupdf.open(source)
        except (RuntimeError, ValueError) as exc:
            raise ExtractionError(f"Cannot open PDF: {source.name}") from exc
        with document:
            if document.needs_pass:
                raise ExtractionError("Password-protected PDFs must be unlocked before extraction")
            for number in parse_pages(pages, len(document)):
                page = document[number - 1]
                area = page.rect.width * page.rect.height
                if area <= 0:
                    raise ExtractionError(f"PDF page {number} has invalid dimensions")
                scale = min(dpi / 72, math.sqrt(max_pixels / area))
                pixmap = page.get_pixmap(
                    matrix=pymupdf.Matrix(scale, scale), colorspace=pymupdf.csRGB, alpha=False
                )
                image = Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples)
                normalized = _normalize(image, max_pixels, rotation)
                image.close()
                del pixmap
                try:
                    yield DocumentPage(number, normalized)
                finally:
                    normalized.close()
    elif suffix in IMAGE_EXTENSIONS:
        try:
            document_image = Image.open(source)
        except (UnidentifiedImageError, OSError) as exc:
            raise ExtractionError(f"Cannot open image: {source.name}") from exc
        with document_image:
            for number in parse_pages(pages, getattr(document_image, "n_frames", 1)):
                document_image.seek(number - 1)
                normalized = _normalize(document_image, max_pixels, rotation)
                try:
                    yield DocumentPage(number, normalized)
                finally:
                    normalized.close()
    else:
        raise ExtractionError("Supported inputs: PDF, JPG, PNG, TIFF, BMP, and WebP")
