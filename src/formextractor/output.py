import os
import re
import tempfile
from datetime import date
from decimal import Decimal
from html import unescape
from html.parser import HTMLParser
from pathlib import Path

from .errors import ExtractionError
from .recognition_types import AreaKind
from .schema import DocumentResult


class _ValueTextParser(HTMLParser):
    """Keep visible text, separating HTML blocks and table cells."""

    blocks = frozenset(
        "address article aside blockquote div dl dt dd fieldset figcaption figure footer "
        "form h1 h2 h3 h4 h5 h6 header hr li main nav ol p pre section table tbody "
        "thead tfoot tr ul".split()
    )

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts = []
        self.hidden = None

    def handle_starttag(self, tag, attrs):
        if self.hidden:
            return
        if tag in ("script", "style"):
            self.hidden = tag
        elif tag in self.blocks or tag == "br":
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if self.hidden:
            if tag == self.hidden:
                self.hidden = None
        elif tag in self.blocks:
            self.parts.append("\n")
        elif tag in ("td", "th"):
            self.parts.append("\t")

    def handle_data(self, data):
        if not self.hidden:
            self.parts.append(data)


def clean_recognized_text(text: str, *, kind: AreaKind = "text") -> str:
    """Remove HTML markup and heading markers while retaining recognized values."""
    parser = _ValueTextParser()
    parser.feed(unescape(text))
    parser.close()
    text = re.sub(r"\t+(?=\n|$)", "", "".join(parser.parts))
    # OCR also emits hashes without a separating space, or spaced sequences of
    # hashes. Remove every leading marker, but keep hashes inside text such as C#.
    text = re.sub(r"^[^\S\r\n]*(?:#[^\S\r\n]*)+", "", text, flags=re.MULTILINE).strip()
    if kind == "date":
        return format_recognized_date(text)
    if kind == "number":
        return format_recognized_number(text)
    return text


def format_recognized_number(text: str) -> str:
    """Return one canonical decimal value without units or surrounding OCR noise."""
    text = text.replace("\N{MINUS SIGN}", "-")
    digits = list(re.finditer(r"[0-9]", text))
    if not digits:
        return ""
    start, end = digits[0].start(), digits[-1].end()
    prefix = text[:start].rstrip()
    # Include a sign and a decimal point before a fractional value such as -.5.
    if prefix.endswith((".", ",")):
        start = len(prefix) - 1
        prefix = prefix[:-1].rstrip()
    if prefix.endswith(("+", "-")):
        start = len(prefix) - 1
    candidate = re.sub(r"\s+", "", text[start:end]).replace(",", ".")
    if not re.fullmatch(r"[+-]?(?:[0-9]+(?:\.[0-9]+)?|\.[0-9]+)", candidate):
        raise ExtractionError("Could not recognize one valid number")
    value = Decimal(candidate)
    if value == 0:
        return "0"
    return format(value, "f").rstrip("0").rstrip(".") if "." in candidate else str(value)


def format_recognized_date(text: str) -> str:
    """Format one valid numeric date, discarding separators and surrounding text."""
    parts = re.findall(r"[0-9]+", text)
    if not parts:
        return ""
    candidates = []
    if len(parts) == 3:
        if len(parts[2]) == 4 and all(1 <= len(part) <= 2 for part in parts[:2]):
            candidates.append((parts[2], parts[1], parts[0]))
        elif len(parts[0]) == 4 and all(1 <= len(part) <= 2 for part in parts[1:]):
            candidates.append(tuple(parts))
    elif len(parts) == 1 and len(parts[0]) == 8:
        digits = parts[0]
        candidates.extend(
            ((digits[4:], digits[2:4], digits[:2]), (digits[:4], digits[4:6], digits[6:]))
        )
    valid = set()
    for year, month, day in candidates:
        try:
            valid.add(date(int(year), int(month), int(day)))
        except ValueError:
            continue
    if len(valid) != 1:
        raise ExtractionError("Could not recognize one valid date with a four-digit year")
    value = valid.pop()
    return f"{value.day:02d}.{value.month:02d}.{value.year:04d}"


def to_text(result: DocumentResult) -> str:
    lines = []
    for page in result.pages:
        lines.append(f"Page {page.page}")
        if page.text is not None:
            lines.extend((page.text, ""))
            continue
        if page.document_type:
            lines.append(page.document_type)
        for field in page.fields:
            number = f"{field.number}. " if field.number is not None else ""
            value = "[unreadable]" if field.value is None else field.value
            lines.append(f"{number}{field.name}: {value}")
        for table in page.tables:
            lines.append(f"Table: {table.name or '[untitled]'}")
            if table.columns:
                lines.append(" | ".join(table.columns))
            for row in table.rows:
                lines.append(" | ".join("[unreadable]" if cell is None else cell for cell in row))
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def validate_output_path(destination: Path, source: Path) -> None:
    if destination.resolve() == source.resolve() or (
        destination.exists() and source.exists() and destination.samefile(source)
    ):
        raise ExtractionError("Output path must not overwrite the input document")


def write_output(destination: Path, content: str, *, source: Path) -> None:
    """Only replace the result after all pages have completed successfully."""
    validate_output_path(destination, source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", newline="\n", dir=destination.parent, delete=False
        ) as handle:
            temporary = Path(handle.name)
            handle.write(content)
        os.replace(temporary, destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
