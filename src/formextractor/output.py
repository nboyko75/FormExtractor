import os
import tempfile
from pathlib import Path

from .errors import ExtractionError
from .schema import DocumentResult


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
