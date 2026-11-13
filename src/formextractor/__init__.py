"""Extract fields and tables from scanned forms using local models."""

from .pipeline import extract_document
from .schema import DocumentResult

__all__ = ["DocumentResult", "extract_document"]
