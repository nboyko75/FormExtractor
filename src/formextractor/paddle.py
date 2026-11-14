"""Local PP-StructureV3 recognition, loaded only when recognition starts."""

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image

from .errors import ExtractionError


@dataclass
class LayoutResult:
    markdown: str
    data: dict[str, Any]


def _windows_short_path(path: str) -> str | None:
    """Return the Windows alias without resolving it back to a Unicode path."""
    import ctypes
    from ctypes import wintypes

    get_short_path = ctypes.WinDLL("kernel32", use_last_error=True).GetShortPathNameW
    get_short_path.argtypes = [wintypes.LPCWSTR, wintypes.LPWSTR, wintypes.DWORD]
    get_short_path.restype = wintypes.DWORD
    size = get_short_path(path, None, 0)
    if not size:
        return None
    buffer = ctypes.create_unicode_buffer(size)
    length = get_short_path(path, buffer, size)
    return buffer.value if 0 < length < size else None


def _prepare_model_cache() -> None:
    """Keep Windows native model I/O away from non-ASCII cache paths."""
    if sys.platform != "win32":
        return
    cache = Path(os.environ.get("PADDLE_PDX_CACHE_HOME") or Path.home() / ".paddlex")
    cache = cache.absolute()
    if str(cache).isascii():
        return

    # On first launch the cache may not exist. Alias its existing parent and
    # keep the remaining components so PaddleX can create them as usual.
    parent = cache
    suffix = []
    while not parent.exists() and parent != parent.parent:
        suffix.append(parent.name)
        parent = parent.parent
    alias = _windows_short_path(str(parent))
    if alias:
        candidate = str(Path(alias).joinpath(*reversed(suffix)))
        if candidate.isascii():
            # PaddleX captures this setting at import time. Do not resolve the
            # alias: it refers to the same files, without moving or copying them.
            os.environ["PADDLE_PDX_CACHE_HOME"] = candidate
            return
    raise ExtractionError(
        f"PaddleOCR cannot load models from the non-ASCII Windows cache path: {cache}\n"
        "Windows did not provide an ASCII short path. Set PADDLE_PDX_CACHE_HOME "
        "to a writable folder whose full path uses only ASCII characters "
        "(for example D:\\FormExtractorModels), then restart the application. "
        "Models will be downloaded there on first recognition."
    )


def _ensure_standard_streams() -> None:
    """Give OCR downloaders writable streams in pythonw/windowed executables."""
    # PaddleX writes download/extraction progress directly to stderr. Windowed
    # processes (including spawned recognition workers) can have neither stream.
    # Keep replacements alive: imported libraries may retain them for later use.
    # Preserve real streams, including the CLI's redirected stdout.
    for name in ("stdout", "stderr"):
        if getattr(sys, name) is None:
            setattr(sys, name, open(os.devnull, "w", encoding="utf-8"))


def _exception_details(exc: BaseException) -> str:
    """Keep the cause that PaddleOCR hides behind its generic runtime errors."""
    details = []
    seen = set()
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        details.append(f"{type(exc).__name__}: {exc}")
        exc = exc.__cause__ or (None if exc.__suppress_context__ else exc.__context__)
    return "\nCaused by: ".join(details)


class PaddleTextDetector:
    """Find text polygons for exclusion from image matching, without reading text."""

    def __init__(self, *, device: str = "cpu") -> None:
        self.device = device
        self._detector = None

    def detect(self, image: Image.Image):
        _ensure_standard_streams()
        _prepare_model_cache()
        try:
            import numpy as np
            from paddleocr import TextDetection

            if self._detector is None:
                self._detector = TextDetection(device=self.device)
            bgr = np.asarray(image.convert("RGB"))[:, :, ::-1].copy()
            results = list(self._detector.predict(input=bgr))
            if len(results) != 1 or "dt_polys" not in results[0]:
                raise ExtractionError("Unexpected text detection output format")
            return results[0]["dt_polys"]
        except ExtractionError:
            raise
        except Exception as exc:
            raise ExtractionError(f"Text detection failed: {_exception_details(exc)}") from exc


class PaddleLayout:
    def __init__(self, *, device: str = "cpu", lang: str = "en") -> None:
        self.device = device
        self.lang = lang
        self._pipeline = None

    def analyze(self, image: Image.Image) -> LayoutResult:
        _ensure_standard_streams()
        _prepare_model_cache()
        try:
            import numpy as np
            from paddleocr import PPStructureV3
        except ImportError as exc:
            raise ExtractionError(
                "PaddleOCR could not be imported. Run: python -m pip install -e .\n"
                + _exception_details(exc)
            ) from exc
        try:
            if self._pipeline is None:
                self._pipeline = PPStructureV3(
                    device=self.device,
                    lang=self.lang,
                    use_doc_orientation_classify=False,
                    use_doc_unwarping=False,
                    use_textline_orientation=False,
                    use_table_recognition=True,
                    use_formula_recognition=False,
                    use_chart_recognition=False,
                    use_seal_recognition=False,
                )
            # Paddle's ndarray interface uses BGR. Keep geometry unchanged so that
            # OCR coordinates refer to exactly the processed page image.
            bgr = np.asarray(image.convert("RGB"))[:, :, ::-1].copy()
            results = list(self._pipeline.predict(input=bgr))
            if len(results) != 1:
                raise ExtractionError("PaddleOCR did not return exactly one page result")
            result = results[0]
            markdown = result.markdown["markdown_texts"]
            data = result.json
            if not isinstance(markdown, str) or not isinstance(data, dict):
                raise ExtractionError("Unexpected PP-StructureV3 output format")
            return LayoutResult(markdown=markdown, data=data)
        except ExtractionError:
            raise
        except Exception as exc:
            raise ExtractionError(f"PaddleOCR failed: {_exception_details(exc)}") from exc
