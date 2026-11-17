"""Local PP-StructureV3 recognition, loaded only when recognition starts."""

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image

from .errors import ExtractionError
from .output import clean_recognized_text
from .recognition_types import AREA_CHARACTERS, AreaKind


@dataclass
class LayoutResult:
    markdown: str
    data: dict[str, Any]

    def area_value(self, *, kind: AreaKind = "text") -> str:
        """Recover text hidden by layout formatting in an otherwise empty area."""
        value = clean_recognized_text(self.markdown, kind=kind)
        if value or kind != "text":
            return value
        # PP-Structure can label a small text crop as an image. Its Markdown
        # then contains only an image placeholder, although OCR read the value.
        # Reuse those readings without rerunning OCR or changing its alphabet.
        result = self.data.get("res", self.data)
        ocr = result.get("overall_ocr_res") if isinstance(result, dict) else None
        texts = ocr.get("rec_texts") if isinstance(ocr, dict) else None
        if isinstance(texts, list) and all(isinstance(text, str) for text in texts):
            return clean_recognized_text("\n".join(texts))
        return value


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


class _CharacterLimitedDecoder:
    """Limit CTC choices before decoding without changing the model's label indices."""

    def __init__(self, decoder, characters: str) -> None:
        import numpy as np

        self.decoder = decoder
        alphabet = decoder.character
        if not set("0123456789").issubset(alphabet):
            raise ExtractionError("The recognition model does not support all digits")
        allowed = set(characters)
        self.allowed = np.array([char in allowed for char in alphabet], dtype=bool)
        self.allowed[decoder.get_ignored_tokens()] = True  # Keep CTC blank/repeat separation.

    def __call__(self, predictions, **kwargs):
        import numpy as np

        scores = np.array(predictions[0], copy=True)
        if scores.ndim != 3 or scores.shape[-1] != len(self.allowed):
            raise ExtractionError("Unexpected character recognition output shape")
        scores[..., ~self.allowed] = -np.inf
        return self.decoder([scores, *predictions[1:]], **kwargs)


class PaddleLayout:
    def __init__(self, *, device: str = "cpu", lang: str = "en") -> None:
        self.device = device
        self.lang = lang
        self._pipeline = None
        self._value_recognizer = None

    def analyze(self, image: Image.Image, *, kind: AreaKind = "text") -> LayoutResult:
        if kind != "text":
            return self._analyze_value(image, kind)
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

    def _analyze_value(self, image: Image.Image, kind: AreaKind) -> LayoutResult:
        """Read a selected number/date directly with a restricted character decoder."""
        if kind not in AREA_CHARACTERS:
            raise ExtractionError(f"Unknown recognition area type: {kind}")
        _ensure_standard_streams()
        _prepare_model_cache()
        try:
            import numpy as np
            from paddleocr import TextRecognition

            if self._value_recognizer is None:
                self._value_recognizer = TextRecognition(
                    model_name="PP-OCRv5_server_rec", device=self.device
                )
            predictor = self._value_recognizer.paddlex_predictor
            # PaddleOCR 3.3/PaddleX exposes the CTC decoder on its predictor.
            # Never replace its dictionary: model output indices must stay fixed.
            if predictor.config.get("PostProcess", {}).get("name") != "CTCLabelDecode":
                raise ExtractionError("This recognition model cannot restrict characters")
            original = predictor.post_op
            predictor.post_op = _CharacterLimitedDecoder(original, AREA_CHARACTERS[kind])
            try:
                bgr = np.asarray(image.convert("RGB"))[:, :, ::-1].copy()
                results = list(self._value_recognizer.predict(input=bgr))
            finally:
                # A date restriction must not leak into the next number or text area.
                predictor.post_op = original
            if len(results) != 1 or not isinstance(results[0].get("rec_text"), str):
                raise ExtractionError("Unexpected character recognition output format")
            text = results[0]["rec_text"].strip()
            if any(char not in AREA_CHARACTERS[kind] for char in text):
                raise ExtractionError("Recognition returned characters outside the selected type")
            return LayoutResult(
                markdown=clean_recognized_text(text, kind=kind),
                data={"rec_text": text, "rec_score": results[0].get("rec_score")},
            )
        except ExtractionError:
            raise
        except Exception as exc:
            raise ExtractionError(f"PaddleOCR failed: {_exception_details(exc)}") from exc
