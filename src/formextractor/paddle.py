"""Local PP-StructureV3 recognition, loaded only when recognition starts."""

import os
import re
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

    def analyze_area(
        self, image: Image.Image, *, kind: AreaKind = "text",
        original_image: Image.Image | None = None,
    ) -> LayoutResult:
        """Recover missed single-line text fields without changing full-page OCR."""
        result = (
            self._analyze_value(image, kind, fallback_image=original_image)
            if kind == "date" and original_image is not None
            else self.analyze(image, kind=kind)
        )
        if kind != "text" or result.area_value():
            return result
        from .field_images import text_line_variants

        variants = text_line_variants(image)
        if not variants:
            return result
        try:
            # Reuse PP-Structure's loaded recognizer and selected language. This
            # bypasses layout/text detection, keeping its full alphabet intact.
            recognizer = self._pipeline.paddlex_pipeline.general_ocr_pipeline.text_rec_model
            readings = []
            for name, pixels in variants:
                predictions = list(recognizer(input=pixels))
                if len(predictions) != 1:
                    raise ExtractionError("Unexpected text line recognition output format")
                text = predictions[0].get("rec_text")
                score = predictions[0].get("rec_score")
                if not isinstance(text, str):
                    raise ExtractionError("Unexpected text line recognition output format")
                value = clean_recognized_text(text)
                if not value or not isinstance(score, (int, float)) or not 0.8 <= score <= 1:
                    return result
                readings.append({"preprocessing": name, "text": value, "score": float(score)})
            if readings[0]["text"].casefold() != readings[1]["text"].casefold():
                return result
            return LayoutResult(
                markdown=readings[0]["text"],
                data={
                    **result.data,
                    "layout_markdown": result.markdown,
                    "text_line_readings": readings,
                },
            )
        except ExtractionError:
            raise
        except Exception as exc:
            raise ExtractionError(
                f"Text field recognition failed: {_exception_details(exc)}"
            ) from exc

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

    def _analyze_value(
        self, image: Image.Image, kind: AreaKind, *, fallback_image: Image.Image | None = None,
    ) -> LayoutResult:
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
            data = {"rec_text": text, "rec_score": results[0].get("rec_score")}
            try:
                value = clean_recognized_text(text, kind=kind)
                if kind == "date" and not value:
                    raise ExtractionError("No valid date text was recognized")
            except ExtractionError as date_error:
                if kind != "date":
                    raise
                # A year suffix can become a digit; slashes read as ones can also
                # resemble a valid compact date. Retry with the restored decoder
                # and require an unambiguous reading instead of replacing digits.
                retry = list(self._value_recognizer.predict(input=bgr))
                if len(retry) != 1 or not isinstance(retry[0].get("rec_text"), str):
                    raise ExtractionError(
                        "Unexpected date recognition output format"
                    ) from date_error
                text = retry[0]["rec_text"].strip()
                data = {
                    "rec_text": text,
                    "rec_score": retry[0].get("rec_score"),
                    "restricted_rec_text": data["rec_text"],
                }
                try:
                    value = clean_recognized_text(text, kind="date")
                    if not value and any(char.isdigit() for char in data["restricted_rec_text"]):
                        raise date_error
                except ExtractionError:
                    recovered = self._recover_date(image, data, original_image=fallback_image)
                    if recovered is not None:
                        return recovered
                    raise
                if not value:
                    recovered = self._recover_date(image, data, original_image=fallback_image)
                    if recovered is not None:
                        return recovered
            return LayoutResult(
                markdown=value,
                data=data,
            )
        except ExtractionError:
            raise
        except Exception as exc:
            raise ExtractionError(f"PaddleOCR failed: {_exception_details(exc)}") from exc

    def _recover_date(
        self, image: Image.Image, data: dict, *, original_image: Image.Image | None = None,
    ) -> LayoutResult | None:
        """Retry intact handwriting when border cleanup has damaged a date stroke."""
        if original_image is not None:
            import numpy as np

            pixels = np.asarray(original_image.convert("RGB"))[:, :, ::-1].copy()
            predictor = self._value_recognizer.paddlex_predictor
            original = predictor.post_op
            try:
                for decoder in (_CharacterLimitedDecoder(original, AREA_CHARACTERS["date"]),
                                original):
                    predictor.post_op = decoder
                    results = list(self._value_recognizer.predict(input=pixels))
                    if len(results) != 1 or not isinstance(results[0].get("rec_text"), str):
                        raise ExtractionError("Unexpected date recognition output format")
                    text, score = results[0]["rec_text"].strip(), results[0].get("rec_score")
                    if not isinstance(score, (int, float)) or not 0.8 <= score <= 1:
                        continue
                    try:
                        value = clean_recognized_text(text, kind="date")
                    except ExtractionError:
                        continue
                    if value:
                        return LayoutResult(
                            markdown=value,
                            data={
                                **data,
                                "borderless_rec_text": data["rec_text"],
                                "rec_text": text,
                                "rec_score": score,
                                "date_original_reading": True,
                            },
                        )
            finally:
                predictor.post_op = original
        return self._recover_slash_date(
            image if original_image is None else original_image, data
        )

    def _recover_slash_date(self, image: Image.Image, data: dict) -> LayoutResult | None:
        """Require two confident, agreeing readings of geometrically separated date parts."""
        from .date_images import date_part_variants

        variants = date_part_variants(image)
        if len(variants) != 2:
            return None
        predictor = self._value_recognizer.paddlex_predictor
        original = predictor.post_op
        attempts = []
        predictor.post_op = _CharacterLimitedDecoder(original, "0123456789")
        try:
            for name, parts in variants:
                readings, scores = [], []
                for index, pixels in enumerate(parts):
                    results = list(self._value_recognizer.predict(input=pixels))
                    if len(results) != 1:
                        return None
                    text, score = results[0].get("rec_text"), results[0].get("rec_score")
                    pattern = r"[0-9]{1,2}" if index < 2 else r"(?:[0-9]{2}|[0-9]{4})"
                    if not isinstance(text, str) or not re.fullmatch(pattern, text.strip()):
                        return None
                    if not isinstance(score, (int, float)) or not 0.8 <= score <= 1:
                        return None
                    readings.append(text.strip())
                    scores.append(float(score))
                text = "/".join(readings)
                try:
                    value = clean_recognized_text(text, kind="date")
                except ExtractionError:
                    return None
                attempts.append(
                    {"preprocessing": name, "text": text, "score": min(scores), "value": value}
                )
        finally:
            predictor.post_op = original
        if attempts[0]["value"] != attempts[1]["value"]:
            return None
        return LayoutResult(
            markdown=attempts[0]["value"],
            data={
                **data,
                "unrestricted_rec_text": data["rec_text"],
                "rec_text": attempts[0]["text"],
                "rec_score": min(attempt["score"] for attempt in attempts),
                "date_part_readings": attempts,
            },
        )
