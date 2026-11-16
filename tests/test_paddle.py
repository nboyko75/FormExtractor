import builtins
import io
import os
import sys
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from formextractor import paddle as paddle_adapter
from formextractor.errors import ExtractionError
from formextractor.paddle import PaddleLayout, PaddleTextDetector
from formextractor.recognition_types import AREA_CHARACTERS


class FakeCTCDecoder:
    character = ["blank", *"0123456789.,+- /O"]

    def get_ignored_tokens(self):
        return [0]

    def __call__(self, predictions, **kwargs):
        indices = predictions[0].argmax(axis=-1)[0]
        text = "".join(
            self.character[index]
            for position, index in enumerate(indices)
            if index != 0 and (position == 0 or index != indices[position - 1])
        )
        return [text], [0.9]


def character_scores(text):
    alphabet = FakeCTCDecoder.character
    scores = np.zeros((1, len(text) * 2, len(alphabet)), dtype=np.float32)
    scores[:, 1::2, 0] = 1  # Blank separates repeated characters, including leading zeroes.
    for position, char in enumerate(text):
        scores[0, position * 2, alphabet.index(char)] = 0.4
        scores[0, position * 2, alphabet.index("O")] = 0.6
    return scores


@pytest.mark.parametrize(
    "kind,value",
    [("number", "-001,5"), ("number", "+12.50"), ("date", "21/03/2026"), ("date", "2026-03-21")],
)
def test_restricted_decoding_chooses_allowed_characters_before_ctc(kind, value):
    scores = character_scores(value)
    original = scores.copy()
    decoder = paddle_adapter._CharacterLimitedDecoder(FakeCTCDecoder(), AREA_CHARACTERS[kind])
    assert FakeCTCDecoder()([scores])[0] == ["O" * len(value)]
    assert decoder([scores])[0] == [value]
    np.testing.assert_array_equal(scores, original)


def test_number_and_date_have_different_decoding_alphabets():
    scores = character_scores("+1,2/3")
    # Let CTC blank win whenever punctuation is excluded for the selected type.
    scores[:, ::2, 0] = 0.3
    number = paddle_adapter._CharacterLimitedDecoder(FakeCTCDecoder(), AREA_CHARACTERS["number"])
    date = paddle_adapter._CharacterLimitedDecoder(FakeCTCDecoder(), AREA_CHARACTERS["date"])
    assert number([scores])[0] == ["+1,23"]
    assert date([scores])[0] == ["12/3"]
    blank = np.zeros_like(scores)
    blank[:, :, 0] = 1
    assert number([blank])[0] == [""]


def test_typed_recognition_is_lazy_reuses_model_and_restores_decoder_after_failure(monkeypatch):
    instances = []

    class Recognizer:
        def __init__(self, **kwargs):
            assert kwargs == {"model_name": "PP-OCRv5_server_rec", "device": "cpu"}
            self.decoder = FakeCTCDecoder()
            self.paddlex_predictor = SimpleNamespace(
                config={"PostProcess": {"name": "CTCLabelDecode"}}, post_op=self.decoder
            )
            self.value = "-001,5"
            self.fail = False
            instances.append(self)

        def predict(self, input):
            np.testing.assert_array_equal(input[0, 0], [0, 0, 255])
            if self.fail:
                raise RuntimeError("Recognition failed")
            texts, scores = self.paddlex_predictor.post_op([character_scores(self.value)])
            return [{"rec_text": texts[0], "rec_score": scores[0]}]

    monkeypatch.setitem(sys.modules, "paddleocr", SimpleNamespace(TextRecognition=Recognizer))
    adapter = PaddleLayout(lang="uk")
    assert not instances
    with Image.new("RGB", (40, 20), "red") as image:
        assert adapter.analyze(image, kind="number").markdown == "-001,5"
        recognizer = instances[0]
        assert recognizer.paddlex_predictor.post_op is recognizer.decoder
        recognizer.value = "21.03.2026"
        assert adapter.analyze(image, kind="date").markdown == "21.03.2026"
        assert recognizer.paddlex_predictor.post_op is recognizer.decoder
        recognizer.fail = True
        with pytest.raises(ExtractionError, match="Recognition failed"):
            adapter.analyze(image, kind="date")
        assert recognizer.paddlex_predictor.post_op is recognizer.decoder
        recognizer.fail = False
        recognizer.value = "+12.5"
        assert adapter.analyze(image, kind="number").markdown == "+12.5"
        # Text recognition still follows the original full-alphabet path.
        adapter._pipeline = SimpleNamespace(
            predict=lambda input: [
                SimpleNamespace(markdown={"markdown_texts": "Model O1"}, json={})
            ]
        )
        monkeypatch.setitem(
            sys.modules,
            "paddleocr",
            SimpleNamespace(TextRecognition=Recognizer, PPStructureV3=object),
        )
        assert adapter.analyze(image).markdown == "Model O1"
    assert len(instances) == 1


def test_unicode_cache_uses_existing_ascii_alias_before_import(monkeypatch, tmp_path):
    cache = tmp_path / "Користувач ПК" / ".paddlex"
    cache.mkdir(parents=True)
    alias = tmp_path / "USER~1" / "PADDLE~1"
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setenv("PADDLE_PDX_CACHE_HOME", str(cache))
    paths = []

    def short_path(path):
        paths.append(path)
        return str(alias)

    monkeypatch.setattr(paddle_adapter, "_windows_short_path", short_path)
    original_import = builtins.__import__

    def check_import(name, *args, **kwargs):
        if name == "paddleocr":
            assert os.environ["PADDLE_PDX_CACHE_HOME"] == str(alias)
            raise ImportError("Stop after checking cache configuration")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", check_import)
    with pytest.raises(ExtractionError, match="Stop after checking"):
        PaddleLayout().analyze(Image.new("RGB", (20, 20)))
    assert paths == [str(cache)]
    assert cache.is_dir()


def test_first_launch_aliases_home_before_cache_exists(monkeypatch, tmp_path):
    home = tmp_path / "Користувач ПК"
    home.mkdir()
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.delenv("PADDLE_PDX_CACHE_HOME", raising=False)
    monkeypatch.setattr(paddle_adapter.Path, "home", lambda: home)
    alias = tmp_path / "USER~1"
    paths = []

    def short_path(path):
        paths.append(path)
        return str(alias)

    monkeypatch.setattr(paddle_adapter, "_windows_short_path", short_path)
    paddle_adapter._prepare_model_cache()
    assert os.environ["PADDLE_PDX_CACHE_HOME"] == str(alias / ".paddlex")
    assert paths == [str(home)]


@pytest.mark.parametrize("alias", [None, "C:\\Users\\Користувач ПК"])
def test_unavailable_ascii_alias_has_actionable_error(monkeypatch, tmp_path, alias):
    cache = tmp_path / "Користувач ПК"
    cache.mkdir()
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setenv("PADDLE_PDX_CACHE_HOME", str(cache))
    monkeypatch.setattr(paddle_adapter, "_windows_short_path", lambda path: alias)
    with pytest.raises(ExtractionError, match="Set PADDLE_PDX_CACHE_HOME"):
        paddle_adapter._prepare_model_cache()
    assert os.environ["PADDLE_PDX_CACHE_HOME"] == str(cache)


@pytest.mark.parametrize("platform,folder", [("win32", "models"), ("linux", "моделі")])
def test_compatible_cache_setting_is_preserved(monkeypatch, tmp_path, platform, folder):
    cache = str(tmp_path / folder)
    monkeypatch.setattr(sys, "platform", platform)
    monkeypatch.setenv("PADDLE_PDX_CACHE_HOME", cache)

    def unexpected_alias(path):
        pytest.fail("Compatible paths must not need a Windows alias")

    monkeypatch.setattr(paddle_adapter, "_windows_short_path", unexpected_alias)
    paddle_adapter._prepare_model_cache()
    assert os.environ["PADDLE_PDX_CACHE_HOME"] == cache


def test_paddle_bgr_geometry_output_and_lazy_initialization(monkeypatch):
    calls = []
    raw = {
        "res": {
            "parsing_res_list": [
                {"block_label": "text", "block_content": "Name: Alex", "block_bbox": [0, 1, 5, 6]},
            ]
        }
    }

    class Pipeline:
        def __init__(self, **kwargs):
            calls.append(kwargs)

        def predict(self, input):
            assert input.shape == (20, 40, 3)
            np.testing.assert_array_equal(input[0, 0], [0, 0, 255])
            yield SimpleNamespace(markdown={"markdown_texts": "Name: Alex"}, json=raw)

    monkeypatch.setitem(sys.modules, "paddleocr", SimpleNamespace(PPStructureV3=Pipeline))
    adapter = PaddleLayout(device="cpu", lang="en")
    image = Image.new("RGB", (40, 20), "red")
    first = adapter.analyze(image)
    adapter.analyze(image)
    assert first.markdown == "Name: Alex"
    assert first.data == raw
    assert len(calls) == 1
    assert calls[0]["use_table_recognition"] is True
    assert calls[0]["use_doc_unwarping"] is False


def test_missing_paddle_dependency_has_install_instructions(monkeypatch):
    monkeypatch.setitem(sys.modules, "paddleocr", None)
    with pytest.raises(ExtractionError, match="python -m pip install -e"):
        PaddleLayout().analyze(Image.new("RGB", (20, 20)))


def test_text_detector_returns_polygons_without_recognizing_text(monkeypatch):
    calls = []
    polygons = [np.array([[1, 2], [30, 2], [30, 15], [1, 15]])]

    class Detector:
        def __init__(self, **kwargs):
            calls.append(kwargs)

        def predict(self, input):
            assert input.shape == (20, 40, 3)
            np.testing.assert_array_equal(input[0, 0], [0, 0, 255])
            yield {"dt_polys": polygons}

    monkeypatch.setitem(sys.modules, "paddleocr", SimpleNamespace(TextDetection=Detector))
    adapter = PaddleTextDetector(device="cpu")
    with Image.new("RGB", (40, 20), "red") as image:
        assert adapter.detect(image) is polygons
        assert adapter.detect(image) is polygons
    assert calls == [{"device": "cpu"}]


def test_text_detector_errors_are_reported_instead_of_matching_text(monkeypatch):
    class Detector:
        def __init__(self, **kwargs):
            raise RuntimeError("Detector unavailable")

    monkeypatch.setitem(sys.modules, "paddleocr", SimpleNamespace(TextDetection=Detector))
    with Image.new("RGB", (20, 20)) as image:
        with pytest.raises(ExtractionError, match="Text detection failed.*Detector unavailable"):
            PaddleTextDetector().detect(image)


def test_pipeline_creation_preserves_underlying_dependency_error(monkeypatch):
    class Pipeline:
        def __init__(self, **kwargs):
            try:
                raise ImportError("Missing distribution metadata: opencv-contrib-python")
            except ImportError as exc:
                raise RuntimeError("A dependency error occurred during pipeline creation.") from exc

    monkeypatch.setitem(sys.modules, "paddleocr", SimpleNamespace(PPStructureV3=Pipeline))
    with pytest.raises(ExtractionError) as error:
        PaddleLayout().analyze(Image.new("RGB", (20, 20)))
    assert "RuntimeError: A dependency error" in str(error.value)
    assert "Caused by: ImportError: Missing distribution metadata: opencv-contrib-python" in str(
        error.value
    )


def test_failed_initialization_can_be_retried(monkeypatch):
    calls = []

    class Pipeline:
        def __init__(self, **kwargs):
            calls.append(kwargs)
            if len(calls) == 1:
                raise RuntimeError("Temporary initialization failure")

        def predict(self, input):
            yield SimpleNamespace(markdown={"markdown_texts": "Recovered"}, json={})

    monkeypatch.setitem(sys.modules, "paddleocr", SimpleNamespace(PPStructureV3=Pipeline))
    adapter = PaddleLayout()
    image = Image.new("RGB", (20, 20))
    with pytest.raises(ExtractionError, match="Temporary initialization failure"):
        adapter.analyze(image)
    assert adapter.analyze(image).markdown == "Recovered"


@pytest.mark.parametrize("missing", [("stdout",), ("stderr",), ("stdout", "stderr"), ()])
def test_console_free_recognition_has_streams_before_import(monkeypatch, missing):
    original_import = builtins.__import__
    existing = {name: io.StringIO() for name in ("stdout", "stderr")}
    streams = {}

    def write_progress(stage):
        for name in ("stdout", "stderr"):
            stream = getattr(sys, name)
            stream.write(f"{stage}\n")
            stream.flush()
            streams[name] = stream

    class Pipeline:
        def __init__(self, **kwargs):
            write_progress("Downloading model")

        def predict(self, input):
            write_progress("Recognizing page")
            yield SimpleNamespace(markdown={"markdown_texts": "Recognized"}, json={})

    def import_with_progress(name, *args, **kwargs):
        if name == "paddleocr":
            write_progress("Importing PaddleOCR")
            return SimpleNamespace(PPStructureV3=Pipeline)
        return original_import(name, *args, **kwargs)

    try:
        with monkeypatch.context() as patch:
            for name, stream in existing.items():
                patch.setattr(sys, name, None if name in missing else stream)
            patch.setattr(builtins, "__import__", import_with_progress)
            adapter = PaddleLayout()
            image = Image.new("RGB", (20, 20))
            assert adapter.analyze(image).markdown == "Recognized"
            first_streams = streams.copy()
            assert adapter.analyze(image).markdown == "Recognized"
            for name, stream in streams.items():
                assert stream is first_streams[name]
                assert not stream.closed
                if name not in missing:
                    assert stream is existing[name]
                    assert "Recognizing page" in stream.getvalue()
    finally:
        for name in missing:
            if name in streams:
                streams[name].close()
        for stream in existing.values():
            stream.close()
