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
from formextractor.paddle import PaddleLayout


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
