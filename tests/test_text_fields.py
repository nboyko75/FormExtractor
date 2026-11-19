import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image, ImageDraw

from formextractor.field_images import text_line_variants
from formextractor.paddle import LayoutResult, PaddleLayout

SAMPLES = Path(__file__).parent / "data" / "text_fields"


@pytest.mark.parametrize("filename", ["f0065_00.png", "f0251_00.png"])
def test_boxed_text_preprocessing_keeps_source_and_adds_white_margin(filename):
    with Image.open(SAMPLES / filename) as image:
        before = image.tobytes()
        variants = text_line_variants(image)
        assert len(variants) == 2
        for _, pixels in variants:
            assert pixels.shape[2] == 3
            assert np.any(pixels < 128)
            assert np.all(pixels[:8] == 255)
            assert np.all(pixels[-8:] == 255)
        assert image.tobytes() == before


@pytest.mark.parametrize("content", ["white", "black", "box", "multiline", "page"])
def test_direct_line_retry_rejects_blank_rules_and_multiple_lines(content):
    with Image.new("RGB", (300, 100) if content != "page" else (300, 500),
                   "black" if content == "black" else "white") as image:
        draw = ImageDraw.Draw(image)
        if content in {"box", "multiline"}:
            draw.rectangle((2, 2, 297, 97), outline="black", width=2)
        if content == "multiline":
            draw.text((20, 20), "First line", fill="black")
            draw.text((20, 65), "Second line", fill="black")
        assert text_line_variants(image) == []


def adapter_with_results(monkeypatch, readings, *, markdown="<img src='field.png'>", texts=None):
    calls = []
    results = iter(readings)
    raw = {"res": {"overall_ocr_res": {"rec_texts": texts or []}}}

    def recognize(input):
        calls.append(input)
        text, score = next(results)
        return [{"rec_text": text, "rec_score": score}]

    pipeline = SimpleNamespace(
        predict=lambda input: [SimpleNamespace(markdown={"markdown_texts": markdown}, json=raw)],
        paddlex_pipeline=SimpleNamespace(
            general_ocr_pipeline=SimpleNamespace(text_rec_model=recognize)
        ),
    )
    monkeypatch.setitem(
        sys.modules, "paddleocr", SimpleNamespace(PPStructureV3=lambda **kw: pipeline)
    )
    return PaddleLayout(lang="uk"), calls, raw


def test_empty_text_area_uses_loaded_language_recognizer_and_retains_layout_evidence(monkeypatch):
    adapter, calls, raw = adapter_with_results(monkeypatch, [("Дубно", .94), ("ДУБНО", .95)])
    with Image.open(SAMPLES / "f0065_00.png") as image:
        result = adapter.analyze_area(image)
    assert result.area_value() == "Дубно"
    assert len(calls) == 2
    assert result.data["res"] == raw["res"]
    assert result.data["layout_markdown"] == "<img src='field.png'>"
    assert len(result.data["text_line_readings"]) == 2
    assert "text_line_readings" not in raw


@pytest.mark.parametrize("readings", [
    [("Dundee", .95), ("Dunbee", .95)],
    [("Dundee", .79)],
    [("Dundee", float("nan"))],
    [("Dundee", None)],
    [("", .99)],
    [("Dundee", .95), ("Dundee", .79)],
])
def test_uncertain_or_disagreeing_readings_are_not_returned(monkeypatch, readings):
    adapter, _, raw = adapter_with_results(monkeypatch, readings)
    with Image.open(SAMPLES / "f0065_00.png") as image:
        result = adapter.analyze_area(image)
    assert result.area_value() == ""
    assert result.data == raw


@pytest.mark.parametrize("markdown,texts,expected", [
    ("Already read", [], "Already read"),
    ("<img src='field.png'>", ["Dundee"], "Dundee"),
])
def test_successful_text_readings_do_not_trigger_retry(monkeypatch, markdown, texts, expected):
    adapter, calls, _ = adapter_with_results(monkeypatch, [], markdown=markdown, texts=texts)
    with Image.open(SAMPLES / "f0065_00.png") as image:
        assert adapter.analyze_area(image).area_value() == expected
    assert calls == []


def test_full_page_api_does_not_trigger_field_retry(monkeypatch):
    adapter, calls, _ = adapter_with_results(monkeypatch, [])
    with Image.open(SAMPLES / "f0065_00.png") as image:
        assert adapter.analyze(image).area_value() == ""
    assert calls == []


@pytest.mark.parametrize("kind", ["number", "date"])
def test_numeric_areas_do_not_trigger_text_retry(monkeypatch, kind):
    adapter = PaddleLayout()
    expected = LayoutResult("", {})
    monkeypatch.setattr(adapter, "_analyze_value", lambda image, kind: expected)
    with Image.open(SAMPLES / "f0065_00.png") as image:
        assert adapter.analyze_area(image, kind=kind) is expected
