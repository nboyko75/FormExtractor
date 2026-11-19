import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image, ImageDraw

from formextractor import paddle as paddle_adapter
from formextractor.areas import RecognitionArea, area_image_path, recognize_batch
from formextractor.date_images import date_part_variants
from formextractor.field_images import remove_area_borders
from formextractor.paddle import LayoutResult, PaddleLayout


@pytest.mark.parametrize("grid", [False, True])
def test_removes_cell_and_table_rules_without_changing_geometry_or_inner_text(grid):
    with Image.new("RGB", (320, 120), "white") as image:
        draw = ImageDraw.Draw(image)
        draw.rectangle((3, 3, 316, 116), outline="black", width=3)
        draw.text((25, 25), "Dundee 12/05/1989", fill="black")
        if grid:
            draw.line((3, 60, 316, 60), fill="black", width=3)
            draw.line((160, 3, 160, 116), fill="black", width=3)
        original = np.asarray(image).copy()
        with remove_area_borders(image) as cleaned:
            pixels = np.asarray(cleaned)
            assert cleaned.size == image.size
            assert np.all(pixels[2:7, 20:300] > 240)
            assert np.all(pixels[113:118, 20:300] > 240)
            assert np.all(pixels[15:105, 2:7] > 240)
            assert np.all(pixels[15:105, 312:318] > 240)
            np.testing.assert_array_equal(pixels[20:45, 20:150], original[20:45, 20:150])
            if grid:
                assert np.all(pixels[58:63, 20:300] > 240)
                assert np.all(pixels[15:105, 158:163] > 240)
        np.testing.assert_array_equal(np.asarray(image), original)


def test_removes_slanted_open_box_edges_and_keeps_descender_beyond_bottom_rule():
    with Image.new("RGB", (300, 100), "white") as image:
        draw = ImageDraw.Draw(image)
        draw.line((2, 5, 299, 10), fill="black", width=3)
        draw.line((2, 80, 299, 85), fill="black", width=3)
        draw.line((2, 5, 2, 80), fill="black", width=3)
        draw.line((150, 65, 135, 94), fill="black", width=3)
        with remove_area_borders(image) as cleaned:
            pixels = np.asarray(cleaned)
            assert np.all(pixels[2:14, 30:120] > 240)
            assert np.all(pixels[76:89, 30:120] > 240)
            assert np.any(pixels[90:96, 132:141] < 128)
            # The stroke must still cross the former rule, not split into two.
            for y in range(75, 89):
                assert np.any(pixels[y, 132:153] < 180)


@pytest.mark.parametrize("mode,color", [("RGB", "white"), ("RGB", "black"), ("RGB", "red"),
                                        ("L", 255), ("L", 0)])
def test_uniform_area_is_preserved(mode, color):
    with Image.new(mode, (200, 60), color) as image, remove_area_borders(image) as cleaned:
        np.testing.assert_array_equal(np.asarray(cleaned), np.asarray(image.convert("RGB")))


def test_unboxed_digits_slashes_and_minus_sign_are_preserved():
    with Image.new("RGB", (300, 100), "white") as image:
        draw = ImageDraw.Draw(image)
        draw.line((145, 15, 145, 85), fill="black", width=3)  # A long handwritten 1.
        draw.line((200, 20, 170, 80), fill="black", width=3)
        draw.line((35, 50, 65, 50), fill="black", width=3)
        with remove_area_borders(image) as cleaned:
            np.testing.assert_array_equal(np.asarray(cleaned), np.asarray(image))


@pytest.mark.parametrize("filename", ["f0065_00.png", "f0251_00.png"])
def test_sample_city_rules_removed_without_clipping_handwriting(filename):
    path = Path(__file__).parent / "data" / "text_fields" / filename
    with Image.open(path) as image, remove_area_borders(image) as cleaned:
        pixels = np.asarray(cleaned)
        assert cleaned.size == image.size
        # Empty right-hand portion of the top border and central writing.
        assert np.mean(pixels[:16, 200:280]) > 250
        np.testing.assert_array_equal(pixels[20:60, 20:280], np.asarray(image)[20:60, 20:280])


@pytest.mark.parametrize("kind,value", [("text", "Dundee"), ("number", "123"),
                                        ("date", "01.02.2026")])
def test_batch_recognizes_and_saves_cleaned_areas_for_every_type(
    tmp_path, no_text_detector, kind, value
):
    source = tmp_path / "boxed.png"
    with Image.new("RGB", (300, 100), "white") as image:
        draw = ImageDraw.Draw(image)
        draw.rectangle((3, 3, 296, 96), outline="black", width=3)
        draw.text((30, 40), "Value", fill="black")
        image.save(source)
    before = source.read_bytes()
    received = []

    class OCR:
        def analyze(self, image, *, kind="text"):
            received.append(np.asarray(image).copy())
            assert np.all(received[-1][2:7, 20:280] > 240)
            return LayoutResult(value, {})

    result = recognize_batch(
        str(source), [str(source)],
        [RecognitionArea(number=1, page=1, kind=kind, x1=0, y1=0, x2=1, y2=1)], "*",
        layout=OCR(), text_detector=no_text_detector, save_areas=True,
    )
    assert result.rows[0].error is None
    assert result.rows[0].values == [value]
    with Image.open(area_image_path(source)) as saved:
        np.testing.assert_array_equal(np.asarray(saved), received[0])
    assert source.read_bytes() == before


def test_date_fallback_retains_box_geometry_but_primary_ocr_receives_clean_pixels(monkeypatch):
    path = Path(__file__).parent / "data" / "date_slashes" / "f0303_00.png"
    adapter = PaddleLayout()
    predictor = SimpleNamespace(config={"PostProcess": {"name": "CTCLabelDecode"}}, post_op=None)
    seen = []

    class Recognizer:
        paddlex_predictor = predictor

        def predict(self, input):
            seen.append(input.copy())
            return [{"rec_text": "08105189", "rec_score": .95}]

    monkeypatch.setattr(paddle_adapter, "_CharacterLimitedDecoder", lambda *args: None)
    monkeypatch.setattr(paddle_adapter, "_prepare_model_cache", lambda: None)
    monkeypatch.setitem(sys.modules, "paddleocr", SimpleNamespace(TextRecognition=object))
    adapter._value_recognizer = Recognizer()
    with Image.open(path) as original, remove_area_borders(original) as cleaned:
        def recover(image, data):
            assert image is original
            assert len(date_part_variants(image)) == 2
            return LayoutResult("08.09.1989", data)

        monkeypatch.setattr(adapter, "_recover_slash_date", recover)
        result = adapter.analyze_area(cleaned, kind="date", original_image=original)
        assert result.markdown == "08.09.1989"
        assert len(seen) == 2
        for pixels in seen:
            np.testing.assert_array_equal(pixels, np.asarray(cleaned)[:, :, ::-1])
