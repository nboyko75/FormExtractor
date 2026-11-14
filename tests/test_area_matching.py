import cv2
import numpy as np
import pytest
from PIL import Image, ImageDraw

from formextractor.area_matching import AreaTemplate, PageMatcher
from formextractor.areas import RecognitionArea, recognize_batch
from formextractor.errors import ExtractionError
from formextractor.paddle import LayoutResult


def sample_area():
    image = Image.new("RGB", (220, 160), "white")
    draw = ImageDraw.Draw(image)
    rng = np.random.default_rng(42)
    for _ in range(100):
        x, y = rng.integers((10, 10), (205, 145))
        radius = int(rng.integers(2, 6))
        draw.ellipse((x, y, x + radius, y + radius), fill="black")
    draw.rectangle((20, 110, 90, 140), fill="red")
    return image


def template_page():
    page = Image.new("RGB", (400, 300), "white")
    patch = sample_area()
    page.paste(patch, (40, 30))
    patch.close()
    area = RecognitionArea(number=1, page=1, x1=0.1, y1=0.1, x2=0.65, y2=190 / 300)
    return page, area


@pytest.mark.parametrize(
    "angle,scale", [(0, 1), (90, 1), (180, 1), (270, 1), (17, 1.25), (-32, 0.8), (127, 1.1)]
)
def test_finds_translated_rotated_scaled_area_and_reads_target_pixels(angle, scale):
    page, area = template_page()
    template = AreaTemplate(page, area)
    patch = sample_area()
    # The value in the new file differs from the selected template.
    ImageDraw.Draw(patch).rectangle((20, 110, 90, 140), fill="blue")
    transform = cv2.getRotationMatrix2D((110, 80), angle, scale)
    transform[:, 2] += (140, 170)
    target = cv2.warpAffine(np.asarray(patch), transform, (500, 500), borderValue=(255, 255, 255))
    with Image.fromarray(target) as image, PageMatcher(image).crop(template) as found:
        assert abs(found.width - 220 * scale) <= 2
        assert abs(found.height - 160 * scale) <= 2
        # Upright crop has the updated blue value at the original bottom left.
        assert found.getpixel((round(50 * scale), round(125 * scale))) == (0, 0, 255)
        with found.resize(patch.size) as normalized:
            error = np.abs(np.asarray(normalized).astype(float) - np.asarray(patch)).mean()
        assert error < 15
    patch.close()
    page.close()


def test_no_match_never_falls_back_to_original_coordinates():
    page, area = template_page()
    with Image.new("RGB", page.size, "white") as unrelated:
        with pytest.raises(ExtractionError, match="No reliable image match"):
            PageMatcher(unrelated).crop(AreaTemplate(page, area))
    page.close()


def test_small_template_without_features_matches_after_quarter_turn():
    with Image.new("RGB", (100, 80), "white") as source:
        draw = ImageDraw.Draw(source)
        draw.rectangle((10, 10, 20, 20), fill="black")
        draw.rectangle((21, 16, 28, 18), fill="black")
        area = RecognitionArea(number=1, page=1, x1=0.08, y1=0.1, x2=0.3, y2=0.3)
        template = AreaTemplate(source, area)
        assert len(template.points) < 6
        with source.crop(area.pixel_box(*source.size)) as patch:
            with Image.new("RGB", (100, 80), "white") as target:
                target.paste(patch, (50, 40))
                with target.rotate(90, expand=True) as rotated:
                    with PageMatcher(rotated).crop(template) as found:
                        assert np.array_equal(np.asarray(found), np.asarray(patch))


def test_batch_publishes_each_area_and_continues_after_match_and_ocr_errors(tmp_path):
    page, first = template_page()
    second = RecognitionArea(number=2, page=1, x1=0.75, y1=0.7, x2=0.95, y2=0.95)
    # A distinctive second template is absent from the target image.
    draw = ImageDraw.Draw(page)
    draw.rectangle((300, 210, 375, 280), fill="black")
    draw.ellipse((310, 220, 350, 255), fill="white")
    source = tmp_path / "source.png"
    page.save(source)
    page.close()
    target, _ = template_page()
    target_path = tmp_path / "target.png"
    target.save(target_path)
    target.close()
    events = []

    class OCR:
        def analyze(self, image):
            events.append("ocr")
            if events.count("ocr") == 1:
                raise RuntimeError("OCR unavailable")
            return LayoutResult(markdown="found", data={})

    result = recognize_batch(
        str(source),
        [str(target_path), str(source)],
        [first, second],
        "*.png",
        layout=OCR(),
        on_row=lambda row: events.append(row),
    )
    assert result.rows[0].values == [None, None]
    assert "Area 1: OCR unavailable" in result.rows[0].error
    assert "Area 2: No reliable image match" in result.rows[0].error
    assert result.rows[1].values == ["found", "found"]
    assert result.rows[1].error is None
    updates = [event for event in events if event != "ocr"]
    assert updates[-3].values == ["found", None]
    assert updates[-2].values == ["found", "found"]
    assert events[0] == "ocr" and events[1].values == [None, None]
