import cv2
import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageFont

from formextractor.area_matching import AreaTemplate, PageMatcher, PageTemplate
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


def test_batch_publishes_each_area_and_continues_after_match_and_ocr_errors(
    tmp_path, no_text_detector
):
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
        text_detector=no_text_detector,
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


def text_patch(text):
    patch = Image.new("RGB", (360, 260), "white")
    with sample_area() as graphic:
        patch.paste(graphic, (60, 10))
    ImageDraw.Draw(patch).text((15, 190), text, fill="black", font=ImageFont.load_default(size=24))
    return patch


@pytest.mark.parametrize("angle,scale", [(0, 1), (90, 1), (17, 1.25), (-32, 0.8)])
def test_changed_text_is_excluded_but_retained_in_upright_crop(angle, scale):
    region = np.float32([[10, 180], [350, 180], [350, 235], [10, 235]])
    area = RecognitionArea(number=1, page=1, x1=0, y1=0, x2=1, y2=1)
    with text_patch("OLD VALUE 12345") as source, text_patch("New 987") as patch:
        template = AreaTemplate(source, area, text_regions=[region])
        transform = cv2.getRotationMatrix2D((180, 130), angle, scale)
        transform[:, 2] += (150, 160)
        pixels = cv2.warpAffine(
            np.asarray(patch), transform, (700, 650), borderValue=(255, 255, 255)
        )
        target_region = cv2.transform(region[None], transform)[0]
        with Image.fromarray(pixels) as target:
            with PageMatcher(target, text_regions=[target_region]).crop(template) as found:
                assert abs(found.width - patch.width * scale) <= 2
                assert abs(found.height - patch.height * scale) <= 2
                with found.resize(patch.size) as normalized:
                    error = np.abs(np.asarray(normalized).astype(float) - np.asarray(patch)).mean()
                    assert error < 15
                    # The crop sent to recognition still contains the new text.
                    assert np.asarray(normalized)[195:220, 20:100].min() < 80


def test_template_fallback_ignores_different_text_boxes_and_text_decoy():
    source_region = [[10, 180], [350, 180], [350, 235], [10, 235]]
    area = RecognitionArea(number=1, page=1, x1=0, y1=0, x2=1, y2=1)
    with text_patch("OLD VALUE 12345") as source, text_patch("New 987") as patch:
        template = AreaTemplate(source, area, text_regions=[source_region])
        template.descriptors = None  # Exercise correlation when SIFT has too few anchors.
        with Image.new("RGB", (850, 350), "white") as target:
            target.paste(patch, (430, 40))
            ImageDraw.Draw(target).text(
                (35, 210), "OLD VALUE 12345", fill="black", font=ImageFont.load_default(size=24)
            )
            regions = [
                [[440, 220], [560, 220], [560, 275], [440, 275]],
                [[30, 200], [370, 200], [370, 255], [30, 255]],
            ]
            with PageMatcher(target, text_regions=regions).crop(template) as found:
                np.testing.assert_array_equal(np.asarray(found), np.asarray(patch))


def test_text_alone_cannot_supply_a_match_even_on_identical_pages():
    area = RecognitionArea(number=1, page=1, x1=0, y1=0, x2=1, y2=1)
    with Image.new("RGB", (360, 100), "white") as page:
        ImageDraw.Draw(page).text((15, 20), "Only text 12345", fill="black")
        regions = [[[5, 10], [350, 10], [350, 80], [5, 80]]]
        template = AreaTemplate(page, area, text_regions=regions)
        with pytest.raises(ExtractionError, match="No reliable image match"):
            PageMatcher(page, text_regions=regions).crop(template)


def test_identical_graphics_cannot_be_disambiguated_by_text():
    region = np.array([[10, 180], [350, 180], [350, 235], [10, 235]])
    area = RecognitionArea(number=1, page=1, x1=0, y1=0, x2=1, y2=1)
    with text_patch("OLD VALUE") as source, text_patch("New 987") as other:
        template = AreaTemplate(source, area, text_regions=[region])
        with Image.new("RGB", (850, 350), "white") as target:
            target.paste(source, (20, 40))
            target.paste(other, (450, 40))
            regions = [region + (20, 40), region + (450, 40)]
            with pytest.raises(ExtractionError, match="No reliable image match"):
                PageMatcher(target, text_regions=regions).crop(template)


def test_batch_detects_text_once_per_page_and_reads_original_crop(tmp_path):
    regions = [[[10, 180], [350, 180], [350, 235], [10, 235]]]
    source_path, target_path = tmp_path / "source.png", tmp_path / "target.png"
    with text_patch("OLD VALUE") as source, text_patch("New 987") as target:
        source.save(source_path)
        target.save(target_path)
        expected = np.asarray(target).copy()
    calls = []

    class Detector:
        def detect(self, image):
            calls.append(image.size)
            return regions

    class OCR:
        def analyze(self, image):
            np.testing.assert_allclose(np.asarray(image), expected, atol=5)
            return LayoutResult(markdown="New 987", data={})

    areas = [RecognitionArea(number=n, page=1, x1=0, y1=0, x2=1, y2=1) for n in (1, 2)]
    result = recognize_batch(
        str(source_path), [str(target_path)], areas, "*.png", layout=OCR(), text_detector=Detector()
    )
    assert result.rows[0].error is None
    assert result.rows[0].values == ["New 987", "New 987"]
    assert calls == [(360, 260), (360, 260)]


def repeated_form():
    """Narrow text fields are ambiguous without the surrounding form structure."""
    page = Image.new("RGB", (600, 800), "white")
    draw = ImageDraw.Draw(page)
    rng = np.random.default_rng(83)
    for _ in range(360):
        x, y = rng.integers((30, 30), (560, 760))
        if 430 < y < 550 or 610 < y < 730:
            continue
        radius = int(rng.integers(3, 9))
        draw.ellipse((x, y, x + radius, y + radius), fill="black")
    areas, regions = [], []
    for index in range(8):
        x, y = 30 + (index % 4) * 140, 450 + (index // 4) * 180
        draw.rectangle((x, y, x + 130, y + 90), outline="black", width=2)
        draw.line((x, y + 45, x + 130, y + 45), fill="black", width=2)
        draw.text((x + 8, y + 58), f"OLD {index}", fill="black")
        regions.append([[x + 4, y + 50], [x + 126, y + 50], [x + 126, y + 84], [x + 4, y + 84]])
        areas.append(
            RecognitionArea(
                number=index + 1,
                page=1,
                x1=(x + 4) / 600,
                y1=(y + 50) / 800,
                x2=(x + 126) / 600,
                y2=(y + 84) / 800,
            )
        )
    return page, areas, np.float32(regions)


@pytest.mark.parametrize("angle,scale", [(0, 1), (90, 1), (180, 1), (270, 1), (0, 2)])
@pytest.mark.parametrize("changed_text", [False, True])
def test_page_context_preserves_all_eight_text_only_fields(angle, scale, changed_text):
    page, areas, regions = repeated_form()
    reference = PageTemplate(page, text_regions=regions)
    templates = [AreaTemplate(page, area, reference=reference) for area in areas]
    with page.copy() as changed:
        if changed_text:
            draw = ImageDraw.Draw(changed)
            for index, area in enumerate(areas):
                x1, y1, x2, y2 = area.pixel_box(*page.size)
                draw.rectangle((x1, y1, x2 - 1, y2 - 1), fill="white")
                draw.text((x1 + 4, y1 + 8), f"NEW {index * 123}", fill="blue")
        target = changed.resize((600 * scale, 800 * scale), Image.Resampling.NEAREST)
    rotated = target.rotate(angle, expand=True)
    # Rotate the detector polygons exactly as PIL rotates the target pixels.
    transformed_regions = regions * scale
    width, height = target.size
    for _ in range(angle // 90):
        transformed_regions = np.stack(
            (transformed_regions[:, :, 1], width - 1 - transformed_regions[:, :, 0]), axis=2
        )
        width, height = height, width
    matcher = PageMatcher(rotated, text_regions=transformed_regions)
    for template in templates:
        assert template.descriptors is None
        with (
            matcher.crop(template) as crop,
            target.crop(template.area.pixel_box(*target.size)) as expected,
        ):
            np.testing.assert_array_equal(np.asarray(crop), np.asarray(expected))
    assert len(matcher.alignments) == 1
    page.close()
    target.close()
    rotated.close()


@pytest.mark.parametrize("angle,scale", [(0, 1), (13, 1.15), (-27, 0.85)])
def test_page_alignment_reads_changed_fields_after_translation_rotation_and_scaling(angle, scale):
    page, areas, regions = repeated_form()
    reference = PageTemplate(page, text_regions=regions)
    changed = page.copy()
    draw = ImageDraw.Draw(changed)
    for index, area in enumerate(areas):
        x1, y1, x2, y2 = area.pixel_box(*page.size)
        draw.rectangle((x1, y1, x2 - 1, y2 - 1), fill="white")
        draw.text((x1 + 4, y1 + 8), f"NEW {index * 123}", fill="blue")
    transform = cv2.getRotationMatrix2D((300, 400), angle, scale)
    transform[:, 2] += (200, 200)
    pixels = cv2.warpAffine(
        np.asarray(changed), transform, (1000, 1200), borderValue=(255, 255, 255)
    )
    target_regions = cv2.transform(regions.reshape(1, -1, 2), transform).reshape(-1, 4, 2)
    with Image.fromarray(pixels) as target:
        matcher = PageMatcher(target, text_regions=target_regions)
        for area in areas:
            template = AreaTemplate(page, area, reference=reference)
            with (
                matcher.crop(template) as crop,
                changed.crop(area.pixel_box(*page.size)) as expected,
            ):
                assert abs(crop.width - expected.width * scale) <= 1
                assert abs(crop.height - expected.height * scale) <= 1
                with crop.resize(expected.size) as normalized:
                    error = np.abs(
                        np.asarray(normalized).astype(float) - np.asarray(expected)
                    ).mean()
                    assert error < 12
        assert matcher.alignments[reference] is not None
    changed.close()
    page.close()


def test_batch_saves_all_eight_masked_fields_in_area_number_order(tmp_path):
    page, areas, regions = repeated_form()
    source = tmp_path / "form.png"
    page.save(source)
    expected = [page.crop(area.pixel_box(*page.size)) for area in areas]

    class Detector:
        def detect(self, image):
            return regions

    class OCR:
        def __init__(self):
            self.count = 0

        def analyze(self, image):
            np.testing.assert_array_equal(np.asarray(image), np.asarray(expected[self.count]))
            self.count += 1
            return LayoutResult(markdown=str(self.count), data={})

    result = recognize_batch(
        str(source),
        [str(source)],
        list(reversed(areas)),
        "*.png",
        layout=OCR(),
        text_detector=Detector(),
        save_areas=True,
    )
    assert result.rows[0].error is None
    assert result.rows[0].values == [str(number) for number in range(1, 9)]
    with Image.open(tmp_path / "form_areas.png") as saved:
        assert saved.size == (
            max(crop.width for crop in expected),
            sum(crop.height for crop in expected),
        )
        top = 0
        for crop in expected:
            with saved.crop((0, top, crop.width, top + crop.height)) as actual:
                np.testing.assert_array_equal(np.asarray(actual), np.asarray(crop))
            top += crop.height
            crop.close()
    page.close()
