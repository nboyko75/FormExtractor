import cv2
import numpy as np
import pytest
from PIL import Image, ImageDraw

from formextractor.area_matching import AreaTemplate, PageMatcher, PageTemplate
from formextractor.areas import RecognitionArea
from formextractor.form_alignment import RuledPage


def ruled_form(changed=False):
    """Repetitive fields, with no distinctive graphics to provide SIFT anchors."""
    page = Image.new("RGB", (600, 800), "white")
    draw = ImageDraw.Draw(page)
    draw.line((40, 90, 280, 90), fill="black", width=2)
    draw.line((350, 60, 560, 60), fill="black", width=2)
    draw.line((350, 110, 560, 110), fill="black", width=2)
    draw.rectangle((40, 150, 560, 390), outline="black", width=2)
    draw.line((230, 150, 230, 390), fill="black", width=2)
    draw.line((410, 150, 410, 390), fill="black", width=2)
    draw.line((410, 190, 560, 190), fill="black", width=2)
    draw.line((410, 230, 560, 230), fill="black", width=2)
    draw.rectangle((40, 410, 560, 740), outline="black", width=2)
    for x in (125, 270, 360, 470):
        draw.line((x, 410, x, 740), fill="black", width=2)
    for y in range(450, 740, 35):
        draw.line((40, y, 560, y), fill="black", width=2)
    boxes = [(55, 465 + 35 * n, 110, 482 + 35 * n) for n in range(8)]
    for index, (x1, y1, _x2, _y2) in enumerate(boxes):
        draw.text(
            (x1 + 2, y1 + 2),
            f"{'NEW' if changed else 'OLD'} {index}",
            fill="blue" if changed else "black",
        )
    regions = np.float32([[[x1, y1], [x2, y1], [x2, y2], [x1, y2]] for x1, y1, x2, y2 in boxes])
    return page, regions


def photograph(page, regions, corners, size):
    transform = cv2.getPerspectiveTransform(
        np.float32([[0, 0], [599, 0], [599, 799], [0, 799]]), np.float32(corners)
    )
    pixels = cv2.warpPerspective(np.asarray(page), transform, size, borderValue=(45, 55, 40))
    polygons = cv2.perspectiveTransform(regions.reshape(1, -1, 2), transform).reshape(-1, 4, 2)
    return Image.fromarray(pixels), polygons, transform


@pytest.mark.parametrize("turns", [0, 1, 2, 3])
def test_photographed_grid_keeps_all_changed_fields_aligned(turns):
    page, regions = ruled_form()
    changed, _ = ruled_form(changed=True)
    source, source_regions, source_transform = photograph(
        page, regions, [(70, 80), (697, 55), (730, 920), (90, 940)], (800, 1000)
    )
    target, target_regions, target_transform = photograph(
        changed, regions, [(110, 120), (755, 165), (690, 915), (65, 960)], (850, 1050)
    )
    rotated = target.rotate(90 * turns, expand=True)
    rotation = np.eye(3)
    width, height = target.size
    for _ in range(turns):
        rotation = np.array([[0, 1, 0], [-1, 0, width - 1], [0, 0, 1]]) @ rotation
        width, height = height, width
    target_regions = cv2.perspectiveTransform(target_regions.reshape(1, -1, 2), rotation)
    matcher = PageMatcher(rotated, text_regions=target_regions.reshape(-1, 4, 2))
    reference = PageTemplate(source, text_regions=source_regions)
    expected_transform = rotation @ target_transform @ np.linalg.inv(source_transform)
    for index, polygon in enumerate(source_regions):
        x1, y1 = polygon.min(axis=0) - 3
        x2, y2 = polygon.max(axis=0) + 3
        area = RecognitionArea(
            number=index + 1,
            page=1,
            x1=x1 / source.width,
            y1=y1 / source.height,
            x2=x2 / source.width,
            y2=y2 / source.height,
        )
        template = AreaTemplate(source, area, reference=reference)
        with matcher.crop(template) as crop:
            # The new blue values survive; the old template pixels cannot pass.
            rgb = np.asarray(crop).astype(float)
            assert (rgb[:, :, 2] - rgb[:, :, 0]).max() > 100
    assert len(matcher.alignments) == 1
    _, actual = matcher.alignments[reference]
    assert actual.shape == (3, 3)
    actual_points = cv2.perspectiveTransform(source_regions.reshape(1, -1, 2), actual)
    expected_points = cv2.perspectiveTransform(source_regions.reshape(1, -1, 2), expected_transform)
    assert np.linalg.norm(actual_points - expected_points, axis=2).max() < 3
    for image in (page, changed, source, target, rotated):
        image.close()


@pytest.mark.parametrize("kind", ["blank", "unrelated", "symmetric"])
def test_paper_outline_alone_or_ambiguous_ruling_cannot_establish_alignment(kind):
    page, regions = ruled_form()
    other = Image.new("RGB", page.size, "white")
    draw = ImageDraw.Draw(other)
    if kind == "unrelated":
        for x in range(60, 560, 60):
            draw.line((x, 40, x, 760), fill="black", width=2)
        for y in range(40, 761, 60):
            draw.line((60, y, 540, y), fill="black", width=2)
    elif kind == "symmetric":
        for x in range(60, 541, 80):
            draw.line((x, 80, x, 720), fill="black", width=2)
        for y in range(80, 721, 80):
            draw.line((60, y, 540, y), fill="black", width=2)
        page.close()
        page = other.copy()
    corners = [(70, 80), (697, 55), (730, 920), (90, 940)]
    source, _, _ = photograph(page, regions, corners, (800, 1000))
    target, _, _ = photograph(other, regions, corners, (800, 1000))
    gray = cv2.cvtColor(np.asarray(source), cv2.COLOR_RGB2GRAY)
    target_gray = cv2.cvtColor(np.asarray(target), cv2.COLOR_RGB2GRAY)
    mask = np.full(gray.shape, 255, np.uint8)
    assert RuledPage(gray, mask).align(RuledPage(target_gray, mask)) is None
    for image in (page, other, source, target):
        image.close()
