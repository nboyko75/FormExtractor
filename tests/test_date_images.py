from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw

from formextractor.date_images import date_part_variants


@pytest.mark.parametrize("filename", ["f0047_00.png", "f0303_00.png"])
def test_boxed_handwriting_splits_into_three_padded_parts(filename):
    with Image.open(Path(__file__).parent / "data" / "date_slashes" / filename) as image:
        before = image.tobytes()
        variants = date_part_variants(image)
        assert len(variants) == 2
        for _, parts in variants:
            assert len(parts) == 3
            for part in parts:
                assert part.shape[2] == 3
                assert np.any(part < 128)
                assert np.all(part[:8] == 255)
                assert np.all(part[-8:] == 255)
                assert np.all(part[:, :12] == 255)
                assert np.all(part[:, -12:] == 255)
        assert image.tobytes() == before


@pytest.mark.parametrize("color", ["white", "black"])
def test_uniform_crop_has_no_date_parts(color):
    with Image.new("RGB", (180, 50), color) as image:
        assert date_part_variants(image) == []


def test_upright_ones_are_not_interpreted_as_slashes():
    with Image.new("RGB", (180, 60), "white") as image:
        draw = ImageDraw.Draw(image)
        draw.rectangle((1, 1, 178, 58), outline="black", width=2)
        for x in (20, 40, 65, 90, 115, 140):
            draw.line((x, 12, x, 45), fill="black", width=3)
        assert date_part_variants(image) == []
