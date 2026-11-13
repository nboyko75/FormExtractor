import pymupdf
import pytest
from PIL import Image

from formextractor.documents import iter_pages, parse_pages
from formextractor.errors import ExtractionError


def test_pdf_selection_rendering_and_pixel_limit(tmp_path):
    source = tmp_path / "scan.pdf"
    with pymupdf.open() as document:
        for index in range(3):
            page = document.new_page(width=300, height=200)
            page.insert_text((20, 30), f"Page {index + 1}")
        document.save(source)
    seen = []
    for page in iter_pages(source, pages="3,1,1", dpi=200, max_pixels=10000):
        seen.append(page.number)
        assert page.image.width * page.image.height <= 10000
        assert page.image.mode == "RGB"
        assert page.image.getextrema()[0][0] < 255
    assert seen == [1, 3]


def test_multipage_tiff_and_rotation(tmp_path):
    source = tmp_path / "scan.tiff"
    Image.new("RGB", (40, 20), "red").save(
        source, save_all=True, append_images=[Image.new("RGB", (40, 20), "blue")]
    )
    seen = []
    for page in iter_pages(source, rotation=90):
        assert page.image.size == (20, 40)
        seen.append(page.image.getpixel((0, 0)))
    assert seen == [(255, 0, 0), (0, 0, 255)]


def test_transparency_and_exif_orientation(tmp_path):
    source = tmp_path / "scan.png"
    image = Image.new("RGBA", (40, 20), (0, 0, 0, 0))
    exif = Image.Exif()
    exif[274] = 6
    image.save(source, exif=exif)
    for page in iter_pages(source):
        assert page.image.size == (20, 40)
        assert page.image.getpixel((0, 0)) == (255, 255, 255)


@pytest.mark.parametrize("spec", ["0", "4", "2-1", "1-4", "", "one", "1,", "1-2-3"])
def test_invalid_page_selection(spec):
    with pytest.raises(ExtractionError):
        parse_pages(spec, 3)


def test_password_protected_pdf(tmp_path):
    source = tmp_path / "locked.pdf"
    with pymupdf.open() as document:
        document.new_page()
        document.save(source, encryption=pymupdf.PDF_ENCRYPT_AES_256, user_pw="secret")
    with pytest.raises(ExtractionError, match="Password-protected"):
        list(iter_pages(source))


def test_corrupt_and_unsupported_inputs(tmp_path):
    for name in ("bad.pdf", "bad.png", "bad.docx"):
        source = tmp_path / name
        source.write_bytes(b"invalid document")
        with pytest.raises(ExtractionError):
            list(iter_pages(source))


def test_generator_closes_image_when_stopped(image_path):
    pages = iter_pages(image_path)
    page = next(pages)
    pages.close()
    with pytest.raises(ValueError):
        page.image.getpixel((0, 0))
