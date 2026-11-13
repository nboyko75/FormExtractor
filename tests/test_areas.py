from pathlib import Path

import pymupdf
import pytest
from PIL import Image, ImageDraw

from formextractor.areas import (
    BatchResult,
    RecognitionArea,
    batch_text,
    matching_files,
    read_areas,
    recognize_batch,
)
from formextractor.errors import ExtractionError
from formextractor.localization import Translator
from formextractor.paddle import LayoutResult
from formextractor.settings import load_settings, save_language, update_settings


def area(number=1, page=1, x1=0.0, x2=0.5):
    return RecognitionArea(number=number, page=page, x1=x1, y1=0.0, x2=x2, y2=1.0)


class ColorOCR:
    def __init__(self):
        self.sizes = []

    def analyze(self, image):
        self.sizes.append(image.size)
        colors = image.getcolors()
        assert len(colors) == 1, "Pixels outside the selected area reached recognition"
        return LayoutResult(markdown=str(colors[0][1]), data={})


def test_batch_crops_only_areas_scales_coordinates_and_orders_columns(tmp_path):
    for name, size in [("b.png", (240, 160)), ("a.png", (120, 80))]:
        with Image.new("RGB", size, "red") as image:
            ImageDraw.Draw(image).rectangle((size[0] // 2, 0, size[0], size[1]), fill="green")
            image.save(tmp_path / name)
    source = tmp_path / "a.png"
    sources = matching_files(source, "*.png")
    layout = ColorOCR()
    notices = []
    result = recognize_batch(
        str(source),
        list(map(str, sources)),
        [area(3), area(1, x1=0.5, x2=1.0)],
        "*.png",
        language="uk",
        layout=layout,
        on_file=lambda filename: notices.append((filename, len(layout.sizes))),
    )
    assert result.columns == [1, 3]
    assert [Path(row.file).name for row in result.rows] == ["a.png", "b.png"]
    assert all(row.values == ["(0, 128, 0)", "(255, 0, 0)"] for row in result.rows)
    assert layout.sizes == [(60, 80), (60, 80), (120, 160), (120, 160)]
    assert notices == [(str(sources[0]), 0), (str(sources[1]), 2)]
    assert result.language == "uk"
    assert BatchResult.model_validate_json(result.model_dump_json()) == result
    text = batch_text(result, Translator())
    lines = [[cell.strip() for cell in line.split("|")[1:-1]] for line in text.splitlines()]
    assert lines[0] == ["File", "Area 1", "Area 3"]
    assert lines[2] == ["a.png", "(0, 128, 0)", "(255, 0, 0)"]


def test_batch_missing_pages_and_unreadable_files_do_not_stop_other_files(tmp_path):
    good = tmp_path / "good.pdf"
    with pymupdf.open() as document:
        document.new_page(width=120, height=80)
        document.new_page(width=120, height=80)
        document.save(good)
    short = tmp_path / "short.png"
    Image.new("RGB", (120, 80), "white").save(short)
    broken = tmp_path / "broken.pdf"
    broken.write_bytes(b"not a PDF")
    result = recognize_batch(
        str(good),
        list(map(str, [broken, short, good])),
        [area(1), area(2, page=2)],
        "*",
        layout=ColorOCR(),
    )
    assert result.rows[0].error and result.rows[0].values == [None, None]
    assert result.rows[1].error == "Missing template pages: 2"
    assert result.rows[1].values[0] and result.rows[1].values[1] is None
    assert result.rows[2].error is None and all(result.rows[2].values)
    assert "Error" in batch_text(result, Translator()).splitlines()[0]


def test_file_masks_are_local_case_insensitive_and_exclude_unsupported_files(tmp_path):
    for name in ["b.PDF", "a.png", "scan.jpg", "notes.txt"]:
        (tmp_path / name).touch()
    (tmp_path / "folder.pdf").mkdir()
    (tmp_path / "folder.pdf" / "nested.pdf").touch()
    assert [path.name for path in matching_files(tmp_path / "a.png", "*.pdf;*.PNG")] == [
        "a.png",
        "b.PDF",
    ]
    assert matching_files(tmp_path / "a.png", "missing*") == []
    for mask in ("", "../*.pdf", "folder\\*.pdf", "D:*.pdf"):
        with pytest.raises(ExtractionError):
            matching_files(tmp_path / "a.png", mask)


def test_area_validation_and_preferences_preserve_unrelated_settings(tmp_path):
    valid = area().model_dump()
    assert read_areas([valid, valid, {**valid, "x2": -1}, "broken"]) == [area()]
    with pytest.raises(ValueError):
        area(x1=0.8, x2=0.2)
    path = tmp_path / "settings.json"
    update_settings({"window": {"geometry": "900x650+50+70"}, "files": {"scan": [valid]}}, path)
    save_language("de", path)
    update_settings({"ocr_language": "uk", "file_mask": "*.pdf"}, path)
    stored = load_settings(path)
    assert stored["interface_language"] == "de"
    assert stored["window"]["geometry"] == "900x650+50+70"
    assert stored["files"] == {"scan": [valid]}


def test_no_area_or_empty_batch_never_runs_ocr():
    with pytest.raises(ExtractionError, match="area"):
        recognize_batch("scan.png", ["scan.png"], [], "*", layout=ColorOCR())
    with pytest.raises(ExtractionError, match="No supported files"):
        recognize_batch("scan.png", [], [area()], "*", layout=ColorOCR())
