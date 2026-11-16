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


def test_batch_crops_only_areas_scales_coordinates_and_orders_columns(tmp_path, no_text_detector):
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
        text_detector=no_text_detector,
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


def test_batch_missing_pages_and_unreadable_files_do_not_stop_other_files(
    tmp_path, no_text_detector
):
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
        text_detector=no_text_detector,
    )
    assert result.rows[0].error and result.rows[0].values == [None, None]
    assert result.rows[1].error == "Missing template pages: 2"
    assert result.rows[1].values[0] and result.rows[1].values[1] is None
    assert result.rows[2].error is None and all(result.rows[2].values)
    assert "Error" in batch_text(result, Translator()).splitlines()[0]


@pytest.mark.parametrize(
    "recognized, expected",
    [
        ('<div style="text-align: center;">00123</div>', "00123"),
        ('<DIV title="a > b"><div>ІСВ 1МБ</div></DIV>', "ІСВ 1МБ"),
        ("<div>First</div><div>Second</div>", "First\n\nSecond"),
        ("<div> </div>", ""),
        ("  Reading < 5 and > 1  ", "Reading < 5 and > 1"),
        (
            "<div><table><tr><td>4291</td></tr></table></div>",
            "<table><tr><td>4291</td></tr></table>",
        ),
    ],
)
def test_recognized_values_omit_div_wrappers(image_path, no_text_detector, recognized, expected):
    class OCR:
        def analyze(self, image):
            return LayoutResult(markdown=recognized, data={})

    updates = []
    result = recognize_batch(
        str(image_path),
        [str(image_path)],
        [area()],
        "*.png",
        layout=OCR(),
        text_detector=no_text_detector,
        on_row=updates.append,
    )
    assert result.rows[0].error is None
    assert result.rows[0].values == [expected]
    assert all(update.values == [expected] for update in updates)
    assert BatchResult.model_validate_json(result.model_dump_json()).rows[0].values == [expected]
    assert "<div" not in batch_text(result, Translator()).lower()


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
    legacy = {key: value for key, value in valid.items() if key != "kind"}
    assert read_areas([legacy])[0].kind == "text"
    assert read_areas([{**valid, "kind": "date"}])[0].kind == "date"
    assert read_areas([{**valid, "kind": "unknown"}]) == []
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


def test_mixed_area_types_reach_recognition_and_survive_saved_results(image_path, no_text_detector):
    selected = [
        area(number).model_copy(update={"kind": kind})
        for number, kind in enumerate(("text", "number", "date"), 1)
    ]
    calls = []

    class OCR:
        def analyze(self, image, *, kind="text"):
            calls.append(kind)
            return LayoutResult(
                {"text": "Vehicle", "number": "-001,5", "date": "21.03.2026"}[kind], {}
            )

    result = recognize_batch(
        str(image_path),
        [str(image_path)],
        selected,
        "*.png",
        layout=OCR(),
        text_detector=no_text_detector,
    )
    assert calls == ["text", "number", "date"]
    assert result.rows[0].error is None
    assert result.rows[0].values == ["Vehicle", "-001,5", "21.03.2026"]
    restored = BatchResult.model_validate_json(result.model_dump_json())
    assert [area.kind for area in restored.areas] == calls


@pytest.mark.parametrize("save_areas", [False, True])
def test_save_area_images_contains_only_crops_in_number_order(
    tmp_path, no_text_detector, save_areas
):
    paths = [tmp_path / name for name in ("first.png", "second.png")]
    with Image.new("RGB", (120, 80), "blue") as image:
        draw = ImageDraw.Draw(image)
        draw.rectangle((0, 0, 39, 79), fill="red")
        draw.rectangle((40, 0, 79, 79), fill="green")
        for path in paths:
            image.save(path)
    originals = [path.read_bytes() for path in paths]
    selected = [area(3, x2=1 / 3), area(1, x1=1 / 3, x2=2 / 3)]

    def on_file(filename):
        if filename == str(paths[1]):
            assert (tmp_path / "first_areas.png").exists() is save_areas

    result = recognize_batch(
        str(paths[0]),
        list(map(str, paths)),
        selected,
        "*.png",
        layout=ColorOCR(),
        text_detector=no_text_detector,
        save_areas=save_areas,
        on_file=on_file,
    )
    assert all(row.error is None for row in result.rows)
    assert [path.read_bytes() for path in paths] == originals
    for path in paths:
        destination = path.with_name(path.stem + "_areas.png")
        assert destination.exists() is save_areas
        if save_areas:
            with Image.open(destination) as saved:
                assert saved.size == (40, 160)
                assert saved.getpixel((20, 20)) == (0, 128, 0)
                assert saved.getpixel((20, 100)) == (255, 0, 0)
                assert set(color for _, color in saved.getcolors()) == {(0, 128, 0), (255, 0, 0)}
    assert matching_files(paths[0], "*") == paths


def test_save_areas_includes_matches_even_when_ocr_fails(tmp_path, no_text_detector):
    source = tmp_path / "source.png"
    with Image.new("RGB", (120, 80), "red") as image:
        image.save(source)

    class BrokenOCR:
        def analyze(self, image):
            raise RuntimeError("OCR failed")

    result = recognize_batch(
        str(source),
        [str(source)],
        [area()],
        "*.png",
        save_areas=True,
        layout=BrokenOCR(),
        text_detector=no_text_detector,
    )
    assert "OCR failed" in result.rows[0].error
    with Image.open(tmp_path / "source_areas.png") as saved:
        assert saved.size == (60, 80)
        assert saved.getpixel((0, 0)) == (255, 0, 0)


def test_area_image_save_failure_keeps_results_and_continues(
    tmp_path, no_text_detector, monkeypatch
):
    from formextractor import areas as module

    paths = [tmp_path / name for name in ("first.png", "second.png")]
    with Image.new("RGB", (120, 80), "red") as image:
        for path in paths:
            image.save(path)
    real_save = module._save_area_image

    def save(destination, crops):
        if destination.name == "first_areas.png":
            raise PermissionError("Read-only output")
        real_save(destination, crops)

    monkeypatch.setattr(module, "_save_area_image", save)
    result = recognize_batch(
        str(paths[0]),
        list(map(str, paths)),
        [area()],
        "*.png",
        save_areas=True,
        layout=ColorOCR(),
        text_detector=no_text_detector,
    )
    assert "Cannot save areas image: Read-only output" in result.rows[0].error
    assert result.rows[0].values == ["(255, 0, 0)"]
    assert result.rows[1].error is None
    assert (tmp_path / "second_areas.png").exists()


def test_area_images_never_overwrite_batch_inputs(tmp_path, no_text_detector):
    source = tmp_path / "scan.png"
    protected = tmp_path / "scan_areas.png"
    with Image.new("RGB", (120, 80), "red") as image:
        image.save(source)
        image.save(protected)
    original = protected.read_bytes()
    result = recognize_batch(
        str(source),
        [str(source), str(protected)],
        [area()],
        "*.png",
        save_areas=True,
        layout=ColorOCR(),
        text_detector=no_text_detector,
    )
    assert "must not overwrite the input document" in result.rows[0].error
    assert protected.read_bytes() == original


def test_save_areas_combines_multiple_pages(tmp_path, no_text_detector):
    source = tmp_path / "scan.tiff"
    with (
        Image.new("RGB", (120, 80), "red") as first,
        Image.new("RGB", (120, 80), "green") as second,
    ):
        first.save(source, save_all=True, append_images=[second])
    result = recognize_batch(
        str(source),
        [str(source)],
        [area(1), area(2, page=2)],
        "*.tiff",
        save_areas=True,
        layout=ColorOCR(),
        text_detector=no_text_detector,
    )
    assert result.rows[0].error is None
    with Image.open(tmp_path / "scan_areas.png") as saved:
        assert saved.size == (60, 160)
        assert saved.getpixel((0, 0)) == (255, 0, 0)
        assert saved.getpixel((0, 80)) == (0, 128, 0)


def test_no_matching_areas_does_not_create_image(tmp_path, no_text_detector):
    source, target = tmp_path / "source.png", tmp_path / "target.png"
    with Image.new("RGB", (120, 80), "red") as image:
        image.save(source)
    with Image.new("RGB", (120, 80), "white") as image:
        image.save(target)
    result = recognize_batch(
        str(source),
        [str(target)],
        [area()],
        "*.png",
        save_areas=True,
        layout=ColorOCR(),
        text_detector=no_text_detector,
    )
    assert "No reliable image match" in result.rows[0].error
    assert not (tmp_path / "target_areas.png").exists()


def test_conflicting_area_image_names_are_reported(tmp_path, no_text_detector):
    paths = [tmp_path / name for name in ("scan.png", "scan.bmp")]
    with Image.new("RGB", (120, 80), "red") as image:
        for path in paths:
            image.save(path)
    result = recognize_batch(
        str(paths[0]),
        list(map(str, paths)),
        [area()],
        "*",
        save_areas=True,
        layout=ColorOCR(),
        text_detector=no_text_detector,
    )
    assert all("same areas image filename" in row.error for row in result.rows)
    assert not (tmp_path / "scan_areas.png").exists()


def test_failed_image_write_preserves_previous_export(tmp_path, monkeypatch):
    from formextractor.areas import _save_area_image

    destination = tmp_path / "scan_areas.png"
    destination.write_bytes(b"previous export")

    def failed_save(self, handle, **kwargs):
        handle.write(b"incomplete image")
        raise OSError("Disk full")

    monkeypatch.setattr(Image.Image, "save", failed_save)
    with Image.new("RGB", (20, 20), "red") as crop:
        with pytest.raises(OSError, match="Disk full"):
            _save_area_image(destination, {1: crop})
    assert destination.read_bytes() == b"previous export"
    assert list(tmp_path.iterdir()) == [destination]
