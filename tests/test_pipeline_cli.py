import json
import subprocess
import sys

import pytest

from formextractor import extract_document
from formextractor.cli import main
from formextractor.errors import ExtractionError
from formextractor.output import to_text, write_output
from formextractor.paddle import LayoutResult
from formextractor.pipeline import extract_explicit_fields


class FakeLayout:
    def __init__(self):
        self.calls = 0

    def analyze(self, image):
        self.calls += 1
        assert image.mode == "RGB"
        return LayoutResult(
            "Form heading\nIdentifier: 00123\nNotes: first\nNotes: second\nUnpaired text",
            {"res": {"rec_boxes": [[1, 2, 3, 4]], "rec_scores": [0.98]}},
        )


def test_paddle_pipeline_preserves_all_text_and_evidence(image_path):
    layout = FakeLayout()
    result = extract_document(image_path, layout=layout)
    page = result.pages[0]
    assert layout.calls == 1
    assert result.mode == "paddle"
    assert result.model == "PP-StructureV3"
    assert page.page == 1
    assert page.ocr["raw"]["res"]["rec_scores"] == [0.98]
    assert page.fields[0].value == "00123"
    assert len(page.fields) == 3
    assert "Unpaired text" in to_text(result)
    assert to_text(result).count("Identifier: 00123") == 1
    assert (page.image_width, page.image_height) == (120, 80)


def test_explicit_fields_do_not_guess_labels():
    fields = extract_explicit_fields(
        "1. Name: Alex\nМісто: Київ\nBlank:\n12:30\nhttps://example.com\n"
        '<td style="color:red">name</td>\n|Name: Alex|\nUnlabelled value'
    )
    assert [(field.name, field.value) for field in fields] == [
        ("Name", "Alex"),
        ("Місто", "Київ"),
        ("Blank", ""),
    ]
    assert fields[0].number == "1"


def test_document_values_omit_div_wrappers_but_keep_original_evidence(image_path):
    markdown = '<div style="text-align: center;">Identifier: 00123</div>'

    class WrappedLayout:
        def analyze(self, image):
            return LayoutResult(markdown, {})

    result = extract_document(image_path, layout=WrappedLayout())
    assert result.pages[0].text == "Identifier: 00123"
    assert result.pages[0].fields[0].value == "00123"
    assert "<div" not in to_text(result)
    assert result.pages[0].ocr["markdown"] == markdown


def test_empty_ocr_is_valid(image_path):
    class BlankLayout:
        def analyze(self, image):
            return LayoutResult("", {"res": {}})

    result = extract_document(image_path, layout=BlankLayout())
    assert result.pages[0].text == ""
    assert result.pages[0].fields == []


def test_cli_json_is_clean_and_text_file_is_utf8(monkeypatch, capsys, image_path):
    class NoisyLayout(FakeLayout):
        def analyze(self, image):
            print("library diagnostic")
            return LayoutResult("Місто: Київ", {"res": {}})

    monkeypatch.setattr("formextractor.cli.PaddleLayout", lambda **kwargs: NoisyLayout())
    assert main([str(image_path)]) == 0
    captured = capsys.readouterr()
    assert json.loads(captured.out)["pages"][0]["text"] == "Місто: Київ"
    assert "library diagnostic" in captured.err
    output = image_path.parent / "output.txt"
    assert main([str(image_path), "--format", "text", "-o", str(output)]) == 0
    assert "Місто: Київ" in output.read_text(encoding="utf-8")


def test_failure_preserves_existing_output(monkeypatch, image_path):
    class FailingLayout:
        def analyze(self, image):
            raise ExtractionError("OCR failed")

    monkeypatch.setattr("formextractor.cli.PaddleLayout", lambda **kwargs: FailingLayout())
    output = image_path.parent / "result.json"
    output.write_text("previous result", encoding="utf-8")
    assert main([str(image_path), "-o", str(output)]) == 1
    assert output.read_text(encoding="utf-8") == "previous result"


def test_cannot_overwrite_input(image_path):
    original = image_path.read_bytes()
    with pytest.raises(ExtractionError, match="overwrite"):
        write_output(image_path, "result", source=image_path)
    assert main([str(image_path), "-o", str(image_path)]) == 1
    assert image_path.read_bytes() == original


def test_invalid_input_fails_before_loading_models(monkeypatch, tmp_path):
    layout = FakeLayout()
    monkeypatch.setattr("formextractor.cli.PaddleLayout", lambda **kwargs: layout)
    assert main([str(tmp_path / "missing.pdf")]) == 1
    assert layout.calls == 0


def test_package_does_not_import_ollama():
    result = subprocess.run(
        [sys.executable, "-c", "import formextractor, sys; assert 'ollama' not in sys.modules"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
