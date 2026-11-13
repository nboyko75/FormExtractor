import json
from string import Formatter

import pytest

from formextractor.localization import TRANSLATIONS, Translator
from formextractor.settings import LANGUAGES, load_language, save_language


@pytest.mark.parametrize("language", LANGUAGES)
def test_translation_catalogs_are_complete_and_preserve_placeholders(language):
    english = TRANSLATIONS["en"]
    translated = TRANSLATIONS[language]
    assert set(translated) == set(english)
    for key, value in translated.items():
        assert value.strip()

        def fields(text):
            return {field for _, field, _, _ in Formatter().parse(text) if field}

        assert fields(value) == fields(english[key]), key


@pytest.mark.parametrize(
    "content",
    ["{", "[]", "null", '{"interface_language": []}', '{"interface_language": "unknown"}'],
)
def test_missing_or_invalid_preferences_use_english(tmp_path, content):
    path = tmp_path / "settings.json"
    assert load_language(path) == "en"
    path.write_text(content, encoding="utf-8")
    assert load_language(path) == "en"


def test_preferences_round_trip_and_invalid_write_preserves_file(tmp_path):
    path = tmp_path / "nested" / "settings.json"
    save_language("uk", path)
    assert load_language(path) == "uk"
    with pytest.raises(ValueError):
        save_language("invalid", path)
    assert json.loads(path.read_text(encoding="utf-8")) == {"interface_language": "uk"}
    assert list(path.parent.iterdir()) == [path]


def test_localized_errors_preserve_context_and_technical_details():
    tr = Translator("de")
    assert tr.error(ValueError("Input file does not exist: scan.pdf")) == (
        "Eingabedatei existiert nicht: scan.pdf"
    )
    assert tr.error(ValueError("Output path must not overwrite the input document")) == (
        tr("output_overwrite")
    )
    assert tr.error(ValueError("Choose a .txt or .json filename.")) == tr("extension_error")
    assert (
        tr.error(ValueError("Native library failed"))
        == "Technische Details:\nNative library failed"
    )


def test_about_version_prefers_installed_msix_manifest(tmp_path, monkeypatch):
    from formextractor import dialogs

    monkeypatch.setattr(dialogs.sys, "frozen", True, raising=False)
    monkeypatch.setattr(dialogs.sys, "executable", str(tmp_path / "FormExtractor.exe"))
    manifest = tmp_path / "AppxManifest.xml"
    manifest.write_text(
        '<Package xmlns="http://schemas.microsoft.com/appx/manifest/foundation/windows10">'
        '<Identity Version="1.2.3.4" /></Package>',
        encoding="utf-8",
    )
    assert dialogs.app_version() == "1.2.3.4"
    manifest.write_text("invalid XML", encoding="utf-8")
    monkeypatch.setattr(dialogs, "version", lambda name: "0.2.0")
    assert dialogs.app_version() == "0.2.0"
