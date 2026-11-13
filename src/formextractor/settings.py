"""Per-user desktop preferences, independent of OCR configuration."""

import json
import os
import tempfile
from pathlib import Path

LANGUAGES = {
    "en": "English",
    "de": "Deutsch",
    "fr": "Français",
    "es": "Español",
    "pt": "Português",
    "it": "Italiano",
    "uk": "Українська",
    "ru": "Русский",
}


def settings_path() -> Path:
    base = os.environ.get("APPDATA") or os.environ.get("XDG_CONFIG_HOME")
    return (Path(base) if base else Path.home() / ".config") / "FormExtractor" / "settings.json"


def load_language(path: Path | None = None) -> str:
    try:
        value = json.loads((path or settings_path()).read_text(encoding="utf-8"))
        language = value.get("interface_language") if isinstance(value, dict) else None
        return language if isinstance(language, str) and language in LANGUAGES else "en"
    except (OSError, ValueError):
        return "en"


def save_language(language: str, path: Path | None = None) -> None:
    if language not in LANGUAGES:
        raise ValueError(f"Unknown interface language: {language}")
    destination = path or settings_path()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=destination.parent, delete=False
        ) as handle:
            temporary = Path(handle.name)
            json.dump({"interface_language": language}, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        os.replace(temporary, destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
