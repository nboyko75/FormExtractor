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


def load_settings(path: Path | None = None) -> dict:
    try:
        value = json.loads((path or settings_path()).read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


def load_language(path: Path | None = None) -> str:
    language = load_settings(path).get("interface_language")
    return language if isinstance(language, str) and language in LANGUAGES else "en"


def save_language(language: str, path: Path | None = None) -> None:
    if language not in LANGUAGES:
        raise ValueError(f"Unknown interface language: {language}")
    update_settings({"interface_language": language}, path)


def update_settings(changes: dict, path: Path | None = None) -> None:
    """Merge preferences so changing one setting does not erase saved templates."""
    value = load_settings(path)
    value.update(changes)
    destination = path or settings_path()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=destination.parent, delete=False
        ) as handle:
            temporary = Path(handle.name)
            json.dump(value, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        os.replace(temporary, destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
