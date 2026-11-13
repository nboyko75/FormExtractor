"""Check extra metadata collection without requiring PyInstaller or OCR models."""

import runpy
import sys
from pathlib import Path
from types import ModuleType

import pytest

HOOK = Path(__file__).resolve().parents[1] / "packaging" / "hooks" / "hook-paddlex.py"


def test_paddlex_hook_collects_ocr_extra_metadata(monkeypatch):
    monkeypatch.setattr(
        "importlib.metadata.requires",
        lambda name: [
            "numpy>=1.24",
            'opencv-contrib-python==4.10.0.84; extra == "ocr"',
            'python-bidi; extra == "ocr"',
            'faiss-cpu; extra == "cv"',
            'unsupported; python_version < "3.0" and extra == "ocr"',
        ],
    )
    calls = []
    hooks = ModuleType("PyInstaller.utils.hooks")

    def copy_metadata(name, recursive=False):
        calls.append((name, recursive))
        return [(f"/installed/{name}.dist-info", f"{name}.dist-info")]

    hooks.copy_metadata = copy_metadata
    monkeypatch.setitem(sys.modules, "PyInstaller.utils.hooks", hooks)
    result = runpy.run_path(str(HOOK))
    assert calls == [("opencv-contrib-python", True), ("python-bidi", True)]
    assert len(result["datas"]) == 2


def test_paddlex_hook_fails_build_when_required_metadata_is_missing(monkeypatch):
    from importlib.metadata import PackageNotFoundError

    monkeypatch.setattr("importlib.metadata.requires", lambda name: ['python-bidi; extra == "ocr"'])
    hooks = ModuleType("PyInstaller.utils.hooks")

    def copy_metadata(name, recursive=False):
        raise PackageNotFoundError(name)

    hooks.copy_metadata = copy_metadata
    monkeypatch.setitem(sys.modules, "PyInstaller.utils.hooks", hooks)
    with pytest.raises(PackageNotFoundError):
        runpy.run_path(str(HOOK))
