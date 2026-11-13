"""Check the catalog against the installed pipeline without loading OCR models."""

import ast
from importlib.metadata import PackageNotFoundError, distribution

import pytest

from formextractor.ocr_languages import OCR_LANGUAGES


def test_languages_match_installed_ppstructure_model_selection():
    try:
        package = distribution("paddleocr")
    except PackageNotFoundError:
        pytest.skip("PaddleOCR must be installed to verify its supported languages")
    path = package.locate_file("paddleocr/_pipelines/pp_structurev3.py")
    tree = ast.parse(path.read_text(encoding="utf-8"))
    selector = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "_get_ocr_model_names"
    )
    # Execute only the pure model-name selector, without importing Paddle or
    # triggering native runtime initialization or model downloads.
    namespace = {}
    exec(compile(ast.Module(body=[selector], type_ignores=[]), str(path), "exec"), namespace)
    select = namespace["_get_ocr_model_names"]
    candidates = {
        node.value
        for node in ast.walk(selector)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }
    supported = {code for code in candidates if all(select(None, code, None))}
    assert set(OCR_LANGUAGES) == supported - {"french", "german"}
    for code in OCR_LANGUAGES:
        assert all(select(None, code, None)), code
