"""Preserve the OCR extra's metadata for PaddleX's runtime dependency checks."""

from importlib.metadata import requires

from packaging.requirements import Requirement
from PyInstaller.utils.hooks import copy_metadata

# Recursive metadata collection skips optional extras. PP-StructureV3 checks
# every distribution in paddlex[ocr], even when some models are disabled.
datas = []
for dependency in requires("paddlex") or []:
    requirement = Requirement(dependency)
    if requirement.marker and requirement.marker.evaluate({"extra": "ocr"}):
        datas += copy_metadata(requirement.name, recursive=True)
