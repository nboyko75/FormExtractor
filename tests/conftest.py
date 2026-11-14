import pytest
from PIL import Image

from formextractor.schema import PageExtraction


@pytest.fixture
def no_text_detector():
    class Detector:
        def detect(self, image):
            return []

    return Detector()


@pytest.fixture
def page_data():
    return {
        "document_type": "Vehicle Inspection Report",
        "fields": [
            {"number": 1, "name": "Vehicle model", "value": "Skoda Yeti"},
            {"number": 2, "name": "Identifier", "value": "00123"},
            {"number": None, "name": "Notes", "value": ""},
            {"number": None, "name": "Notes", "value": None},
        ],
        "tables": [
            {
                "name": "Technical condition",
                "columns": ["Component", "Condition"],
                "rows": [["Engine", "Good"], ["Transmission", "Inspection required"]],
            }
        ],
    }


@pytest.fixture
def page_extraction(page_data):
    return PageExtraction.model_validate(page_data)


@pytest.fixture
def image_path(tmp_path):
    path = tmp_path / "scan.png"
    Image.new("RGB", (120, 80), "white").save(path)
    return path
