import json

import pytest

from formextractor.errors import ExtractionError
from formextractor.schema import parse_extraction


def test_fenced_unicode_and_repeated_fields(page_data):
    page_data["fields"][0]["name"] = "Модель автомобіля"
    parsed = parse_extraction("```json\n" + json.dumps(page_data, ensure_ascii=False) + "\n```")
    assert parsed.fields[0].name == "Модель автомобіля"
    assert parsed.fields[1].value == "00123"
    assert parsed.fields[2].value == ""
    assert parsed.fields[3].value is None


@pytest.mark.parametrize("value", [123, True, [], {}])
def test_rejects_coercion_of_field_values(page_data, value):
    page_data["fields"][0]["value"] = value
    with pytest.raises(ExtractionError):
        parse_extraction(json.dumps(page_data))


@pytest.mark.parametrize(
    "response",
    [
        "{}",
        '{"fields": []}',
        "Here is your JSON: {}",
        '{"fields": [',
        '{"document_type":null,"fields":[],"fields":[],"tables":[]}',
    ],
)
def test_rejects_missing_truncated_or_duplicate_properties(response):
    with pytest.raises(ExtractionError):
        parse_extraction(response)


def test_rejects_misaligned_table(page_data):
    page_data["tables"][0]["rows"][0].append("extra cell")
    with pytest.raises(ExtractionError):
        parse_extraction(json.dumps(page_data))


def test_rejects_empty_label_and_extra_properties(page_data):
    page_data["fields"][0]["name"] = "  "
    with pytest.raises(ExtractionError):
        parse_extraction(json.dumps(page_data))
    page_data["fields"][0]["name"] = "Name"
    page_data["fields"][0]["confidence"] = 0.99
    with pytest.raises(ExtractionError):
        parse_extraction(json.dumps(page_data))
