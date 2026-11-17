import pytest

from formextractor.errors import ExtractionError
from formextractor.output import clean_recognized_text


@pytest.mark.parametrize(
    "text,expected",
    [
        ("# -001,50 км", "-1.5"),
        ("<span>+12.500</span>", "12.5"),
        ("319 320 м/г", "319320"),
        ("1\u00a0234,50", "1234.5"),
        ("0,70 л", "0.7"),
        ("120.00", "120"),
        ("000120", "120"),
        ("-0,00", "0"),
        ("−12,5", "-12.5"),
        ("-.5", "-0.5"),
        (".5", "0.5"),
        ("123...", "123"),
        ("Reading: 5426 км.", "5426"),
        ("999999999999999999999999.123456789", "999999999999999999999999.123456789"),
        ("---", ""),
    ],
)
def test_number_values_are_canonical_and_idempotent(text, expected):
    actual = clean_recognized_text(text, kind="number")
    assert actual == expected
    assert clean_recognized_text(actual, kind="number") == expected


@pytest.mark.parametrize("text", ["12/34", "12 and 34", "1.2.3", "1,234.50", "12-34", "1e3"])
def test_ambiguous_numbers_are_reported_instead_of_joined(text):
    with pytest.raises(ExtractionError, match="one valid number"):
        clean_recognized_text(text, kind="number")


def test_text_identifiers_keep_leading_zeros_and_letters():
    assert clean_recognized_text("00123E3", kind="text") == "00123E3"
