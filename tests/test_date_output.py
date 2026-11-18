import pytest

from formextractor.errors import ExtractionError
from formextractor.output import clean_recognized_text


@pytest.mark.parametrize(
    "text,expected",
    [
        ("21.03.2026р.", "21.03.2026"),
        ("21.03.2026p.", "21.03.2026"),
        ("21/03/2026", "21.03.2026"),
        ("21-3-2026", "21.03.2026"),
        ("2026-03-21", "21.03.2026"),
        ("2026/3/21", "21.03.2026"),
        ("03/21/2026", "21.03.2026"),
        ("06/04/2026", "06.04.2026"),
        ("21 03 2026", "21.03.2026"),
        ("21,03,2026", "21.03.2026"),
        ("21.03.26", "21.03.2026"),
        ("03/21/26", "21.03.2026"),
        ("01.01.49", "01.01.2049"),
        ("01.01.50", "01.01.1950"),
        ("210326", "21.03.2026"),
        ("21032026", "21.03.2026"),
        ("20260321", "21.03.2026"),
        ("21 March 2026", "21.03.2026"),
        ("March 21, 2026", "21.03.2026"),
        ("2026-Mar-21", "21.03.2026"),
        ("21 березня 2026 р.", "21.03.2026"),
        ("21 марта 2026 г.", "21.03.2026"),
        ("6 квіт. 26", "06.04.2026"),
        ("29/02/24", "29.02.2024"),
        ("08/05/89", "08.05.1989"),
        ("08.10.5189", "08.10.5189"),
        ("08051989", "08.05.1989"),
    ],
)
def test_common_dates_are_normalized_and_idempotent(text, expected):
    assert clean_recognized_text(text, kind="date") == expected
    assert clean_recognized_text(expected, kind="date") == expected


@pytest.mark.parametrize(
    "text",
    [
        "21.03.20267.",
        "21.03.20260",
        "29/02/25",
        "31 April 2026",
        "21/03",
        "99.21.03.2026",
        "21 March 2026 and 22 March 2026",
        "21.03.2026 / 22.03.2026",
        "08105189",
        "08105189.",
        "08/05189",
        "08105/89",
    ],
)
def test_invalid_dates_do_not_get_truncated_or_silently_repaired(text):
    with pytest.raises(ExtractionError, match="valid date"):
        clean_recognized_text(text, kind="date")
