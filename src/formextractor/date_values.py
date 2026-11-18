"""Parse common date spellings without guessing missing digits or invalid dates."""

import re
from datetime import date

from .errors import ExtractionError

_MONTHS = {
    word: number
    for number, words in enumerate(
        (
            "jan january січ січень січня янв январь января",
            "feb february лют лютий лютого фев февраль февраля",
            "mar march бер березень березня мар март марта",
            "apr april квіт квітень квітня апр апрель апреля",
            "may трав травень травня май мая",
            "jun june черв червень червня июн июнь июня",
            "jul july лип липень липня июл июль июля",
            "aug august серп серпень серпня авг август августа",
            "sep sept september вер вересень вересня сен сент сентябрь сентября",
            "oct october жовт жовтень жовтня окт октябрь октября",
            "nov november лист листопад листопада ноя ноябрь ноября",
            "dec december груд грудень грудня дек декабрь декабря",
        ),
        1,
    )
    for word in words.split()
}


def _year(text):
    year = int(text)
    # A fixed, documented century window makes results independent of today's date.
    return year + (2000 if year < 50 else 1900) if len(text) == 2 else year


def _has_slash_conflict(text: str, value: date) -> bool:
    """Detect a second valid reading when OCR may have read date slashes as ones."""
    text = text.strip().rstrip(".")
    for day_width in (1, 2):
        for month_width in (1, 2):
            match = re.fullmatch(
                rf"([0-9]{{{day_width}}})([1/])"
                rf"([0-9]{{{month_width}}})([1/])([0-9]{{2}}|[0-9]{{4}})",
                text,
            )
            if match is None:
                continue
            day, first_separator, month, second_separator, year = match.groups()
            if "1" not in (first_separator, second_separator):
                continue
            day, month = int(day), int(month)
            if month > 12 and 1 <= day <= 12:
                day, month = month, day
            try:
                alternative = date(_year(year), month, day)
            except ValueError:
                continue
            if alternative != value:
                return True
    return False


def format_recognized_date(text: str) -> str:
    """Normalize numeric or named-month dates to DD.MM.YYYY; prefer day first."""
    parts = re.findall(r"[0-9]+", text)
    if not parts:
        return ""
    months = [
        _MONTHS[word] for word in re.findall(r"[^\W\d_]+", text.casefold()) if word in _MONTHS
    ]
    candidates = []
    if months:
        if len(months) == 1 and len(parts) == 2:
            if len(parts[0]) == 4 and 1 <= len(parts[1]) <= 2:
                candidates.append((int(parts[0]), months[0], int(parts[1])))
            elif 1 <= len(parts[0]) <= 2 and len(parts[1]) in (2, 4):
                candidates.append((_year(parts[1]), months[0], int(parts[0])))
    elif len(parts) == 3:
        if len(parts[0]) == 4 and all(1 <= len(p) <= 2 for p in parts[1:]):
            candidates.append(tuple(map(int, parts)))
        elif len(parts[2]) in (2, 4) and all(1 <= len(p) <= 2 for p in parts[:2]):
            day, month = map(int, parts[:2])
            # Recognize US dates only when their order is unambiguous.
            if month > 12 and 1 <= day <= 12:
                day, month = month, day
            candidates.append((_year(parts[2]), month, day))
    elif len(parts) == 1 and len(parts[0]) in (6, 8):
        digits = parts[0]
        candidates.append((_year(digits[4:]), int(digits[2:4]), int(digits[:2])))
        if len(digits) == 8:
            candidates.append((int(digits[:4]), int(digits[4:6]), int(digits[6:])))
    valid = set()
    for year, month, day in candidates:
        try:
            valid.add(date(year, month, day))
        except ValueError:
            continue
    if len(valid) != 1:
        raise ExtractionError("Could not recognize one valid date")
    value = valid.pop()
    if _has_slash_conflict(text, value):
        raise ExtractionError("Could not recognize one valid date: ambiguous '/' and '1'")
    return f"{value.day:02d}.{value.month:02d}.{value.year:04d}"
