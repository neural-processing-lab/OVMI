"""Reference distributions used by OVMI."""

from __future__ import annotations

from pathlib import Path
from typing import Hashable
from urllib.request import urlretrieve
from zipfile import ZipFile
from xml.etree.ElementTree import iterparse

import pandas as pd

SUBTLEX_UK_URL = "https://osf.io/d3jbg/download"
SUBTLEX_UK_FILENAME = "SUBTLEX-UK.xlsx"
WORD_COLUMN = "Spelling"
FREQUENCY_COLUMN = "FreqCount"

_DEFAULT_REFERENCE: dict[Hashable, float] | None = None


def default_reference() -> dict[Hashable, float]:
    """Return the default SUBTLEX-UK reference distribution."""

    global _DEFAULT_REFERENCE
    if _DEFAULT_REFERENCE is None:
        _DEFAULT_REFERENCE = load_subtlex_uk()
    return _DEFAULT_REFERENCE


def load_subtlex_uk(
    path: str | Path | None = None,
    *,
    cache_dir: str | Path | None = None,
    download: bool = True,
) -> dict[str, float]:
    """Load the SUBTLEX-UK frequency norm as a word-to-frequency mapping.

    If ``path`` is omitted, the file is read from the OVMI cache directory. When
    the cached file is absent and ``download`` is true, it is downloaded from
    OSF.
    """

    xlsx_path = Path(path).expanduser() if path is not None else _subtlex_cache_path(cache_dir)
    if not xlsx_path.exists():
        if not download:
            raise FileNotFoundError(f"SUBTLEX-UK file not found: {xlsx_path}")
        _download_subtlex_uk(xlsx_path)

    try:
        return _load_subtlex_uk_with_pandas(xlsx_path)
    except ImportError:
        return _load_subtlex_uk_from_xlsx_xml(xlsx_path)


def _subtlex_cache_path(cache_dir: str | Path | None = None) -> Path:
    if cache_dir is None:
        cache_root = Path.home() / ".cache" / "ovmi"
    else:
        cache_root = Path(cache_dir).expanduser()
    return cache_root / SUBTLEX_UK_FILENAME


def _download_subtlex_uk(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    try:
        urlretrieve(SUBTLEX_UK_URL, temporary_path)
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _load_subtlex_uk_with_pandas(path: Path) -> dict[str, float]:
    frame = pd.read_excel(path, usecols=[WORD_COLUMN, FREQUENCY_COLUMN])
    return _frame_to_reference(frame)


def _frame_to_reference(frame: pd.DataFrame) -> dict[str, float]:
    missing = {WORD_COLUMN, FREQUENCY_COLUMN}.difference(frame.columns)
    if missing:
        raise ValueError(f"SUBTLEX-UK file is missing columns: {sorted(missing)!r}")

    frame = frame[[WORD_COLUMN, FREQUENCY_COLUMN]].dropna()
    frame[WORD_COLUMN] = frame[WORD_COLUMN].astype(str).str.strip()
    frame[FREQUENCY_COLUMN] = pd.to_numeric(frame[FREQUENCY_COLUMN], errors="coerce")
    frame = frame[(frame[WORD_COLUMN] != "") & frame[FREQUENCY_COLUMN].notna()]
    frame = frame[frame[FREQUENCY_COLUMN] > 0]

    grouped = frame.groupby(WORD_COLUMN, sort=False)[FREQUENCY_COLUMN].sum()
    return {word: float(frequency) for word, frequency in grouped.items()}


def _load_subtlex_uk_from_xlsx_xml(path: Path) -> dict[str, float]:
    shared_strings = _xlsx_shared_strings(path)
    rows = _xlsx_rows(path, shared_strings)

    try:
        header = next(rows)
    except StopIteration as exc:
        raise ValueError("SUBTLEX-UK workbook is empty.") from exc

    try:
        word_index = header.index(WORD_COLUMN)
        frequency_index = header.index(FREQUENCY_COLUMN)
    except ValueError as exc:
        raise ValueError(
            f"SUBTLEX-UK workbook must contain {WORD_COLUMN!r} and {FREQUENCY_COLUMN!r} columns."
        ) from exc

    reference: dict[str, float] = {}
    for row in rows:
        if len(row) <= max(word_index, frequency_index):
            continue
        word = str(row[word_index]).strip()
        if not word:
            continue
        try:
            frequency = float(row[frequency_index])
        except (TypeError, ValueError):
            continue
        if frequency <= 0:
            continue
        reference[word] = reference.get(word, 0.0) + frequency

    return reference


def _xlsx_shared_strings(path: Path) -> list[str]:
    strings: list[str] = []
    with ZipFile(path) as archive:
        with archive.open("xl/sharedStrings.xml") as handle:
            for event, element in iterparse(handle, events=("end",)):
                if element.tag.endswith("}si"):
                    parts = [
                        text_element.text
                        for text_element in element.iter()
                        if text_element.tag.endswith("}t") and text_element.text
                    ]
                    strings.append("".join(parts))
                    element.clear()
    return strings


def _xlsx_rows(path: Path, shared_strings: list[str]):
    with ZipFile(path) as archive:
        with archive.open("xl/worksheets/sheet1.xml") as handle:
            for event, element in iterparse(handle, events=("end",)):
                if not element.tag.endswith("}row"):
                    continue

                values_by_column: dict[int, str] = {}
                for cell in element:
                    if not cell.tag.endswith("}c"):
                        continue
                    cell_reference = cell.attrib.get("r", "")
                    column = _xlsx_column_index(cell_reference)
                    if column is None:
                        continue
                    value = _xlsx_cell_value(cell, shared_strings)
                    values_by_column[column] = value

                if values_by_column:
                    max_column = max(values_by_column)
                    yield [values_by_column.get(index, "") for index in range(max_column + 1)]
                element.clear()


def _xlsx_cell_value(cell, shared_strings: list[str]) -> str:
    raw_value = ""
    for child in cell:
        if child.tag.endswith("}v") and child.text is not None:
            raw_value = child.text
            break

    if cell.attrib.get("t") == "s" and raw_value:
        return shared_strings[int(raw_value)]
    return raw_value


def _xlsx_column_index(cell_reference: str) -> int | None:
    letters = ""
    for character in cell_reference:
        if character.isalpha():
            letters += character
        else:
            break
    if not letters:
        return None

    index = 0
    for character in letters.upper():
        index = index * 26 + (ord(character) - ord("A") + 1)
    return index - 1
