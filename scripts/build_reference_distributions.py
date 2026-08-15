#!/usr/bin/env python3
"""Build the fixed lexical reference distributions used in the paper table.

The outputs are deliberately simple two-column CSV files (``word,weight``), so
the headline table can be regenerated without executing the exploratory
notebook.  We retain counts/frequencies rather than probabilities because OVMI
normalises every reference internally.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import math
import sys
from collections import Counter
from pathlib import Path
from urllib.request import urlretrieve
from zipfile import ZipFile

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ovmi import load_subtlex_uk  # noqa: E402
from ovmi.core import entropy  # noqa: E402

from build_systems_csv import MOSES_WORDS, normalize_word  # noqa: E402


DEFAULT_SUBTLEX = PROJECT_ROOT / "experiments/data/cache/SUBTLEX-UK.xlsx"
DEFAULT_PREDICTIONS = PROJECT_ROOT / "experiments/data/ovmi-predictions"
DEFAULT_SWITCHBOARD = PROJECT_ROOT / "experiments/data/cache/switchboard.zip"
DEFAULT_BNC_CONTEXT_GOVERNED = PROJECT_ROOT / "experiments/data/cache/bnc-cg.num.gz"
DEFAULT_BNC_DEMOGRAPHIC = PROJECT_ROOT / "experiments/data/cache/bnc-demog.num.gz"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data/references"
SWITCHBOARD_URL = (
    "https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/"
    "packages/corpora/switchboard.zip"
)
SWITCHBOARD_SHA256 = "6a1a22b659e2fe616129addab0e7967335e67c7dae6a6e63be10778dd0455d06"
BNC_CONTEXT_GOVERNED_URL = "https://kilgarriff.co.uk/BNClists/cg.num.gz"
BNC_DEMOGRAPHIC_URL = "https://kilgarriff.co.uk/BNClists/demog.num.gz"
BNC_CONTEXT_GOVERNED_SHA256 = "0b2452e85958c706ddd48a29e87a3f72453311bd5e9caa4e1f8a100148b17c67"
BNC_DEMOGRAPHIC_SHA256 = "a3ecf651a878c163ce6646166cddd6beb4959f85012c93f1c7538e0ff572bdaf"

UNIVERSAL_CORE_WORDS = """
all in some can it stop different like that do look turn finished make up get
more want go not what good on when he open where help put who here same why i
she you
""".split()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subtlex", type=Path, default=DEFAULT_SUBTLEX)
    parser.add_argument("--predictions-dir", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--switchboard", type=Path, default=DEFAULT_SWITCHBOARD)
    parser.add_argument(
        "--bnc-context-governed", type=Path, default=DEFAULT_BNC_CONTEXT_GOVERNED
    )
    parser.add_argument("--bnc-demographic", type=Path, default=DEFAULT_BNC_DEMOGRAPHIC)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def normalized_subtlex(path: Path) -> dict[str, float]:
    result: dict[str, float] = {}
    for word, frequency in load_subtlex_uk(path=path).items():
        key = normalize_word(word)
        result[key] = result.get(key, 0.0) + float(frequency)
    return result


def restricted_subtlex(
    subtlex: dict[str, float], vocabulary: list[str]
) -> dict[str, float]:
    result = {
        normalize_word(word): float(subtlex.get(normalize_word(word), 0.0))
        for word in vocabulary
    }
    missing = sorted(word for word, weight in result.items() if weight <= 0)
    if missing:
        raise ValueError(f"Words absent from SUBTLEX-UK: {missing}")
    return result


def sherlock_test_counts(predictions_dir: Path) -> dict[str, float]:
    """Return the empirical target-word distribution of the LibriBrain test chapter."""

    target_arrays: list[np.ndarray] = []
    for run_dir in sorted(path for path in predictions_dir.iterdir() if path.is_dir()):
        archive_path = run_dir / "test_predictions_best.npz"
        if not archive_path.exists():
            continue
        with np.load(archive_path, allow_pickle=True) as archive:
            targets = np.asarray(
                [normalize_word(word) for word in archive["target_word"]], dtype=str
            )
        target_arrays.append(targets)
    if not target_arrays:
        raise FileNotFoundError(f"No MEG-XL test archives found in {predictions_dir}")
    first = target_arrays[0]
    if any(not np.array_equal(first, other) for other in target_arrays[1:]):
        raise ValueError("MEG-XL runs do not contain the same LibriBrain test targets")
    return {word: float(count) for word, count in Counter(first).items()}


def ensure_switchboard(path: Path) -> None:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        urlretrieve(SWITCHBOARD_URL, path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != SWITCHBOARD_SHA256:
        raise ValueError(
            f"Unexpected Switchboard archive digest {digest}; expected {SWITCHBOARD_SHA256}"
        )


def _switchboard_lexeme(word: str, tag: str) -> str | None:
    """Normalise one token from the NLTK Switchboard sample's tagged transcript."""

    word = normalize_word(word)
    tag = tag.upper()
    contractions = {
        ("n't", "RB"): "not",
        ("'re", "VBP"): "are",
        ("'ve", "VBP"): "have",
        ("'ll", "MD"): "will",
        ("'m", "BEM"): "am",
        ("'d", "MD"): "would",
        ("'d", "HVD"): "had",
        ("'s", "BES"): "is",
        ("'s", "HVS"): "has",
    }
    if (word, tag) in contractions:
        return contractions[(word, tag)]
    if word.startswith("'") or tag in {",", ".", ":", "``", "''", "POS", "XX"}:
        return None
    if word.endswith("-") or not any(character.isalpha() for character in word):
        return None
    if not all(character.isalpha() or character in {"'", "-"} for character in word):
        return None
    return word


def switchboard_counts(path: Path) -> dict[str, float]:
    """Count lexical tokens in the 36-call NLTK Switchboard corpus sample."""

    ensure_switchboard(path)
    counts: Counter[str] = Counter()
    with ZipFile(path) as archive:
        with archive.open("switchboard/tagged") as handle:
            for raw_line in handle:
                line = raw_line.decode("utf-8").strip()
                for tagged_token in line.split()[1:]:
                    if "/" not in tagged_token:
                        continue
                    word, tag = tagged_token.rsplit("/", 1)
                    lexeme = _switchboard_lexeme(word, tag)
                    if lexeme:
                        counts[lexeme] += 1
    if not counts:
        raise ValueError("Switchboard parsing yielded no lexical tokens")
    return {word: float(count) for word, count in counts.items()}


def ensure_download(path: Path, url: str, expected_sha256: str) -> None:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        urlretrieve(url, path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != expected_sha256:
        raise ValueError(
            f"Unexpected digest for {path}: {digest}; expected {expected_sha256}"
        )


def _bnc_lexeme(word: str, tag: str) -> str | None:
    word = normalize_word(word)
    tag = tag.casefold()
    contractions = {
        "n't": "not",
        "'re": "are",
        "'ve": "have",
        "'ll": "will",
        "'m": "am",
    }
    if word in contractions:
        return contractions[word]
    if word == "'s":
        return "is" if tag.startswith("v") else None
    if word == "'d":
        return "had" if tag.startswith("vh") else "would"
    if word.startswith("'") or "_" in word or word.endswith("-"):
        return None
    if not word or not all(character.isalpha() or character == "-" for character in word):
        return None
    return word


def bnc_spoken_counts(context_governed_path: Path, demographic_path: Path) -> dict[str, float]:
    """Aggregate the context-governed and demographic BNC spoken lists."""

    ensure_download(
        context_governed_path,
        BNC_CONTEXT_GOVERNED_URL,
        BNC_CONTEXT_GOVERNED_SHA256,
    )
    ensure_download(
        demographic_path,
        BNC_DEMOGRAPHIC_URL,
        BNC_DEMOGRAPHIC_SHA256,
    )
    counts: Counter[str] = Counter()
    for path in (context_governed_path, demographic_path):
        with gzip.open(path, mode="rt", encoding="latin-1") as handle:
            for line_number, line in enumerate(handle, start=1):
                fields = line.split()
                if len(fields) != 4:
                    raise ValueError(f"Malformed BNC row {path}:{line_number}: {line!r}")
                frequency, word, tag, _file_count = fields
                lexeme = _bnc_lexeme(word, tag)
                if lexeme:
                    counts[lexeme] += int(frequency)
    if not counts:
        raise ValueError("BNC spoken parsing yielded no lexical tokens")
    return {word: float(count) for word, count in counts.items()}


def write_reference(path: Path, reference: dict[str, float]) -> None:
    if not reference or any(not math.isfinite(value) or value <= 0 for value in reference.values()):
        raise ValueError(f"Reference {path.stem} must have finite positive weights")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["word", "weight"])
        for word in sorted(reference):
            writer.writerow([word, f"{reference[word]:.12g}"])


def reference_entropy(reference: dict[str, float]) -> float:
    return entropy(np.asarray(list(reference.values()), dtype=np.float64))


def main() -> None:
    args = parse_args()
    subtlex = normalized_subtlex(args.subtlex)
    references = {
        "subtlex_uk": subtlex,
        "switchboard_conversational": switchboard_counts(args.switchboard),
        "ucv_aac": restricted_subtlex(subtlex, UNIVERSAL_CORE_WORDS),
        "sherlock_libribrain_test": sherlock_test_counts(args.predictions_dir),
        "individual_target_moses": restricted_subtlex(subtlex, MOSES_WORDS),
        "bnc_spoken": bnc_spoken_counts(
            args.bnc_context_governed, args.bnc_demographic
        ),
    }
    for name, reference in references.items():
        output_path = args.output_dir / f"{name}.csv"
        write_reference(output_path, reference)
        print(
            f"Wrote {len(reference):,} types to {output_path}; "
            f"H(p)={reference_entropy(reference):.12f} bits"
        )


if __name__ == "__main__":
    main()
