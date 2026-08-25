"""Compute OVMI for MEG-XL named 50-word vocabularies.

Usage:
    PYTHONPATH=src python3 tests/compute_megxl_named_vocab_ovmi.py

Optional:
    PYTHONPATH=src python3 tests/compute_megxl_named_vocab_ovmi.py \
        --subtlex-path /path/to/SUBTLEX-UK.xlsx
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ovmi import load_subtlex_uk, ovmi


DEFAULT_CONFIG = (
    Path.home()
    / "MEG-XL"
    / "configs"
    / "eval_criss_cross_word_classification_libribrain100.yaml"
)


def main() -> None:
    args = parse_args()
    named_sets = read_named_retrieval_sets(args.config)
    reference = load_reference(args.subtlex_path, args.cache_dir)

    print(f"SUBTLEX-UK words: {len(reference):,}")
    print(f"Macro accuracy: {args.accuracy:.6f}")
    print()

    for name in ("datafit50", "moses50"):
        vocabulary = named_sets[name]
        details = ovmi(reference, vocabulary, accuracy=args.accuracy, return_details=True)
        missing = [word for word in vocabulary if word not in reference]

        print(name)
        print(f"  vocabulary_size:        {len(vocabulary)}")
        print(f"  missing_from_reference: {missing}")
        print(f"  coverage:               {details.coverage:.12f}")
        print(f"  in_vocab_information:   {details.in_vocab_information:.12f}")
        print(f"  ovmi:                   {details.score:.12f}")
        print()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="MEG-XL eval YAML containing evaluation.named_retrieval_sets.",
    )
    parser.add_argument(
        "--accuracy",
        type=float,
        default=0.02,
        help="Homogeneous scalar macro accuracy. Chance for 50 words is 0.02.",
    )
    parser.add_argument(
        "--subtlex-path",
        type=Path,
        default=None,
        help="Optional path to a local SUBTLEX-UK .xlsx file.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("/private/tmp/ovmi-cache"),
        help="Cache directory used if SUBTLEX-UK must be downloaded.",
    )
    return parser.parse_args()


def load_reference(subtlex_path: Path | None, cache_dir: Path) -> dict[str, float]:
    if subtlex_path is not None:
        return load_subtlex_uk(path=subtlex_path)
    return load_subtlex_uk(cache_dir=cache_dir)


def read_named_retrieval_sets(path: Path) -> dict[str, list[str]]:
    """Read the simple named_retrieval_sets block without requiring PyYAML."""

    lines = path.expanduser().read_text(encoding="utf-8").splitlines()
    sets: dict[str, list[str]] = {}
    current_name: str | None = None
    in_block = False

    for raw_line in lines:
        stripped = raw_line.strip()
        indent = len(raw_line) - len(raw_line.lstrip(" "))

        if stripped == "named_retrieval_sets:":
            in_block = True
            current_name = None
            continue

        if not in_block:
            continue

        if indent <= 2 and stripped and not stripped.startswith("-"):
            break

        if indent == 4 and stripped.endswith(":"):
            current_name = stripped[:-1]
            sets[current_name] = []
            continue

        if indent == 6 and stripped.startswith("- ") and current_name is not None:
            sets[current_name].append(_unquote_yaml_string(stripped[2:].strip()))

    required = {"datafit50", "moses50"}
    missing = required.difference(sets)
    if missing:
        raise ValueError(f"Missing named retrieval sets in {path}: {sorted(missing)!r}")

    return sets


def _unquote_yaml_string(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


if __name__ == "__main__":
    main()
