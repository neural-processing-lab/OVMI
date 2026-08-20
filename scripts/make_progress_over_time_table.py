#!/usr/bin/env python3
"""Emit a compact LaTeX table for the progress-over-time figure points."""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import sys

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from make_progress_over_time_figure import (  # noqa: E402
    DEFAULT_SYSTEMS,
    POINT_SPECS,
)


DEFAULT_OUTPUT = PROJECT_ROOT / "tables/progress_over_time_results.tex"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--systems", type=Path, default=DEFAULT_SYSTEMS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def format_vocab(value: float) -> str:
    integer = int(value)
    return f"{integer:,}"


def latex_label(label: str) -> str:
    without_vocab = re.sub(r" \((?:V=)?[\d,]+k?\)$", "", label)
    return without_vocab.replace(" - ", "--")


def build_rows(systems_path: Path = DEFAULT_SYSTEMS) -> list[dict[str, object]]:
    systems = pd.read_csv(systems_path)
    rows = []
    for system_id, variant, label, year, *_ in POINT_SPECS:
        matches = systems.loc[systems["system_id"] == system_id]
        if len(matches) != 1:
            raise ValueError(f"Expected exactly one row for {system_id}")
        source = matches.iloc[0]
        values = {
            "label": label,
            "year": year,
            "V": float(source["V"]),
            "P": float(source[f"P_{variant}"]),
            "coverage": float(source["coverage"]),
            "within_mi": float(source[f"I_invocab_{variant}_bits"]),
            "ovmi": float(source[f"OVMI_{variant}_bits"]),
            "entropy": float(source["reference_entropy_bits"]),
            "lower_bound": bool(
                variant == "system" and source["P_system_is_lower_bound"]
            ),
        }
        numeric = [
            values[key]
            for key in ("V", "P", "coverage", "within_mi", "ovmi", "entropy")
        ]
        if not all(np.isfinite(value) for value in numeric):
            raise ValueError(f"Missing table value for {system_id}:{variant}")
        values["normalised_percent"] = (
            100.0 * values["ovmi"] / values["entropy"]
        )
        rows.append(values)
    return rows


def render_table(rows: list[dict[str, object]]) -> str:
    entropies = np.asarray([row["entropy"] for row in rows], dtype=float)
    if np.ptp(entropies) > 1e-10:
        raise AssertionError("All table rows must use one SUBTLEX-UK entropy")

    body = []
    for row in rows:
        reported_p = f"{100.0 * row['P']:.1f}\\%"
        if row["lower_bound"]:
            reported_p = f"$\\geq {reported_p}$"
        body.append(
            " & ".join([
                latex_label(str(row["label"])),
                format_vocab(float(row["V"])),
                reported_p,
                f"{row['coverage']:.3f}",
                f"{row['within_mi']:.3f}",
                f"{row['ovmi']:.3f}",
                f"{row['normalised_percent']:.1f}\\%",
            ]) + r" \\"
        )

    entropy = float(entropies[0])
    lines = [
        r"\begin{table}[t]",
        r"  \centering",
        (
            r"  \caption{Reported operating points under the SUBTLEX-UK "
            f"reference ($H(p)={entropy:.2f}$ bits). Coverage is $C(S)$; "
            r"within-vocabulary MI and OVMI are in bits. Values of $P$ shown "
            r"with $\geq$ are WER-derived lower bounds.}"
        ),
        r"  \label{tab:progress-results}",
        r"  \small",
        r"  \begin{tabular}{lrrrrrr}",
        r"    \toprule",
        (
            r"    System & $V$ & reported $P$ & Coverage & within-vocab MI "
            r"& OVMI & OVMI/$H(p)$ \\"
        ),
        r"    \midrule",
        *[f"    {line}" for line in body],
        r"    \bottomrule",
        r"  \end{tabular}",
        r"\end{table}",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    rows = build_rows(args.systems)
    latex = render_table(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(latex, encoding="utf-8")
    print(f"Wrote {args.output}")
    print(f"Rows: {len(rows)}")


if __name__ == "__main__":
    main()
