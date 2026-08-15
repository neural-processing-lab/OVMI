#!/usr/bin/env python3
"""Validate and render the cross-study confound reference table."""

from __future__ import annotations

import argparse
import csv
import math
import re
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFOUNDS = PROJECT_ROOT / "data/confounds.csv"
DEFAULT_SYSTEMS = PROJECT_ROOT / "data/systems.csv"
DEFAULT_NOTES = PROJECT_ROOT / "data/confounds_notes.md"
DEFAULT_OUTPUT = PROJECT_ROOT / "tables/confounds.tex"
MISSING = "—"

FIELDS = (
    "recording_modality",
    "speech_type",
    "participants",
    "hours_per_participant",
    "task_structure",
    "vocabulary_provenance",
    "language_model",
    "calibration_burden",
    "split_discipline",
)
CSV_FIELDS = ("row_key", "system_id", "display_name", *FIELDS)
TRACE_RE = re.compile(r"<!--\s*trace:\s*([^\s.]+)\.([a-z_]+)\s*-->")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--confounds", type=Path, default=DEFAULT_CONFOUNDS)
    parser.add_argument("--systems", type=Path, default=DEFAULT_SYSTEMS)
    parser.add_argument("--notes", type=Path, default=DEFAULT_NOTES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != CSV_FIELDS:
            raise ValueError(f"{path} must have exactly these columns: {CSV_FIELDS}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"{path} contains no rows")
    for row_number, row in enumerate(rows, start=2):
        for field in CSV_FIELDS:
            if not row[field].strip():
                raise ValueError(
                    f"{path}:{row_number}: blank {field}; use the em dash {MISSING!r} "
                    "when the paper does not report a value"
                )
        if any(token in row[field].lower() for field in FIELDS for token in ("inferred", "estimated")):
            raise ValueError(
                f"{path}:{row_number}: inferred/estimated values are forbidden; use {MISSING!r}"
            )
    keys = [row["row_key"] for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError(f"{path} contains duplicate row_key values")
    return rows


def finite(value: str) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def point_name(system_id: str, system_name: str, variant: str) -> str:
    if system_id == "meg_masc_2023_v50":
        return "MEG-MASC 2023"
    if system_id.startswith("megxl_"):
        return "MEG-XL 2026"
    if system_id == "dascoli_libribrain100_s0_v50":
        return "LibriBrain100 2025"
    if system_id == "armeni_2022_v50":
        return "Armeni 2022"
    stem = system_name.replace(" et al.", "")
    return f"{stem} (isolated)" if variant == "neural" else f"{stem} (+LM)"


def expected_main_order(path: Path) -> list[tuple[str, str]]:
    """Reproduce Table 1's displayed point set and ordering from systems.csv."""
    with path.open(newline="", encoding="utf-8") as handle:
        systems = list(csv.DictReader(handle))
    points: list[dict[str, str | float]] = []
    for row in systems:
        if row["plot_eligible"].strip().lower() not in {"true", "1", "yes"}:
            continue
        group = "attempted" if row["speech_condition"] == "attempted" else "perceived"
        for variant, probability_field, score_field in (
            ("neural", "P_neural", "OVMI_neural_bits"),
            ("system", "P_system", "OVMI_system_bits"),
        ):
            if not finite(row[probability_field]):
                continue
            points.append({
                "row_key": f"{row['system_id']}:{variant}",
                "display_name": point_name(row["system_id"], row["system_name"], variant),
                "group": group,
                "score": float(row[score_field]),
            })
    ordered: list[tuple[str, str]] = []
    for group in ("attempted", "perceived"):
        members = [point for point in points if point["group"] == group]
        members.sort(key=lambda point: float(point["score"]), reverse=True)
        ordered.extend((str(point["row_key"]), str(point["display_name"])) for point in members)
    return ordered


def validate_main_order(rows: list[dict[str, str]], systems_path: Path) -> None:
    actual = [row["row_key"] for row in rows]
    expected = expected_main_order(systems_path)
    expected_keys_in_order = [key for key, _ in expected]
    if actual != expected_keys_in_order:
        actual_keys = set(actual)
        expected_keys = {key for key, _ in expected}
        raise ValueError(
            "confound rows do not exactly match Table 1\n"
            f"missing: {sorted(expected_keys - actual_keys)}\n"
            f"extra: {sorted(actual_keys - expected_keys)}\n"
            f"expected order: {[key for key, _ in expected]}\n"
            f"actual order: {actual}"
        )
    names = [row["display_name"] for row in rows]
    if len(names) != len(set(names)):
        raise ValueError("display_name values must identify repeated V/LM variants uniquely")


def validate_traces(rows: list[dict[str, str]], notes_path: Path) -> None:
    notes = notes_path.read_text(encoding="utf-8")
    traces = set(TRACE_RE.findall(notes))
    required = {(row["row_key"], field) for row in rows for field in FIELDS}
    missing = sorted(required - traces)
    stale = sorted(traces - required)
    if missing or stale:
        raise ValueError(
            f"source trace mismatch in {notes_path}\nmissing: {missing}\nstale: {stale}"
        )
    if "No populated entry is inferred" not in notes:
        raise ValueError(f"{notes_path} must state how inferred values are handled")


def latex_escape(value: str) -> str:
    replacements = {
        "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#",
        "_": r"\_", "{": r"\{", "}": r"\}",
        "—": r"\textemdash{}", "–": "--",
    }
    return "".join(replacements.get(character, character) for character in value)


def cell(value: str) -> str:
    return latex_escape(value)


def caption() -> str:
    return (
        "Cross-study confounds accompanying Table~\\ref{tab:main}. These factors are not "
        "controlled in the main comparison and are confounded with one another in the current "
        "literature: every attempted-speech system here is also invasive, clinical, and "
        "long-calibration. No entry in Table~\\ref{tab:main} should therefore be read as "
        "attributing a gap to recording modality. An em dash denotes information not reported; "
        "N/A denotes a sentence-level criterion that does not apply to an isolated-word result."
    )


def render(rows: list[dict[str, str]], output_path: Path) -> None:
    lines = [
        r"% Requires \usepackage{booktabs,tabularx,array}.",
        r"\begin{table*}[t]",
        rf"\caption{{{caption()}}}",
        r"\label{tab:confounds}",
        r"\centering",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{2pt}",
        r"\renewcommand{\arraystretch}{1.14}",
        r"\textit{Acquisition and participant factors}",
        r"\par\smallskip",
        r"\begin{tabularx}{\textwidth}{@{}>{\raggedright\arraybackslash}p{0.145\textwidth}>{\raggedright\arraybackslash}p{0.09\textwidth}>{\raggedright\arraybackslash}p{0.085\textwidth}>{\raggedright\arraybackslash}p{0.13\textwidth}>{\raggedright\arraybackslash}p{0.16\textwidth}X@{}}",
        r"\toprule",
        r"System & Modality & Speech & Participants & Data / participant & Calibration / fine-tuning burden \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(" & ".join([
            cell(row["display_name"]), cell(row["recording_modality"]),
            cell(row["speech_type"]), cell(row["participants"]),
            cell(row["hours_per_participant"]), cell(row["calibration_burden"]),
        ]) + r" \\")
    lines.extend([
        r"\bottomrule",
        r"\end{tabularx}",
        r"\par\medskip",
        r"\textit{Task and evaluation factors}",
        r"\par\smallskip",
        r"\begin{tabularx}{\textwidth}{@{}>{\raggedright\arraybackslash}p{0.145\textwidth}>{\raggedright\arraybackslash}p{0.19\textwidth}>{\raggedright\arraybackslash}p{0.205\textwidth}>{\raggedright\arraybackslash}p{0.215\textwidth}X@{}}",
        r"\toprule",
        r"System & Task structure & Vocabulary provenance & Language model & Sentence-level split discipline \\",
        r"\midrule",
    ])
    for row in rows:
        lines.append(" & ".join([
            cell(row["display_name"]), cell(row["task_structure"]),
            cell(row["vocabulary_provenance"]), cell(row["language_model"]),
            cell(row["split_discipline"]),
        ]) + r" \\")
    lines.extend([
        r"\bottomrule",
        r"\end{tabularx}",
        r"\end{table*}",
        "",
    ])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")


def report_sparsity(rows: list[dict[str, str]]) -> list[tuple[str, int]]:
    counts = Counter({field: sum(MISSING in row[field] for row in rows) for field in FIELDS})
    ranked = sorted(counts.items(), key=lambda item: (-item[1], FIELDS.index(item[0])))
    print(f"CHECK Table 1 row set/order: {len(rows)} rows match exactly")
    print("SPARSITY (cells containing an em dash; quote these counts in the reporting-checklist text):")
    for field, missing in ranked:
        print(f"  {field}: {missing}/{len(rows)} ({100 * missing / len(rows):.1f}%)")
    return ranked


def main() -> None:
    args = parse_args()
    rows = read_csv(args.confounds)
    validate_main_order(rows, args.systems)
    validate_traces(rows, args.notes)
    render(rows, args.output)
    report_sparsity(rows)
    print(f"CHECK source traces: {len(rows) * len(FIELDS)} cells covered by {args.notes}")
    print(f"Wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
