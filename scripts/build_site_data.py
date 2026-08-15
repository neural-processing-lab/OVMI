#!/usr/bin/env python3
"""Generate the static website's leaderboard data from the analysis pipeline."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import re
import sys

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = PROJECT_ROOT / "scripts"
SRC_DIR = PROJECT_ROOT / "src"
for directory in (SCRIPT_DIR, SRC_DIR):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

import make_main_table as table  # noqa: E402


DEFAULT_OUTPUT = PROJECT_ROOT / "site/data/leaderboard.json"
DEFAULT_FRONTIER = PROJECT_ROOT / "data/noiseless_frequency_frontier.csv"

REFERENCE_METADATA = {
    "subtlex": {
        "slug": "subtlex",
        "label": "Broad speech — SUBTLEX-UK",
        "short_label": "SUBTLEX-UK",
        "description": (
            "Broad spoken-English frequency norm derived from British film and "
            "television subtitles."
        ),
        "source_url": "https://doi.org/10.1080/17470218.2013.850521",
    },
    "conversation": {
        "slug": "switchboard",
        "label": "Conversation — Switchboard",
        "short_label": "Switchboard",
        "description": "Empirical unigram distribution from the 36-call NLTK sample.",
        "source_url": "https://www.nltk.org/nltk_data/",
    },
    "ucv": {
        "slug": "ucv",
        "label": "AAC / clinical — Universal Core Vocabulary (UCV)",
        "short_label": "UCV",
        "description": (
            "A 36-word AAC vocabulary weighted by SUBTLEX-UK frequency and "
            "renormalised within the set."
        ),
        "source_url": "https://praacticalaac.org/praactical/aac-vocabulary-lists/",
    },
    "narrative": {
        "slug": "sherlock",
        "label": "Narrative — Sherlock Holmes",
        "short_label": "Sherlock",
        "description": (
            "Empirical distribution of 3,550 target tokens in the held-out "
            "LibriBrain Sherlock test split."
        ),
        "source_url": "https://doi.org/10.1038/s41467-025-65499-0",
    },
}

PRIMARY_SOURCES = {
    "meg_masc_2023_v50": (
        "Gwilliams et al. (2023); local V=50 evaluation",
        "https://doi.org/10.1038/s41597-023-02752-5",
    ),
    "dascoli_libribrain100_s0_v50": (
        "d'Ascoli et al. (2025); local LibriBrain100 evaluation",
        "https://doi.org/10.1038/s41467-025-65499-0",
    ),
    "armeni_2022_v50": (
        "Armeni et al. (2022); local V=50 evaluation",
        "https://doi.org/10.1038/s41597-022-01382-7",
    ),
    "moses_2021_v50": (
        "Moses et al. (2021)",
        "https://doi.org/10.1056/NEJMoa2027540",
    ),
    "willett_2023_v50": (
        "Willett et al. (2023)",
        "https://doi.org/10.1038/s41586-023-06377-x",
    ),
    "willett_2023_v125k": (
        "Willett et al. (2023)",
        "https://doi.org/10.1038/s41586-023-06377-x",
    ),
    "card_2024_v125k": (
        "Card et al. (2024)",
        "https://doi.org/10.1056/NEJMoa2314132",
    ),
}

# Decoder provenance is shown explicitly for leaderboard rows whose scores
# were obtained with a decoder developed in a different study.
DECODER_METHODS = {
    "meg_masc_2023_v50": "MEG-XL",
    "dascoli_libribrain100_s0_v50": "d’Ascoli et al. (2025)",
    "armeni_2022_v50": "d’Ascoli et al. (2025)",
}

PROGRESS_YEAR_OVERRIDES = {
    "dascoli_libribrain100_s0_v50": 2025,
    "armeni_2022_v50": 2025,
    "meg_masc_2023_v50": 2026,
}

UNCERTAINTY_LABELS = {
    "wilson95": "95% Wilson interval",
    "bootstrap95": "Published 95% bootstrap interval",
    "published95": "Published 95% interval",
    "seed_sem": "Mean ± one SEM across seeds",
    "none": "Not reported",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--systems", type=Path, default=table.DEFAULT_SYSTEMS)
    parser.add_argument(
        "--references-dir", type=Path, default=table.DEFAULT_REFERENCES,
    )
    parser.add_argument(
        "--predictions-dir", type=Path, default=table.DEFAULT_PREDICTIONS,
    )
    parser.add_argument("--cmudict", type=Path, default=table.DEFAULT_CMUDICT)
    parser.add_argument("--armeni-text", type=Path, default=table.DEFAULT_ARMENI_TEXT)
    parser.add_argument(
        "--meg-masc-vocabulary", type=Path,
        default=table.DEFAULT_MEG_MASC_VOCABULARY,
    )
    parser.add_argument("--frontier", type=Path, default=DEFAULT_FRONTIER)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def metric_metadata(point: table.SystemPoint) -> dict[str, object]:
    if point.probability_source == "w":
        return {
            "type": "wer",
            "label": "WER",
            "reported_value": 1.0 - point.probability,
            "p_correct": point.probability,
            "p_is_lower_bound": point.lower_bound,
        }
    if point.probability_source == "b":
        metric_type = "balanced_top1_accuracy"
        label = "Balanced top-1 accuracy"
    elif point.system_id.startswith("meg_masc_"):
        metric_type = "top1_accuracy"
        label = "Top-1 accuracy"
    else:
        metric_type = "isolated_word_accuracy"
        label = "Isolated-word accuracy"
    return {
        "type": metric_type,
        "label": label,
        "reported_value": point.probability,
        "p_correct": point.probability,
        "p_is_lower_bound": False,
    }


def source_urls(source: str) -> list[str]:
    urls = re.findall(r"https?://[^;\s\"]+", source)
    return [url.rstrip(".,)") for url in dict.fromkeys(urls)]


def finite(value: object) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def build_reference_result(
    point_index: int,
    reference_key: str,
    scores: dict[tuple[int, str], table.CellScore],
    uncertainties: dict[tuple[int, str], table.CellUncertainty],
) -> dict[str, object]:
    cell = scores[(point_index, reference_key)]
    interval = uncertainties.get((point_index, reference_key))
    result: dict[str, object] = {
        "coverage": cell.coverage,
        "in_vocab_information_bits": cell.in_vocab_information,
        "supported_vocab_entropy_bits": cell.own_entropy,
        "ovmi_bits": cell.score,
        "ovmi_percent": cell.percentage,
    }
    if interval is None:
        result["uncertainty"] = None
    else:
        result["uncertainty"] = {
            "kind": interval.kind,
            "label": UNCERTAINTY_LABELS.get(interval.kind, interval.kind),
            "low_bits": interval.low.score,
            "high_bits": interval.high.score,
            "low_percent": interval.low.percentage,
            "high_percent": interval.high.percentage,
        }
    return result


def build_payload(args: argparse.Namespace) -> dict[str, object]:
    systems = pd.read_csv(args.systems)
    systems = systems.loc[systems["plot_eligible"].astype(bool)].copy()
    systems_by_id = {
        str(row["system_id"]): row for _, row in systems.iterrows()
    }
    references, entropies = table.load_references(args.references_dir)
    vocabularies = table.reconstruct_vocabularies(
        systems, references, args.predictions_dir, args.cmudict,
        args.armeni_text, args.meg_masc_vocabulary,
    )
    points = table.expand_system_points(systems)
    scores = table.score_all(points, vocabularies, references, entropies)
    uncertainties = table.score_uncertainties(
        points, vocabularies, references, entropies
    )
    table.validate_scores(points, scores)

    reference_payload = {}
    for key in table.MAIN_REFERENCE_KEYS:
        metadata = dict(REFERENCE_METADATA[key])
        metadata["entropy_bits"] = entropies[key]
        reference_payload[key] = metadata

    system_payload = []
    for point_index, point in enumerate(points):
        row = systems_by_id[point.system_id]
        citation, primary_url = PRIMARY_SOURCES[point.system_id]
        full_source = str(row["source"])
        group_key = (
            "attempted_invasive" if point.group == "attempted"
            else "perceived_noninvasive"
        )
        system_payload.append({
            "id": f"{point.system_id}-{point.probability_source}",
            "system_id": point.system_id,
            "variant": (
                "language_model" if point.probability_source == "w"
                else "decoder"
            ),
            "system": point.display_name,
            "year": int(row["year"]),
            "progress_year": PROGRESS_YEAR_OVERRIDES.get(
                point.system_id, int(row["year"])
            ),
            "group": group_key,
            "setting": str(row["speech_condition"]),
            "modality": str(row["modality"]),
            "decoder_method": DECODER_METHODS.get(point.system_id),
            "task": str(row["task"]),
            "vocabulary_size": point.vocabulary_size,
            "vocabulary_provenance": str(row["vocabulary_kind"]),
            "metric": metric_metadata(point),
            "uncertainty_method": str(row["ci_method"]),
            "source": {
                "citation": citation,
                "primary_url": primary_url,
                "note": full_source,
                "urls": source_urls(full_source),
            },
            "references": {
                key: build_reference_result(
                    point_index, key, scores, uncertainties
                )
                for key in table.MAIN_REFERENCE_KEYS
            },
        })

    frontier_payload: dict[str, list[dict[str, float | int]]] = {}
    if args.frontier.exists():
        frontier = pd.read_csv(args.frontier)
        subtlex = frontier.loc[frontier["reference"] == "subtlex-uk"]
        frontier_payload["subtlex"] = [
            {
                "vocabulary_size": int(row["V"]),
                "coverage": float(row["coverage"]),
                "in_vocab_information_bits": float(row["H_pS_bits"]),
            }
            for _, row in subtlex.iterrows()
        ]

    return {
        "schema_version": 1,
        "provenance": {
            "generator": "scripts/build_site_data.py",
            "systems": "data/systems.csv",
            "references": "data/references/*.csv",
            "scoring_pipeline": "scripts/make_main_table.py",
            "note": (
                "Values are generated from the same vocabulary reconstruction, "
                "reference distributions, and uncertainty propagation used by "
                "the paper tables."
            ),
        },
        "references": reference_payload,
        "systems": system_payload,
        "top_frequency_curves": frontier_payload,
    }


def validate_payload(payload: dict[str, object]) -> None:
    references = payload["references"]
    systems = payload["systems"]
    assert isinstance(references, dict) and isinstance(systems, list)
    if set(references) != set(table.MAIN_REFERENCE_KEYS):
        raise AssertionError("Site data does not contain the four main references")
    if len(systems) != 9:
        raise AssertionError(f"Expected nine leaderboard rows; got {len(systems)}")
    for system in systems:
        if set(system["references"]) != set(table.MAIN_REFERENCE_KEYS):
            raise AssertionError(f"Missing reference score for {system['id']}")
        for result in system["references"].values():
            error = abs(
                result["ovmi_bits"]
                - result["coverage"] * result["in_vocab_information_bits"]
            )
            if error > table.CHECK_TOLERANCE:
                raise AssertionError(f"OVMI factorisation failed for {system['id']}")


def main() -> None:
    args = parse_args()
    payload = build_payload(args)
    validate_payload(payload)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {len(payload['systems'])} leaderboard rows to {args.output}")


if __name__ == "__main__":
    main()
