#!/usr/bin/env python3
"""Generate the paper's headline cross-reference OVMI table.

The script expands each plot-eligible record in ``data/systems.csv`` into the
same neural-only and language-model-assisted points shown in the contour
figure, reconstructs the documented vocabulary, and evaluates that fixed
system against every lexical reference in ``data/references``.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
SCRIPT_DIR = PROJECT_ROOT / "scripts"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from ovmi import ovmi  # noqa: E402
from ovmi.core import OVMIResult, entropy  # noqa: E402

from build_systems_csv import (  # noqa: E402
    DASCOLI_LIBRIBRAIN_WORDS,
    MOSES_WORDS,
    cmudict_words,
    normalize_word,
    top_words_from_plaintext,
)


DEFAULT_SYSTEMS = PROJECT_ROOT / "data/systems.csv"
DEFAULT_REFERENCES = PROJECT_ROOT / "data/references"
DEFAULT_PREDICTIONS = PROJECT_ROOT / "experiments/data/ovmi-predictions"
DEFAULT_CMUDICT = PROJECT_ROOT / "experiments/data/cache/cmudict.dict"
DEFAULT_ARMENI_TEXT = PROJECT_ROOT / "experiments/data/cache/armeni_advs.txt"
DEFAULT_MEG_MASC_VOCABULARY = (
    PROJECT_ROOT / "data/vocabularies/meg_masc_2023_v50.csv"
)
DEFAULT_TANG_VOCABULARY = PROJECT_ROOT / "data/vocabularies/tang_decoder_vocab.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "tables/main_table.tex"
DEFAULT_NO_SPECIALISATION_GAP_OUTPUT = (
    PROJECT_ROOT / "tables/main_table_no_specialisation_gap.tex"
)
DEFAULT_BITS_ONLY_OUTPUT = PROJECT_ROOT / "tables/main_table_bits_only.tex"
DEFAULT_APPENDIX_OUTPUT = PROJECT_ROOT / "tables/main_table_individual_target.tex"
DEFAULT_APPENDIX_BITS_ONLY_OUTPUT = (
    PROJECT_ROOT / "tables/main_table_individual_target_bits_only.tex"
)
CHECK_TOLERANCE = 1e-10


@dataclass(frozen=True)
class ReferenceSpec:
    key: str
    filename: str
    header: str


REFERENCES = (
    ReferenceSpec("subtlex", "subtlex_uk.csv", r"\shortstack{Broad spoken\\SUBTLEX--UK}"),
    ReferenceSpec("conversation", "switchboard_conversational.csv", r"\shortstack{Conversational\\Switchboard}"),
    ReferenceSpec("ucv", "ucv_aac.csv", r"\shortstack{AAC / clinical\\UCV}"),
    ReferenceSpec("narrative", "sherlock_libribrain_test.csv", r"\shortstack{Narrative prose\\Sherlock}"),
    ReferenceSpec("individual", "individual_target_moses.csv", r"\shortstack{Individual target\\Moses set}"),
)
MAIN_REFERENCE_KEYS = ("subtlex", "conversation", "ucv", "narrative")


@dataclass(frozen=True)
class SystemPoint:
    system_id: str
    display_name: str
    group: str
    vocabulary_size: int
    probability: float
    probability_source: str
    csv_ovmi: float
    lower_bound: bool
    uncertainty_kind: str
    probability_low: float | None
    probability_high: float | None


@dataclass(frozen=True)
class CellScore:
    score: float
    percentage: float
    coverage: float
    in_vocab_information: float
    own_entropy: float


@dataclass(frozen=True)
class CellUncertainty:
    kind: str
    low: CellScore
    high: CellScore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--systems", type=Path, default=DEFAULT_SYSTEMS)
    parser.add_argument("--references-dir", type=Path, default=DEFAULT_REFERENCES)
    parser.add_argument("--predictions-dir", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--cmudict", type=Path, default=DEFAULT_CMUDICT)
    parser.add_argument("--armeni-text", type=Path, default=DEFAULT_ARMENI_TEXT)
    parser.add_argument(
        "--meg-masc-vocabulary", type=Path,
        default=DEFAULT_MEG_MASC_VOCABULARY,
    )
    parser.add_argument(
        "--tang-vocabulary", type=Path, default=DEFAULT_TANG_VOCABULARY,
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--no-specialisation-gap-output", type=Path,
        default=DEFAULT_NO_SPECIALISATION_GAP_OUTPUT,
    )
    parser.add_argument(
        "--bits-only-output", type=Path, default=DEFAULT_BITS_ONLY_OUTPUT,
    )
    parser.add_argument(
        "--appendix-output", type=Path, default=DEFAULT_APPENDIX_OUTPUT,
        help="Appendix table containing the individual-target reference column.",
    )
    parser.add_argument(
        "--appendix-bits-only-output", type=Path,
        default=DEFAULT_APPENDIX_BITS_ONLY_OUTPUT,
    )
    return parser.parse_args()


def load_reference(path: Path) -> dict[str, float]:
    # Lexical strings such as "null" and "nan" are valid corpus words, not
    # missing-value sentinels.
    frame = pd.read_csv(path, keep_default_na=False)
    if list(frame.columns) != ["word", "weight"]:
        raise ValueError(f"{path} must contain exactly word,weight columns")
    if frame["word"].duplicated().any():
        raise ValueError(f"{path} contains duplicate words")
    weights = pd.to_numeric(frame["weight"], errors="coerce")
    if weights.isna().any() or (weights <= 0).any():
        raise ValueError(f"{path} contains invalid weights")
    return {
        normalize_word(word): float(weight)
        for word, weight in zip(frame["word"].astype(str), weights)
    }


def load_references(directory: Path) -> tuple[dict[str, dict[str, float]], dict[str, float]]:
    references = {
        spec.key: load_reference(directory / spec.filename) for spec in REFERENCES
    }
    entropies = {
        key: entropy(np.asarray(list(reference.values()), dtype=np.float64))
        for key, reference in references.items()
    }
    return references, entropies


def megxl_frequency_vocabulary(
    predictions_dir: Path, subtlex: dict[str, float], max_v: int
) -> list[str]:
    """Recover the exact frequency-selected vocabulary used by the contour script."""

    selected_by_run: list[list[str]] = []
    for run_dir in sorted(path for path in predictions_dir.iterdir() if path.is_dir()):
        predictions_path = run_dir / "test_predictions_best.npz"
        vocabulary_path = run_dir / "vocab_embeddings.npz"
        if not predictions_path.exists() or not vocabulary_path.exists():
            continue
        with np.load(predictions_path, allow_pickle=True) as archive:
            targets = {normalize_word(word) for word in archive["target_word"]}
        with np.load(vocabulary_path, allow_pickle=True) as archive:
            decoder_words = {normalize_word(word) for word in archive["word"]}
        selected = sorted(
            (
                word
                for word in targets.intersection(decoder_words)
                if subtlex.get(word, 0.0) > 0
            ),
            key=lambda word: (-subtlex[word], word),
        )[:max_v]
        selected_by_run.append(selected)
    if not selected_by_run:
        raise FileNotFoundError(f"No MEG-XL prediction archives found in {predictions_dir}")
    first = selected_by_run[0]
    if len(first) < max_v:
        raise ValueError(f"MEG-XL vocabulary has {len(first)} words; need {max_v}")
    if any(first != other for other in selected_by_run[1:]):
        raise ValueError("MEG-XL runs do not yield the same frequency-selected vocabulary")
    return first


def load_ranked_vocabulary(path: Path, expected_size: int) -> list[str]:
    frame = pd.read_csv(path, keep_default_na=False)
    if list(frame.columns) != ["rank", "word", "count"]:
        raise ValueError(f"{path} must contain exactly rank,word,count columns")
    if frame["rank"].tolist() != list(range(1, expected_size + 1)):
        raise ValueError(f"{path} must contain consecutive ranks 1..{expected_size}")
    words = [normalize_word(word) for word in frame["word"]]
    if len(words) != expected_size or len(words) != len(set(words)):
        raise ValueError(f"{path} must contain {expected_size} unique words")
    return words


def reconstruct_vocabularies(
    systems: pd.DataFrame,
    references: dict[str, dict[str, float]],
    predictions_dir: Path,
    cmudict_path: Path,
    armeni_text_path: Path,
    meg_masc_vocabulary_path: Path = DEFAULT_MEG_MASC_VOCABULARY,
    tang_vocabulary_path: Path = DEFAULT_TANG_VOCABULARY,
) -> dict[str, list[str]]:
    megxl_rows = systems.loc[systems["system_id"].str.startswith("megxl_")]
    megxl_words: list[str] = []
    if not megxl_rows.empty:
        max_megxl_v = int(megxl_rows["V"].max())
        megxl_words = megxl_frequency_vocabulary(
            predictions_dir, references["subtlex"], max_megxl_v
        )
    meg_masc_words = load_ranked_vocabulary(meg_masc_vocabulary_path, 50)
    armeni_words, _ = top_words_from_plaintext(
        armeni_text_path, vocabulary_size=50
    )
    cmu_words = cmudict_words(cmudict_path)
    tang_words = json.loads(tang_vocabulary_path.read_text(encoding="utf-8"))

    vocabularies: dict[str, list[str]] = {}
    for _, row in systems.iterrows():
        system_id = str(row["system_id"])
        vocabulary_size = int(row["V"])
        if system_id.startswith("megxl_"):
            vocabulary = megxl_words[:vocabulary_size]
        elif system_id == "meg_masc_2023_v50":
            vocabulary = meg_masc_words
        elif system_id == "dascoli_libribrain100_s0_v50":
            vocabulary = list(DASCOLI_LIBRIBRAIN_WORDS)
        elif system_id == "armeni_2022_v50":
            vocabulary = armeni_words
        elif system_id == "tang_2023_v6867":
            vocabulary = tang_words
        elif system_id in {"moses_2021_v50", "willett_2023_v50"}:
            vocabulary = list(MOSES_WORDS)
        elif system_id in {"willett_2023_v125k", "card_2024_v125k"}:
            vocabulary = cmu_words
        else:
            raise ValueError(f"No documented vocabulary reconstruction for {system_id}")
        vocabulary = [normalize_word(word) for word in vocabulary]
        if len(vocabulary) != len(set(vocabulary)):
            raise ValueError(f"Vocabulary for {system_id} contains duplicates")
        # The papers call the two CMUdict lexica 125k; the local dictionary is
        # the documented proxy and its exact entry count need not equal 125,000.
        if vocabulary_size != 125_000 and len(vocabulary) != vocabulary_size:
            raise ValueError(
                f"Vocabulary for {system_id} has {len(vocabulary)} entries, "
                f"but systems.csv says V={vocabulary_size}"
            )
        vocabularies[system_id] = vocabulary
    return vocabularies


def _display_name(system_id: str, base_name: str, variant: str) -> str:
    year = {
        "meg_masc": "2023",
        "megxl": "2026",
        "dascoli": "2025",
        "armeni": "2022",
        "moses": "2021",
        "willett": "2023",
        "card": "2024",
    }
    if system_id == "meg_masc_2023_v50":
        return f"MEG-MASC {year['meg_masc']}"
    if system_id.startswith("megxl_"):
        return f"MEG-XL {year['megxl']}"
    if system_id == "dascoli_libribrain100_s0_v50":
        return "LibriBrain100 2025"
    if system_id == "armeni_2022_v50":
        return "Armeni 2022"
    if system_id == "tang_2023_v6867":
        return "Tang 2023"
    if variant == "neural":
        return f"{base_name.replace(' et al.', '')} (isolated)"
    return f"{base_name.replace(' et al.', '')} (+LM)"


def _finite_or_none(value: object) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def expand_system_points(systems: pd.DataFrame) -> list[SystemPoint]:
    points: list[SystemPoint] = []
    eligible = systems.loc[systems["plot_eligible"].astype(bool)].copy()
    for _, row in eligible.iterrows():
        group = "attempted" if row["speech_condition"] == "attempted" else "perceived"
        system_id = str(row["system_id"])
        for variant, p_column, ovmi_column, low_column, high_column, uncertainty_column in (
            (
                "neural", "P_neural", "OVMI_neural_bits", "P_neural_ci_low",
                "P_neural_ci_high", "uncertainty_neural",
            ),
            (
                "system", "P_system", "OVMI_system_bits", "P_system_ci_low",
                "P_system_ci_high", "uncertainty_system",
            ),
        ):
            if not math.isfinite(float(row[p_column])):
                continue
            if system_id.startswith("meg_masc_"):
                probability_source = "a"
            elif system_id.startswith(("megxl_", "dascoli_", "armeni_")):
                probability_source = "b"
            elif variant == "neural":
                probability_source = "a"
            else:
                probability_source = "w"
            lower_bound = bool(row["P_system_is_lower_bound"]) and variant == "system"
            probability_low = _finite_or_none(row.get(low_column))
            probability_high = _finite_or_none(row.get(high_column))
            uncertainty_kind = str(row.get(uncertainty_column, "none"))
            if uncertainty_kind == "nan":
                uncertainty_kind = "none"
            points.append(SystemPoint(
                system_id=system_id,
                display_name=_display_name(system_id, str(row["system_name"]), variant),
                group=group,
                vocabulary_size=int(row["V"]),
                probability=float(row[p_column]),
                probability_source=probability_source,
                csv_ovmi=float(row[ovmi_column]),
                lower_bound=lower_bound,
                uncertainty_kind=uncertainty_kind,
                probability_low=probability_low,
                probability_high=probability_high,
            ))
    return points


def score_cell(
    reference: dict[str, float], reference_entropy: float,
    vocabulary: list[str], probability: float,
) -> CellScore:
    result = ovmi(
        reference, vocabulary, accuracy=probability, return_details=True
    )
    noiseless = ovmi(reference, vocabulary, accuracy=1.0, return_details=True)
    assert isinstance(result, OVMIResult) and isinstance(noiseless, OVMIResult)
    percentage = 100.0 * result.score / reference_entropy
    return CellScore(
        score=result.score,
        percentage=percentage,
        coverage=result.coverage,
        in_vocab_information=result.in_vocab_information,
        own_entropy=noiseless.in_vocab_information,
    )


def score_all(
    points: list[SystemPoint], vocabularies: dict[str, list[str]],
    references: dict[str, dict[str, float]], entropies: dict[str, float],
) -> dict[tuple[int, str], CellScore]:
    scores: dict[tuple[int, str], CellScore] = {}
    for point_index, point in enumerate(points):
        vocabulary = vocabularies[point.system_id]
        for spec in REFERENCES:
            scores[(point_index, spec.key)] = score_cell(
                references[spec.key], entropies[spec.key], vocabulary, point.probability
            )
    return scores


def score_uncertainties(
    points: list[SystemPoint], vocabularies: dict[str, list[str]],
    references: dict[str, dict[str, float]], entropies: dict[str, float],
) -> dict[tuple[int, str], CellUncertainty]:
    uncertainties: dict[tuple[int, str], CellUncertainty] = {}
    missing = []
    at_or_below_chance = []
    monotonicity_checks = 0
    for point_index, point in enumerate(points):
        chance = 1.0 / point.vocabulary_size
        if point.probability <= chance:
            at_or_below_chance.append(
                (point.display_name, point.probability, chance)
            )
            continue
        if point.probability_low is None or point.probability_high is None:
            missing.append(point.display_name)
            continue
        if not (
            chance < point.probability_low <= point.probability
            <= point.probability_high <= 1.0
        ):
            raise AssertionError(
                f"Invalid P uncertainty endpoints for {point.display_name}: "
                f"chance={chance}, low={point.probability_low}, "
                f"P={point.probability}, high={point.probability_high}"
            )
        vocabulary = vocabularies[point.system_id]
        # Endpoints plus an interior point are sufficient to falsify the expected
        # monotone ordering without turning this table check into a curve sweep.
        grid = np.linspace(point.probability_low, point.probability_high, 3)
        for spec in REFERENCES:
            values = np.asarray([
                float(ovmi(references[spec.key], vocabulary, accuracy=probability))
                for probability in grid
            ])
            if np.any(np.diff(values) < -CHECK_TOLERANCE):
                raise AssertionError(
                    f"OVMI is not numerically monotone in P for "
                    f"{point.display_name} under {spec.key}"
                )
            low = score_cell(
                references[spec.key], entropies[spec.key], vocabulary,
                point.probability_low,
            )
            high = score_cell(
                references[spec.key], entropies[spec.key], vocabulary,
                point.probability_high,
            )
            uncertainties[(point_index, spec.key)] = CellUncertainty(
                kind=point.uncertainty_kind, low=low, high=high,
            )
            monotonicity_checks += 1

    print(
        f"CHECK monotonic uncertainty propagation: {monotonicity_checks} "
        "row-reference intervals passed"
    )
    print(f"At/below chance rows: {at_or_below_chance or 'none'}")
    print(f"Rows without uncertainty: {missing or 'none'}")
    if at_or_below_chance:
        raise AssertionError(
            "Cannot propagate above-chance monotone intervals for rows at/below chance"
        )
    expected_missing = [
        point.display_name for point in points
        if point.uncertainty_kind == "none_single_seed"
    ]
    if missing != expected_missing:
        raise AssertionError(
            f"Unexpected rows without uncertainty: {missing}; expected {expected_missing}"
        )
    return uncertainties


def home_turf(point: SystemPoint, reference_key: str) -> bool:
    caregiving = point.system_id in {"moses_2021_v50", "willett_2023_v50"}
    narrative = point.system_id.startswith(
        ("meg_masc_", "megxl_", "dascoli_", "armeni_", "tang_")
    )
    return (caregiving and reference_key in {"ucv", "individual"}) or (
        narrative and reference_key == "narrative"
    )


def design_reference(point: SystemPoint) -> str | None:
    if point.system_id in {"moses_2021_v50", "willett_2023_v50"}:
        return "individual"
    if point.system_id.startswith(
        ("meg_masc_", "megxl_", "dascoli_", "armeni_", "tang_")
    ):
        return "narrative"
    return None


def validate_scores(
    points: list[SystemPoint], scores: dict[tuple[int, str], CellScore]
) -> None:
    factor_errors = []
    entropy_excesses = []
    percentage_excesses = []
    contour_errors = []
    for point_index, point in enumerate(points):
        for spec in REFERENCES:
            cell = scores[(point_index, spec.key)]
            factor_errors.append(abs(cell.score - cell.coverage * cell.in_vocab_information))
            entropy_excesses.append(
                (point.display_name, spec.key, cell.score - cell.coverage * cell.own_entropy)
            )
            percentage_excesses.append((point.display_name, spec.key, cell.percentage - 100.0))
        contour_errors.append(
            abs(scores[(point_index, "subtlex")].score - point.csv_ovmi)
        )

    max_factor_error = max(factor_errors, default=0.0)
    max_entropy_excess = max((entry[2] for entry in entropy_excesses), default=-math.inf)
    max_percentage_excess = max((entry[2] for entry in percentage_excesses), default=-math.inf)
    max_contour_error = max(contour_errors, default=0.0)
    violations = [entry for entry in entropy_excesses if entry[2] > CHECK_TOLERANCE]
    percentage_violations = [entry for entry in percentage_excesses if entry[2] > CHECK_TOLERANCE]

    print(
        f"CHECK C(S)*I: {len(points) * len(REFERENCES)} cells; "
        f"max absolute error={max_factor_error:.3g}"
    )
    print(
        f"CHECK contour cross-check: {len(points)} SUBTLEX points; "
        f"max absolute error={max_contour_error:.3g}"
    )
    print(
        "CHECK own-vocabulary ceiling C(S)*H(p_S): "
        f"{len(violations)} violations; max excess={max_entropy_excess:.3g} bits"
    )
    print(
        f"CHECK normalised scores <=100%: {len(percentage_violations)} violations; "
        f"max excess={max_percentage_excess:.3g} percentage points"
    )
    if max_factor_error > CHECK_TOLERANCE:
        raise AssertionError("A table cell does not equal C(S)*I(X;Y | X in S)")
    if max_contour_error > 5e-10:
        raise AssertionError("A SUBTLEX table cell disagrees with data/systems.csv")
    if violations:
        raise AssertionError(f"Own-vocabulary entropy violations: {violations}")
    if percentage_violations:
        raise AssertionError(f"OVMI/H(p) above 100%: {percentage_violations}")


def spearman_report(
    points: list[SystemPoint], scores: dict[tuple[int, str], CellScore]
) -> pd.DataFrame:
    values = pd.DataFrame({
        spec.key: [scores[(index, spec.key)].score for index in range(len(points))]
        for spec in REFERENCES
    })
    ranks = values.rank(method="average", ascending=False)
    correlations = ranks.corr(method="pearson")
    print("Spearman rank correlations (OVMI; percentages give the same ranks):")
    print(correlations.to_string(float_format=lambda value: f"{value:.3f}"))
    identical = []
    for left_index, left in enumerate(REFERENCES):
        for right in REFERENCES[left_index + 1:]:
            if np.array_equal(ranks[left.key].to_numpy(), ranks[right.key].to_numpy()):
                identical.append((left.key, right.key))
    if identical:
        print(f"FLAG identical system rankings: {identical}")
    else:
        print("CHECK rank distinctness: no two reference columns rank all systems identically")
    return correlations


def key_uncertainty_report(
    points: list[SystemPoint], scores: dict[tuple[int, str], CellScore],
    uncertainties: dict[tuple[int, str], CellUncertainty],
) -> None:
    def values(system_id: str, probability_source: str) -> tuple[float, float, float]:
        matches = [
            index for index, point in enumerate(points)
            if point.system_id == system_id
            and point.probability_source == probability_source
        ]
        if len(matches) != 1:
            raise AssertionError(
                f"Expected one point for {system_id}/{probability_source}; got {matches}"
            )
        index = matches[0]
        cell = scores[(index, "subtlex")]
        interval = uncertainties[(index, "subtlex")]
        return cell.score, interval.low.score, interval.high.score

    libri = values("dascoli_libribrain100_s0_v50", "b")
    moses = values("moses_2021_v50", "a")
    headline_difference = abs(libri[0] - moses[0])
    libri_sem_half_width = max(libri[0] - libri[1], libri[2] - libri[0])
    moses_half_width = max(moses[0] - moses[1], moses[2] - moses[0])
    print(
        "KEY SUBTLEX LibriBrain100: "
        f"{libri[0]:.9f} bits, seed-SEM endpoints "
        f"[{libri[1]:.9f}, {libri[2]:.9f}]"
    )
    print(
        "KEY SUBTLEX Moses isolated: "
        f"{moses[0]:.9f} bits, 95% Wilson interval "
        f"[{moses[1]:.9f}, {moses[2]:.9f}]"
    )
    print(
        f"KEY headline difference: {headline_difference:.9f} bits; smaller than "
        f"LibriBrain100 SEM={libri_sem_half_width:.9f}: "
        f"{headline_difference < libri_sem_half_width}; smaller than Moses interval "
        f"half-width={moses_half_width:.9f}: "
        f"{headline_difference < moses_half_width}"
    )

    willett_isolated = values("willett_2023_v50", "a")
    willett_lm = values("willett_2023_v50", "w")
    separated = (
        willett_isolated[2] < willett_lm[1]
        or willett_lm[2] < willett_isolated[1]
    )
    print(
        "KEY SUBTLEX Willett isolated vs +LM: "
        f"{willett_isolated[0]:.9f} "
        f"[{willett_isolated[1]:.9f}, {willett_isolated[2]:.9f}] vs "
        f"{willett_lm[0]:.9f} "
        f"[{willett_lm[1]:.9f}, {willett_lm[2]:.9f}]; "
        f"non-overlapping={separated}"
    )


def latex_escape(value: str) -> str:
    replacements = {
        "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#",
        "_": r"\_", "{": r"\{", "}": r"\}",
    }
    return "".join(replacements.get(character, character) for character in value)


def format_vocabulary_size(value: int) -> str:
    return "125k" if value == 125_000 else str(value)


def format_probability(point: SystemPoint) -> str:
    lower_bound = r"\dagger" if point.lower_bound else ""
    return (
        f"{100.0 * point.probability:.1f}"
        rf"\%$^{{\mathrm{{{point.probability_source}}}{lower_bound}}}$"
    )


def format_cell(
    cell: CellScore, uncertainty: CellUncertainty | None, aligned: bool,
    include_percentage: bool,
) -> str:
    bit_value = rf"${cell.score:.3f}$"
    if uncertainty is None:
        bit_value = rf"${cell.score:.3f}^{{\ddagger}}$"
        uncertainty_value = None
    elif uncertainty.kind in {"seed_sem", "participant_sem"}:
        lower = max(0.0, cell.score - uncertainty.low.score)
        upper = max(0.0, uncertainty.high.score - cell.score)
        if f"{lower:.3f}" == f"{upper:.3f}":
            uncertainty_value = rf"$\pm {upper:.3f}$"
        else:
            uncertainty_value = rf"$-{lower:.3f}/+{upper:.3f}$"
    else:
        uncertainty_value = (
            rf"$[{uncertainty.low.score:.3f},{uncertainty.high.score:.3f}]$"
        )
    if include_percentage:
        percent_value = f"({cell.percentage:.1f}\\%)"
        first_line = rf"{bit_value} {{\tiny {percent_value}}}"
        if uncertainty_value is None:
            content = first_line
        else:
            content = rf"\shortstack[r]{{{first_line}\\{uncertainty_value}}}"
    else:
        if uncertainty_value is None:
            content = bit_value
        else:
            content = rf"\shortstack[r]{{{bit_value}\\{uncertainty_value}}}"
    if aligned:
        return rf"\textbf{{{content}}}$^{{\star}}$"
    return content


def table_caption(
    include_percentages: bool, include_specialisation_gap: bool = True,
    include_design_alignment: bool = True,
) -> str:
    individual_table_label = (
        "tab:main-individual-target"
        if include_percentages
        else "tab:main-individual-target-bits-only"
    )
    if include_percentages:
        value_description = (
            r"Each cell reports OVMI $=C(S)I(X;Y\mid X\in S)$ in bits, with "
            r"point-estimate OVMI/$H(p)$ in parentheses and uncertainty beneath; "
            r"raw bits should not be compared "
            r"across reference columns, whereas percentages use the displayed "
            r"column entropy as divisor. "
        )
    else:
        value_description = (
            r"Each cell reports OVMI $=C(S)I(X;Y\mid X\in S)$ in bits. "
            r"Normalised percentages are omitted for compactness and equal each "
            r"cell divided by the displayed column $H(p)$; raw bits should not be "
            r"compared across reference columns. "
        )
    specialisation_gap_description = (
        r"The specialisation gap is OVMI on the system's most specific available "
        r"design reference minus its SUBTLEX--UK OVMI; a large value indicates "
        r"tuning to one setting and an em dash means that the design distribution "
        r"is unavailable. "
        if include_specialisation_gap else ""
    )
    design_alignment_description = (
        r"Bold $\star$ cells are design-aligned (narrative-trained/evaluated "
        r"systems on Sherlock, or the Moses/Willett caregiving vocabulary on UCV "
        r"and its own target distribution) and should not be read as general "
        r"capability. "
        if include_design_alignment else ""
    )
    return (
        r"Cross-study comparison on one axis. Each cell reports OVMI "
        + value_description.removeprefix("Each cell reports OVMI ")
        + r"Rows are ordered within each "
        r"block by SUBTLEX--UK OVMI. The attempted/invasive and perceived/non-invasive "
        r"labels are jointly stated because these factors are perfectly confounded in "
        r"the included literature; the blocks are not a modality comparison. "
        r"Superscripts on $P$ identify balanced top-1 accuracy ($\mathrm{b}$), "
        r"isolated-word accuracy ($\mathrm{a}$), and $1-\mathrm{WER}$ "
        r"($\mathrm{w}$); $\dagger$ marks a WER-derived lower bound because insertions "
        r"can increase WER without a failed intended word. A full-confusion-matrix "
        r"estimate would be marked $\mathrm{c}$, but none is available for these rows. "
        r"Bracketed ranges are 95\% sampling intervals for invasive systems: Wilson "
        r"score intervals for isolated-word accuracies and published sentence/trial "
        r"intervals for WER-derived rows. $\pm$ values are OVMI endpoints obtained "
        r"by mapping mean $P$ plus or minus one SEM across three "
        r"training seeds for the MEG evaluations or three participants for Tang; "
        r"these are not confidence intervals "
        r"and exclude test-set sampling. Per-word results within each seed were not "
        r"available for a nested bootstrap. Reference "
        r"distributions, hence $C(S)$ and $H(p_S)$, are treated as fixed. "
        r"The symmetric-channel approximation, WER-to-$P$ bridge, and isolated-versus-"
        r"continuous task mismatch are directional model qualifications and are not "
        r"included in these uncertainty ranges. "
        + design_alignment_description
        + specialisation_gap_description
        + r"The individual "
        + rf"target, reported separately in Table~\ref{{{individual_table_label}}} to keep "
        r"this table within the full text width at scriptsize, is Moses's 50-word "
        r"caregiving set with SUBTLEX weights. LibriBrain100 "
        r"2025 uses the method of d'Ascoli et al. trained on LibriBrain100 subject 0. "
        r"The conversational reference is the 36-call NLTK Switchboard sample; "
        r"Sherlock is the empirical 3,550-token LibriBrain test distribution."
    )


def render_table(
    points: list[SystemPoint], scores: dict[tuple[int, str], CellScore],
    uncertainties: dict[tuple[int, str], CellUncertainty],
    entropies: dict[str, float], output_path: Path, *,
    include_percentages: bool = True,
    include_specialisation_gap: bool = True,
    include_design_alignment: bool = True,
) -> None:
    main_references = tuple(
        spec for spec in REFERENCES if spec.key in MAIN_REFERENCE_KEYS
    )
    indexed_points = list(enumerate(points))
    blocks = (
        ("attempted", "Attempted speech (invasive)"),
        ("perceived", "Perceived speech (non-invasive)"),
    )
    column_count = 7 + int(include_specialisation_gap)
    column_spec = (
        r"@{\extracolsep{\fill}}lrr*{4}{r}r@{}"
        if include_specialisation_gap
        else r"@{\extracolsep{\fill}}lrr*{4}{r}@{}"
    )
    table_label = (
        "tab:main" if include_specialisation_gap
        else "tab:main-no-specialisation-gap"
    )
    if not include_percentages:
        table_label += "-bits-only"
    lines = [
        r"\begin{table*}[t]",
        rf"\caption{{{table_caption(include_percentages, include_specialisation_gap, include_design_alignment)}}}",
        rf"\label{{{table_label}}}",
        r"\centering",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{1.1pt}",
        r"\renewcommand{\arraystretch}{1.12}",
        rf"\begin{{tabular*}}{{\textwidth}}{{{column_spec}}}",
        r"\toprule",
        r"System & $V$ & $P$ & "
        + " & ".join(spec.header for spec in main_references)
        + (r" & \shortstack{Specialisation\\gap}" if include_specialisation_gap else "")
        + r" \\",
        r" & & & "
        + " & ".join(
            rf"\multicolumn{{1}}{{c}}{{\scriptsize $H(p)={entropies[spec.key]:.2f}$}}"
            for spec in main_references
        )
        + (r" &" if include_specialisation_gap else "")
        + r" \\",
        r"\midrule",
    ]
    for block_index, (group_key, block_label) in enumerate(blocks):
        if block_index:
            lines.append(r"\midrule")
        lines.append(
            rf"\multicolumn{{{column_count}}}{{l}}{{\textbf{{{block_label}}}}} \\"
        )
        members = [entry for entry in indexed_points if entry[1].group == group_key]
        members.sort(key=lambda entry: scores[(entry[0], "subtlex")].score, reverse=True)
        for member_index, (point_index, point) in enumerate(members):
            if member_index:
                lines.append(r"\addlinespace[1.2pt]")
            cells = [
                format_cell(
                    scores[(point_index, spec.key)],
                    uncertainties.get((point_index, spec.key)),
                    include_design_alignment and home_turf(point, spec.key),
                    include_percentages,
                )
                for spec in main_references
            ]
            design_key = design_reference(point)
            if design_key is None:
                gap = r"---"
            else:
                gap_value = (
                    scores[(point_index, design_key)].score
                    - scores[(point_index, "subtlex")].score
                )
                gap = f"{gap_value:+.3f}"
            row_values = [
                latex_escape(point.display_name),
                format_vocabulary_size(point.vocabulary_size),
                format_probability(point),
                *cells,
            ]
            if include_specialisation_gap:
                row_values.append(gap)
            lines.append(" & ".join(row_values) + r" \\")
    lines.extend([
        r"\bottomrule",
        r"\end{tabular*}",
        r"\end{table*}",
        "",
    ])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {len(points)} system points to {output_path}")


def render_individual_target_appendix(
    points: list[SystemPoint], scores: dict[tuple[int, str], CellScore],
    uncertainties: dict[tuple[int, str], CellUncertainty],
    entropy_value: float, output_path: Path, *,
    include_percentages: bool = True,
) -> None:
    indexed_points = list(enumerate(points))
    blocks = (
        ("attempted", "Attempted speech (invasive)"),
        ("perceived", "Perceived speech (non-invasive)"),
    )
    lines = [
        r"\begin{table}[t]",
        r"\caption{OVMI under the individual-target reference omitted from "
        r"Table~\ref{tab:main} for width. The reference is Moses's 50-word "
        r"caregiving set weighted by SUBTLEX--UK; formatting and $P$ markers "
        r"follow Table~\ref{tab:main}.}",
        (
            r"\label{tab:main-individual-target}"
            if include_percentages
            else r"\label{tab:main-individual-target-bits-only}"
        ),
        r"\centering",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{4pt}",
        r"\renewcommand{\arraystretch}{1.12}",
        r"\begin{tabular}{@{}lrrr@{}}",
        r"\toprule",
        r"System & $V$ & $P$ & \shortstack{Individual target\\Moses set} \\",
        rf" & & & \multicolumn{{1}}{{c}}{{\scriptsize $H(p)={entropy_value:.2f}$}} \\",
        r"\midrule",
    ]
    for block_index, (group_key, block_label) in enumerate(blocks):
        if block_index:
            lines.append(r"\midrule")
        lines.append(rf"\multicolumn{{4}}{{l}}{{\textbf{{{block_label}}}}} \\")
        members = [entry for entry in indexed_points if entry[1].group == group_key]
        members.sort(key=lambda entry: scores[(entry[0], "subtlex")].score, reverse=True)
        for member_index, (point_index, point) in enumerate(members):
            if member_index:
                lines.append(r"\addlinespace[1.2pt]")
            lines.append(" & ".join([
                latex_escape(point.display_name),
                format_vocabulary_size(point.vocabulary_size),
                format_probability(point),
                format_cell(
                    scores[(point_index, "individual")],
                    uncertainties.get((point_index, "individual")),
                    home_turf(point, "individual"),
                    include_percentages,
                ),
            ]) + r" \\")
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
        "",
    ])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote individual-target appendix table to {output_path}")


def main() -> None:
    args = parse_args()
    systems = pd.read_csv(args.systems)
    systems = systems.loc[systems["plot_eligible"].astype(bool)].copy()
    references, entropies = load_references(args.references_dir)
    vocabularies = reconstruct_vocabularies(
        systems, references, args.predictions_dir, args.cmudict, args.armeni_text,
        args.meg_masc_vocabulary, args.tang_vocabulary,
    )
    points = expand_system_points(systems)
    scores = score_all(points, vocabularies, references, entropies)
    uncertainties = score_uncertainties(
        points, vocabularies, references, entropies
    )
    validate_scores(points, scores)
    spearman_report(points, scores)
    key_uncertainty_report(points, scores, uncertainties)
    render_table(
        points, scores, uncertainties, entropies, args.output,
        include_percentages=True,
    )
    render_table(
        points, scores, uncertainties, entropies,
        args.no_specialisation_gap_output,
        include_percentages=True, include_specialisation_gap=False,
        include_design_alignment=False,
    )
    render_table(
        points, scores, uncertainties, entropies, args.bits_only_output,
        include_percentages=False,
    )
    render_individual_target_appendix(
        points, scores, uncertainties, entropies["individual"],
        args.appendix_output, include_percentages=True,
    )
    render_individual_target_appendix(
        points, scores, uncertainties, entropies["individual"],
        args.appendix_bits_only_output, include_percentages=False,
    )


if __name__ == "__main__":
    main()
