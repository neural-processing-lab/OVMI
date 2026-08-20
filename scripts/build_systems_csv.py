#!/usr/bin/env python3
"""Build data/systems.csv from primary-paper statistics and local evaluations.

This is the provenance-producing companion to ``make_contour_figure.py``.  The
plotter itself only reads the generated CSV.  The large-vocabulary coverage
calculation uses the CMU Pronouncing Dictionary identified by Willett et al.
and Card et al.; pass a local copy with ``--cmudict`` or allow this script to
cache the upstream dictionary.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from urllib.request import urlretrieve

import numpy as np
import pandas as pd

from ovmi import load_subtlex_uk, ovmi
from ovmi.core import entropy


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SUBTLEX = PROJECT_ROOT / "data/references/subtlex_uk.csv"
DEFAULT_PREDICTIONS = PROJECT_ROOT / "experiments/data/ovmi-predictions"
DEFAULT_CMUDICT = PROJECT_ROOT / "experiments/data/cache/cmudict.dict"
DEFAULT_ARMENI_TEXT = PROJECT_ROOT / "experiments/data/cache/armeni_advs.txt"
DEFAULT_MEG_MASC_TEXT_DIR = PROJECT_ROOT / "experiments/data/cache/meg_masc_stories"
DEFAULT_MEG_MASC_TEXTS = tuple(
    DEFAULT_MEG_MASC_TEXT_DIR / filename
    for filename in (
        "cable_spool_fort.txt",
        "easy_money.txt",
        "lw1.txt",
        "the_black_willow.txt",
    )
)
DEFAULT_OUTPUT = PROJECT_ROOT / "data/systems.csv"
DEFAULT_FRONTIER_OUTPUT = PROJECT_ROOT / "data/noiseless_frequency_frontier.csv"
DEFAULT_MEG_MASC_VOCABULARY_OUTPUT = (
    PROJECT_ROOT / "data/vocabularies/meg_masc_2023_v50.csv"
)
DEFAULT_TANG_VOCABULARY = PROJECT_ROOT / "data/vocabularies/tang_decoder_vocab.json"
CMUDICT_URL = "https://raw.githubusercontent.com/cmusphinx/cmudict/master/cmudict.dict"
ARMENI_TEXT_URL = "https://sherlock-holm.es/stories/plain-text/advs.txt"

MOSES_WORDS = """
am are bad bring clean closer comfortable coming computer do faith family feel
glasses going good goodbye have hello help here hope how hungry i is it like music
my need no not nurse okay outside please right success tell that they thirsty tired
up very what where yes you
""".split()

DASCOLI_LIBRIBRAIN_WORDS = """
is the a to it i not was we be he that have this they of there and are in but
will so all my for she were any really at out our am its had him an very has do
can time think good always new people as on
""".split()
DASCOLI_BALANCED_ACCURACIES = (0.256, 0.256, 0.262)
ARMENI_BALANCED_ACCURACIES = (0.211, 0.210, 0.202)
MEG_MASC_TOP1_ACCURACIES = (0.093, 0.083, 0.080)
TANG_WERS = (0.9407, 0.9354, 0.9243)
MEG_MASC_STORY_SHA256 = {
    "cable_spool_fort.txt": "2ec7caadad9319ef11deb5717b8c770dcf1d5f828a468a46fc35ed88af8d7cd0",
    "easy_money.txt": "6e36ab3c8df67546cf10b9d0a0d354899dc64f1e2312c61b1eea8cb24a5aa5ad",
    "lw1.txt": "0b08e81e1307af517e10008d58a6117c133dfed003b7bd674953442b68ec1d9c",
    "the_black_willow.txt": "c215948ccd26b060f25dbad70bc90931905eb29f90657fa99663269ae06afa3f",
}

MEGXL_SOURCE = (
    "Jayalath & Parker Jones (2026), MEG-XL, arXiv:2602.02494v2, pp. 5-6 "
    "(metric and Table 1) and p. 14 (Table 4); top-1 macro accuracy derived from "
    "five local pnpl/ovmi-predictions test artifacts. https://arxiv.org/abs/2602.02494"
)
MEG_MASC_SOURCE = (
    "Local V=50 evaluation on MEG-MASC (Gwilliams et al., 2023): supplied "
    "top-1 accuracies 9.3%, 8.3%, and 8.0% across three training seeds. "
    "Vocabulary derived from the 50 most frequent "
    "pre-tokenized lexical tokens across the four supplied MASC stories. "
    "https://doi.org/10.1038/s41597-023-02752-5; "
    "dataset https://doi.org/10.17605/OSF.IO/AG3KJ"
)
ARMENI_SOURCE = (
    "Armeni et al. (2022), Scientific Data 9:278, 10-hour within-participant "
    "MEG narrative dataset; local V=50 evaluation top-1 balanced accuracies "
    "21.1%, 21.0%, and 20.2% across three seeds. Vocabulary derived from the "
    f"50 most frequent normalized word tokens in {ARMENI_TEXT_URL}. "
    "https://doi.org/10.1038/s41597-022-01382-7"
)
DASCOLI_SOURCE = (
    "Local evaluation of the method of d'Ascoli et al. (2025), Nature "
    "Communications 16:10521, trained on LibriBrain100 subject 0 and evaluated "
    "on the Sherlock test set with the documented 50-word vocabulary: top-1 "
    "balanced accuracies 25.6%, 25.6%, and 26.2% across three seeds. "
    "https://doi.org/10.1038/s41467-025-65499-0"
)
BRAIN2QWERTY_SOURCE = (
    "Zhang, Levy et al. (2026), Accurate Decoding of Natural Sentences from "
    "Non-Invasive Brain Recordings (Brain2Qwerty v2), abstract p. 1 and Discussion "
    "p. 10. https://facebookresearch.github.io/brain2qwerty/assets/brain2qwerty_v2.pdf"
)
TANG_SOURCE = (
    "Tang et al. (2023), Nature Neuroscience 26:858-866, perceived-speech "
    "test-story WERs S1 94.07%, S2 93.54%, and S3 92.43%. The participant "
    "mean and SEM are computed locally. The decoder vocabulary contains the "
    "6,867 words occurring at least twice in the encoding-model training data. "
    "https://doi.org/10.1038/s41593-023-01304-9"
)
MOSES_SOURCE = (
    "Moses et al. (2021), NEJM 385:217-227, Results/Word Detection and Classification, "
    "pp. 222-223: 47.1% over 9,000 isolated-word attempts; Results, p. 221: median "
    "WER 25.6% (95% CI 17.1-37.1%) with language model across 15 blocks of 10 "
    "sentence trials (150 trials total). https://doi.org/10.1056/NEJMoa2027540"
)
WILLETT_SOURCE = (
    "Willett et al. (2023), Nature 620:1031-1036, Fig. 1d and Results p. 1032: "
    "94% 50-way isolated-word accuracy (20 trials/word); Table 1 p. 1034: 9.1% "
    "50-word WER (95% trial-bootstrap CI 7.2-11.2%; 250 vocal evaluation trials) "
    "and 23.8% 125,000-word WER (95% trial-bootstrap CI 21.8-25.9%; 400 vocal "
    "evaluation trials). https://doi.org/10.1038/s41586-023-06377-x"
)
CARD_SOURCE = (
    "Card et al. (2024), NEJM 391:609-618, Results/Online decoding performance: "
    "final five 125,000-word Copy Task sessions WER 2.5% (95% CI 2.0-3.1%). "
    "The paper estimates confidence intervals by 10,000-fold resampling of "
    "individual sentence trials; the aggregate is over five sessions. "
    "https://doi.org/10.1056/NEJMoa2314132"
)
SUBTLEX_SOURCE = (
    "SUBTLEX-UK: van Heuven et al. (2014), Quarterly Journal of Experimental "
    "Psychology 67:1176-1190, Spelling/FreqCount. https://doi.org/10.1080/17470218.2013.850521"
)
CMUDICT_SOURCE = (
    "CMU Pronouncing Dictionary word list (the 125k LM lexicon named by Willett/Card); "
    f"retrieved from {CMUDICT_URL}"
)


COLUMNS = [
    "system_id", "system_name", "label", "year", "modality", "invasiveness",
    "speech_condition", "task", "vocabulary_kind", "V", "trajectory",
    "operating_point", "plot_eligible", "exclusion_reason", "P_neural",
    "P_system", "P_system_is_lower_bound", "P_neural_ci_low", "P_neural_ci_high",
    "P_system_ci_low", "P_system_ci_high", "uncertainty_neural",
    "uncertainty_system", "seed_values_neural", "n_seeds_neural", "n_trials",
    "n_system_sentences", "n_system_blocks", "n_system_sessions", "ci_method", "coverage", "H_pS_bits", "I_invocab_neural_bits",
    "I_invocab_system_bits", "I_neural_ci_low_bits", "I_neural_ci_high_bits",
    "OVMI_neural_bits", "OVMI_system_bits", "reference", "reference_entropy_bits",
    "chance_level", "at_or_below_chance", "ovmi_check_abs_error_neural",
    "ovmi_check_abs_error_system", "source",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subtlex", type=Path, default=DEFAULT_SUBTLEX)
    parser.add_argument("--predictions-dir", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--cmudict", type=Path, default=DEFAULT_CMUDICT)
    parser.add_argument("--armeni-text", type=Path, default=DEFAULT_ARMENI_TEXT)
    parser.add_argument(
        "--tang-vocabulary", type=Path, default=DEFAULT_TANG_VOCABULARY,
    )
    parser.add_argument(
        "--meg-masc-texts", type=Path, nargs=4, default=DEFAULT_MEG_MASC_TEXTS,
        metavar=("CABLE", "EASY", "LW1", "WILLOW"),
        help="The four pre-tokenized MEG-MASC story text files.",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--frontier-output", type=Path, default=DEFAULT_FRONTIER_OUTPUT,
        help="Output CSV for the noiseless top-frequency vocabulary frontier.",
    )
    parser.add_argument(
        "--meg-masc-vocabulary-output", type=Path,
        default=DEFAULT_MEG_MASC_VOCABULARY_OUTPUT,
        help="Output the derived MEG-MASC V=50 vocabulary and token counts.",
    )
    parser.add_argument(
        "--megxl-vocabulary-sizes", type=int, nargs="+", default=(10, 20, 50, 100),
        help="Discrete MEG-XL vocabulary sizes to include (default: 10 20 50 100).",
    )
    return parser.parse_args()


def normalize_word(value: object) -> str:
    value = unicodedata.normalize("NFKC", str(value).strip())
    return value.casefold().replace("’", "'")


def normalized_reference(path: Path) -> dict[str, float]:
    if path.suffix.casefold() == ".csv":
        frame = pd.read_csv(path, keep_default_na=False)
        if list(frame.columns) != ["word", "weight"]:
            raise ValueError(f"{path} must contain exactly word,weight columns")
        raw = dict(zip(frame["word"].astype(str), frame["weight"].astype(float)))
    else:
        raw = load_subtlex_uk(path=path)
    result: dict[str, float] = {}
    for word, frequency in raw.items():
        normalized = normalize_word(word)
        result[normalized] = result.get(normalized, 0.0) + float(frequency)
    return result


def cmudict_words(path: Path) -> list[str]:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        urlretrieve(CMUDICT_URL, path)
    words = set()
    with path.open(encoding="latin-1") as handle:
        for line in handle:
            word = line.split(" ", 1)[0]
            word = re.sub(r"\(\d+\)$", "", word)
            words.add(normalize_word(word))
    return sorted(words)


def json_vocabulary(path: Path, expected_size: int) -> list[str]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError(f"Vocabulary must be a JSON list: {path}")
    words = [normalize_word(word) for word in raw]
    if len(words) != expected_size or len(set(words)) != expected_size:
        raise ValueError(f"{path} must contain {expected_size} unique words")
    return words


def top_words_from_plaintext(
    path: Path,
    *,
    vocabulary_size: int,
    source_url: str = ARMENI_TEXT_URL,
) -> tuple[list[str], Counter[str]]:
    """Return frequency-ranked normalized tokens with deterministic tie breaks."""

    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        urlretrieve(source_url, path)
    text = unicodedata.normalize("NFKC", path.read_text(encoding="utf-8"))
    text = text.casefold().replace("’", "'")
    counts = Counter(re.findall(r"[a-z]+(?:'[a-z]+)*", text))
    ranked = sorted(counts, key=lambda word: (-counts[word], word))
    if len(ranked) < vocabulary_size:
        raise ValueError(
            f"Plaintext contains only {len(ranked)} token types; need {vocabulary_size}"
        )
    return ranked[:vocabulary_size], counts


def top_words_from_pretokenized_texts(
    paths: list[Path] | tuple[Path, ...], *, vocabulary_size: int,
) -> tuple[list[str], Counter[str]]:
    """Count lexical whitespace tokens while preserving corpus clitic forms.

    The supplied MEG-MASC stories already separate punctuation and clitics with
    whitespace.  Standalone punctuation is discarded; forms such as ``n't``
    and ``'s`` remain distinct decoder labels.  Ties are resolved alphabetically.
    """

    lexical_token = re.compile(r"(?:[a-z]+(?:[-'][a-z]+)*|'[a-z]+)\Z")
    counts: Counter[str] = Counter()
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(
                f"Missing MEG-MASC story {path}; pass all four files with "
                "--meg-masc-texts"
            )
        expected_hash = MEG_MASC_STORY_SHA256.get(path.name)
        actual_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        if expected_hash is not None and actual_hash != expected_hash:
            raise ValueError(
                f"MEG-MASC story hash mismatch for {path.name}: {actual_hash}"
            )
        for raw_token in path.read_text(encoding="utf-8").split():
            token = normalize_word(raw_token)
            if lexical_token.fullmatch(token):
                counts[token] += 1
    ranked = sorted(counts, key=lambda word: (-counts[word], word))
    if len(ranked) < vocabulary_size:
        raise ValueError(
            f"MEG-MASC stories contain only {len(ranked)} lexical types; "
            f"need {vocabulary_size}"
        )
    return ranked[:vocabulary_size], counts


def write_ranked_vocabulary(
    path: Path, words: list[str], counts: Counter[str],
) -> None:
    frame = pd.DataFrame({
        "rank": np.arange(1, len(words) + 1),
        "word": words,
        "count": [counts[word] for word in words],
    })
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def wilson_interval(success_probability: float, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    denominator = 1.0 + z * z / n
    centre = (success_probability + z * z / (2.0 * n)) / denominator
    half_width = z * math.sqrt(
        success_probability * (1.0 - success_probability) / n + z * z / (4.0 * n * n)
    ) / denominator
    return centre - half_width, centre + half_width


def seed_sem_interval(
    values: np.ndarray | tuple[float, ...],
) -> tuple[float, tuple[float, float]]:
    """Return a seed mean and mean +/- one SEM, clipped to [0, 1]."""

    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or len(values) < 2 or np.any(~np.isfinite(values)):
        raise ValueError("seed values must be a finite one-dimensional array")
    mean = float(np.mean(values))
    standard_error = float(np.std(values, ddof=1) / math.sqrt(len(values)))
    interval = (max(0.0, mean - standard_error), min(1.0, mean + standard_error))
    return mean, interval


def details(reference: dict[str, float], vocabulary: list[str], probability: float | None):
    if probability is None or not math.isfinite(probability):
        return None
    return ovmi(reference, vocabulary, accuracy=probability, return_details=True)


def system_row(
    reference: dict[str, float],
    vocabulary: list[str],
    *,
    P_neural: float | None = None,
    P_system: float | None = None,
    P_neural_ci: tuple[float, float] | None = None,
    P_system_ci: tuple[float, float] | None = None,
    **metadata,
) -> dict[str, object]:
    neural = details(reference, vocabulary, P_neural)
    system = details(reference, vocabulary, P_system)
    low = details(reference, vocabulary, P_neural_ci[0]) if P_neural_ci else None
    high = details(reference, vocabulary, P_neural_ci[1]) if P_neural_ci else None
    noiseless = ovmi(reference, vocabulary, accuracy=1.0, return_details=True)
    coverage = noiseless.coverage
    default_p = P_system if P_system is not None else P_neural
    row = {
        **metadata,
        "P_neural": P_neural,
        "P_system": P_system,
        "P_neural_ci_low": P_neural_ci[0] if P_neural_ci else None,
        "P_neural_ci_high": P_neural_ci[1] if P_neural_ci else None,
        "P_system_ci_low": P_system_ci[0] if P_system_ci else None,
        "P_system_ci_high": P_system_ci[1] if P_system_ci else None,
        "coverage": coverage,
        "H_pS_bits": noiseless.in_vocab_information,
        "I_invocab_neural_bits": neural.in_vocab_information if neural else None,
        "I_invocab_system_bits": system.in_vocab_information if system else None,
        "I_neural_ci_low_bits": low.in_vocab_information if low else None,
        "I_neural_ci_high_bits": high.in_vocab_information if high else None,
        "OVMI_neural_bits": neural.score if neural else None,
        "OVMI_system_bits": system.score if system else None,
        "reference": "subtlex-uk",
        "reference_entropy_bits": REFERENCE_ENTROPY,
        "chance_level": 1.0 / len(vocabulary) if vocabulary else None,
        "at_or_below_chance": bool(default_p <= 1.0 / len(vocabulary)) if default_p is not None else None,
        "ovmi_check_abs_error_neural": (
            abs(neural.score - neural.coverage * neural.in_vocab_information) if neural else None
        ),
        "ovmi_check_abs_error_system": (
            abs(system.score - system.coverage * system.in_vocab_information) if system else None
        ),
    }
    return row


def frequency_frontier(
    reference: dict[str, float],
    *,
    max_sweep_v: int = 100_000,
    num_points: int = 256,
) -> pd.DataFrame:
    """Return the noiseless top-frequency vocabulary frontier in O(N)."""

    frequencies = np.asarray(sorted(reference.values(), reverse=True), dtype=np.float64)
    frequencies = frequencies[frequencies > 0]
    probabilities = frequencies / frequencies.sum()
    cumulative_mass = np.cumsum(probabilities)
    cumulative_p_log_p = np.cumsum(probabilities * np.log2(probabilities))

    sweep_end = min(max_sweep_v, len(probabilities))
    vocabulary_sizes = np.unique(
        np.rint(np.geomspace(2, sweep_end, num=num_points)).astype(int)
    )
    anchors = np.asarray([2, 50, 250, 1_000, 15_000, sweep_end, len(probabilities)])
    vocabulary_sizes = np.unique(
        np.concatenate([vocabulary_sizes, anchors[(anchors >= 2) & (anchors <= len(probabilities))]])
    )

    indices = vocabulary_sizes - 1
    coverage_values = cumulative_mass[indices]
    entropy_values = (
        np.log2(coverage_values)
        - cumulative_p_log_p[indices] / coverage_values
    )
    coverage_values[vocabulary_sizes == len(probabilities)] = 1.0
    entropy_values[vocabulary_sizes == len(probabilities)] = entropy(probabilities)
    return pd.DataFrame({
        "reference": "subtlex-uk",
        "V": vocabulary_sizes,
        "coverage": coverage_values,
        "H_pS_bits": entropy_values,
        "selection": "top-frequency",
    })


def megxl_rows(
    reference: dict[str, float],
    predictions_dir: Path,
    vocabulary_sizes: list[int] | tuple[int, ...],
) -> list[dict[str, object]]:
    vocabulary_sizes = sorted(set(vocabulary_sizes))
    if not vocabulary_sizes or vocabulary_sizes[0] < 2:
        raise ValueError("MEG-XL vocabulary sizes must all be at least 2")
    max_v = vocabulary_sizes[-1]
    run_dirs = sorted(path for path in predictions_dir.iterdir() if path.is_dir() and not path.name.startswith("."))
    if not run_dirs:
        raise FileNotFoundError(f"No MEG-XL prediction runs found in {predictions_dir}")

    probabilities = []
    trial_counts = []
    common_vocabulary = None
    for run_dir in run_dirs:
        with np.load(run_dir / "test_predictions_best.npz", allow_pickle=True) as archive:
            targets = np.asarray([normalize_word(word) for word in archive["target_word"]])
            predicted_embeddings = archive["pred_embedding"].astype(np.float64)
        with np.load(run_dir / "vocab_embeddings.npz", allow_pickle=True) as archive:
            words = np.asarray([normalize_word(word) for word in archive["word"]])
            vocabulary_embeddings = archive["t5_embedding"].astype(np.float64)

        word_to_index = {word: index for index, word in enumerate(words)}
        rows_by_word: dict[str, list[int]] = defaultdict(list)
        for row_index, word in enumerate(targets):
            rows_by_word[str(word)].append(row_index)
        selected_words = sorted(
            (word for word in rows_by_word if word in word_to_index and reference.get(word, 0.0) > 0),
            key=lambda word: (-reference[word], word),
        )[:max_v]
        if common_vocabulary is None:
            common_vocabulary = selected_words
        elif common_vocabulary != selected_words:
            raise ValueError("MEG-XL runs do not yield the same frequency-selected vocabulary")

        indices = np.asarray([word_to_index[word] for word in selected_words])
        predictions = predicted_embeddings / np.maximum(
            np.linalg.norm(predicted_embeddings, axis=1, keepdims=True), 1e-12
        )
        candidates = vocabulary_embeddings[indices]
        candidates /= np.maximum(np.linalg.norm(candidates, axis=1, keepdims=True), 1e-12)
        similarity = predictions @ candidates.T

        run_probabilities = []
        run_trials = []
        for vocabulary_size in range(2, max_v + 1):
            per_word = []
            n_trials = 0
            for local_index, word in enumerate(selected_words[:vocabulary_size]):
                rows = np.asarray(rows_by_word[word])
                n_trials += len(rows)
                per_word.append(float(np.mean(similarity[rows, :vocabulary_size].argmax(axis=1) == local_index)))
            run_probabilities.append(float(np.mean(per_word)))
            run_trials.append(n_trials)
        probabilities.append(run_probabilities)
        trial_counts.append(run_trials)

    probabilities_array = np.asarray(probabilities)
    if probabilities_array.shape[0] != 5:
        raise ValueError(
            "MEG-XL uncertainty is specified as a five-seed t interval; "
            f"found {probabilities_array.shape[0]} runs"
        )
    rows = []
    for vocabulary_size in vocabulary_sizes:
        probability_index = vocabulary_size - 2
        seed_probabilities = probabilities_array[:, probability_index]
        probability, probability_ci = seed_sem_interval(seed_probabilities)
        vocabulary = common_vocabulary[:vocabulary_size]
        rows.append(system_row(
            reference,
            vocabulary,
            system_id=f"megxl_v{vocabulary_size:03d}",
            system_name="MEG-XL",
            label=f"MEG-XL V={vocabulary_size}",
            year=2026,
            modality="MEG",
            invasiveness="non-invasive",
            speech_condition="perceived",
            task="audiobook listening; word-locked retrieval",
            vocabulary_kind="top-V SUBTLEX among locally evaluated LibriBrain words",
            V=vocabulary_size,
            trajectory=False,
            operating_point=vocabulary_size == 50,
            plot_eligible=True,
            exclusion_reason="",
            P_neural=probability,
            P_system=None,
            P_system_is_lower_bound=False,
            P_neural_ci=probability_ci,
            uncertainty_neural="seed_sem",
            uncertainty_system="none",
            seed_values_neural=";".join(f"{value:.12g}" for value in seed_probabilities),
            n_seeds_neural=5,
            ci_method="mean +/- SEM across five seed-level macro per-word accuracies",
            n_trials=int(trial_counts[0][probability_index]),
            source=f"{MEGXL_SOURCE}; {SUBTLEX_SOURCE}",
        ))
    return rows


def main() -> None:
    global REFERENCE_ENTROPY
    args = parse_args()
    if len(DASCOLI_LIBRIBRAIN_WORDS) != 50 or len(set(DASCOLI_LIBRIBRAIN_WORDS)) != 50:
        raise AssertionError("D'Ascoli LibriBrain vocabulary must contain 50 unique words")
    reference = normalized_reference(args.subtlex)
    REFERENCE_ENTROPY = entropy(np.asarray(list(reference.values()), dtype=np.float64))
    cmu_words = cmudict_words(args.cmudict)
    tang_words = json_vocabulary(args.tang_vocabulary, 6_867)
    armeni_words, armeni_counts = top_words_from_plaintext(
        args.armeni_text, vocabulary_size=50,
    )
    meg_masc_words, meg_masc_counts = top_words_from_pretokenized_texts(
        args.meg_masc_texts, vocabulary_size=50,
    )
    write_ranked_vocabulary(
        args.meg_masc_vocabulary_output, meg_masc_words, meg_masc_counts,
    )
    meg_masc_probability, meg_masc_ci = seed_sem_interval(
        MEG_MASC_TOP1_ACCURACIES
    )

    rows = [system_row(
        reference,
        meg_masc_words,
        system_id="meg_masc_2023_v50",
        system_name="MEG-MASC",
        label="MEG-MASC 2023 (V=50)",
        year=2023,
        modality="MEG",
        invasiveness="non-invasive",
        speech_condition="perceived",
        task="naturalistic story listening; local 50-way word classification",
        vocabulary_kind=(
            "top-50 frequency-ranked pre-tokenized lexical tokens across the "
            "four MEG-MASC stories"
        ),
        V=50,
        trajectory=False,
        operating_point=True,
        plot_eligible=True,
        exclusion_reason="",
        P_neural=meg_masc_probability,
        P_system=None,
        P_system_is_lower_bound=False,
        P_neural_ci=meg_masc_ci,
        P_system_ci=None,
        uncertainty_neural="seed_sem",
        uncertainty_system="none",
        seed_values_neural="0.093;0.083;0.080",
        n_seeds_neural=3,
        n_system_sentences=None,
        n_system_blocks=None,
        n_system_sessions=None,
        ci_method=(
            "mean +/- SEM across three seed-level top-1 accuracies; "
            "test-set sampling excluded"
        ),
        n_trials=None,
        source=f"{MEG_MASC_SOURCE}; {SUBTLEX_SOURCE}",
    )]

    dascoli_probability, dascoli_ci = seed_sem_interval(DASCOLI_BALANCED_ACCURACIES)
    armeni_probability, armeni_ci = seed_sem_interval(ARMENI_BALANCED_ACCURACIES)

    rows.append(system_row(
        reference,
        DASCOLI_LIBRIBRAIN_WORDS,
        system_id="dascoli_libribrain100_s0_v50",
        system_name="LibriBrain100",
        label="LibriBrain100 2025 (V=50)",
        year=2025,
        modality="MEG",
        invasiveness="non-invasive",
        speech_condition="perceived",
        task="LibriBrain100 subject 0; Sherlock test-set word classification",
        vocabulary_kind="documented 50-word Sherlock evaluation vocabulary",
        V=50,
        trajectory=False,
        operating_point=True,
        plot_eligible=True,
        exclusion_reason="",
        P_neural=dascoli_probability,
        P_system=None,
        P_system_is_lower_bound=False,
        P_neural_ci=dascoli_ci,
        P_system_ci=None,
        uncertainty_neural="seed_sem",
        uncertainty_system="none",
        seed_values_neural="0.256;0.256;0.262",
        n_seeds_neural=3,
        n_system_sentences=None,
        n_system_blocks=None,
        n_system_sessions=None,
        ci_method="mean +/- SEM across three seed-level top-1 balanced accuracies; test-set sampling excluded",
        n_trials=None,
        source=f"{DASCOLI_SOURCE}; {SUBTLEX_SOURCE}",
    ))

    rows.append(system_row(
        reference,
        armeni_words,
        system_id="armeni_2022_v50",
        system_name="Armeni et al.",
        label="Armeni 2022 (V=50)",
        year=2022,
        modality="MEG",
        invasiveness="non-invasive",
        speech_condition="perceived",
        task="audiobook listening; local 50-way balanced word classification",
        vocabulary_kind="top-50 frequency-ranked tokens in the Armeni Sherlock plaintext",
        V=50,
        trajectory=False,
        operating_point=True,
        plot_eligible=True,
        exclusion_reason="",
        P_neural=armeni_probability,
        P_system=None,
        P_system_is_lower_bound=False,
        P_neural_ci=armeni_ci,
        P_system_ci=None,
        uncertainty_neural="seed_sem",
        uncertainty_system="none",
        seed_values_neural="0.211;0.210;0.202",
        n_seeds_neural=3,
        n_system_sentences=None,
        n_system_blocks=None,
        n_system_sessions=None,
        ci_method="mean +/- SEM across three seed-level top-1 balanced accuracies; test-set sampling excluded",
        n_trials=None,
        source=f"{ARMENI_SOURCE}; {SUBTLEX_SOURCE}",
    ))

    tang_probabilities = tuple(1.0 - wer for wer in TANG_WERS)
    tang_probability, tang_ci = seed_sem_interval(tang_probabilities)
    rows.append(system_row(
        reference,
        tang_words,
        system_id="tang_2023_v6867",
        system_name="Tang et al.",
        label="Tang 2023 (V=6,867)",
        year=2023,
        modality="fMRI",
        invasiveness="non-invasive",
        speech_condition="perceived",
        task="continuous reconstruction of perceived stories",
        vocabulary_kind=(
            "published 6,867-word decoder vocabulary; words occurring at "
            "least twice in the encoding-model training data"
        ),
        V=6867,
        trajectory=False,
        operating_point=True,
        plot_eligible=True,
        exclusion_reason="",
        P_neural=None,
        P_system=tang_probability,
        P_system_is_lower_bound=True,
        P_neural_ci=None,
        P_system_ci=tang_ci,
        uncertainty_neural="none",
        uncertainty_system="participant_sem",
        seed_values_neural=None,
        n_seeds_neural=None,
        n_system_sentences=None,
        n_system_blocks=None,
        n_system_sessions=None,
        ci_method=(
            "mean +/- SEM across three participant-level WERs; propagated "
            "through the historical symmetric-channel estimate"
        ),
        n_trials=None,
        source=f"{TANG_SOURCE}; {SUBTLEX_SOURCE}",
    ))

    rows.append({
        **{column: None for column in COLUMNS},
        "system_id": "brain2qwerty_v2",
        "system_name": "Brain2Qwerty v2",
        "label": "Brain2Qwerty v2",
        "year": 2026,
        "modality": "MEG",
        "invasiveness": "non-invasive",
        "speech_condition": "typed",
        "task": "delayed typing by healthy participants",
        "vocabulary_kind": "open sentence generation; no fixed word-class vocabulary",
        "V": None,
        "trajectory": False,
        "operating_point": False,
        "plot_eligible": False,
        "exclusion_reason": "typing is not speech and no fixed vocabulary maps WER to the requested plane",
        "P_system": 0.61,
        "P_system_is_lower_bound": True,
        "reference": "subtlex-uk",
        "reference_entropy_bits": REFERENCE_ENTROPY,
        "source": f"{BRAIN2QWERTY_SOURCE}; 0.61 is 1 - reported mean WER 0.39",
    })

    moses_ci = wilson_interval(0.471, 9000)
    rows.append(system_row(
        reference,
        MOSES_WORDS,
        system_id="moses_2021_v50",
        system_name="Moses et al.",
        label="Moses 2021 (V=50)",
        year=2021,
        modality="ECoG",
        invasiveness="invasive",
        speech_condition="attempted",
        task="isolated-word classification / sentence decoding",
        vocabulary_kind="published Moses 50-word set",
        V=50,
        trajectory=False,
        operating_point=True,
        plot_eligible=True,
        exclusion_reason="",
        P_neural=0.471,
        P_system=0.744,
        P_system_is_lower_bound=True,
        P_neural_ci=moses_ci,
        P_system_ci=(1.0 - 0.371, 1.0 - 0.171),
        uncertainty_neural="wilson95",
        uncertainty_system="published95",
        seed_values_neural=None,
        n_seeds_neural=None,
        n_system_sentences=150,
        n_system_blocks=15,
        n_system_sessions=None,
        ci_method=(
            "neural: Wilson 95% binomial CI on 9,000 isolated-word attempts; "
            "system: published 95% CI for median WER across 15 ten-sentence blocks"
        ),
        n_trials=9000,
        source=f"{MOSES_SOURCE}; {SUBTLEX_SOURCE}",
    ))

    willett_ci = wilson_interval(0.94, 1000)
    rows.append(system_row(
        reference,
        MOSES_WORDS,
        system_id="willett_2023_v50",
        system_name="Willett et al.",
        label="Willett 2023 (V=50)",
        year=2023,
        modality="intracortical",
        invasiveness="invasive",
        speech_condition="attempted",
        task="isolated-word classification / sentence decoding",
        vocabulary_kind="Moses 50-word set reused by Willett",
        V=50,
        trajectory=False,
        operating_point=True,
        plot_eligible=True,
        exclusion_reason="",
        P_neural=0.94,
        P_system=0.909,
        P_system_is_lower_bound=True,
        P_neural_ci=willett_ci,
        P_system_ci=(1.0 - 0.112, 1.0 - 0.072),
        uncertainty_neural="wilson95",
        uncertainty_system="bootstrap95",
        seed_values_neural=None,
        n_seeds_neural=None,
        n_system_sentences=250,
        n_system_blocks=None,
        n_system_sessions=5,
        ci_method=(
            "neural: Wilson 95% binomial CI on 20 trials x 50 words; system: "
            "published 95% percentile CI from 10,000 bootstrap resamples over 250 sentence trials"
        ),
        n_trials=1000,
        source=f"{WILLETT_SOURCE}; {SUBTLEX_SOURCE}",
    ))

    rows.append(system_row(
        reference,
        cmu_words,
        system_id="willett_2023_v125k",
        system_name="Willett et al.",
        label="Willett 2023 (V=125k)",
        year=2023,
        modality="intracortical",
        invasiveness="invasive",
        speech_condition="attempted",
        task="continuous sentence decoding",
        vocabulary_kind="CMUdict proxy for reported 125,000-word LM lexicon",
        V=125000,
        trajectory=False,
        operating_point=True,
        plot_eligible=True,
        exclusion_reason="",
        P_neural=None,
        P_system=0.762,
        P_system_is_lower_bound=True,
        P_system_ci=(1.0 - 0.259, 1.0 - 0.218),
        uncertainty_neural="none",
        uncertainty_system="bootstrap95",
        seed_values_neural=None,
        n_seeds_neural=None,
        n_system_sentences=400,
        n_system_blocks=None,
        n_system_sessions=5,
        ci_method="published 95% percentile CI from 10,000 bootstrap resamples over 400 sentence trials",
        n_trials=None,
        source=f"{WILLETT_SOURCE}; {CMUDICT_SOURCE}; {SUBTLEX_SOURCE}",
    ))

    rows.append(system_row(
        reference,
        cmu_words,
        system_id="card_2024_v125k",
        system_name="Card et al.",
        label="Card 2024 (V=125k)",
        year=2024,
        modality="intracortical",
        invasiveness="invasive",
        speech_condition="attempted",
        task="continuous sentence decoding",
        vocabulary_kind="CMUdict proxy for reported 125,000-word LM lexicon",
        V=125000,
        trajectory=False,
        operating_point=True,
        plot_eligible=True,
        exclusion_reason="",
        P_neural=None,
        P_system=0.975,
        P_system_is_lower_bound=True,
        P_system_ci=(1.0 - 0.031, 1.0 - 0.020),
        uncertainty_neural="none",
        uncertainty_system="bootstrap95",
        seed_values_neural=None,
        n_seeds_neural=None,
        n_system_sentences=None,
        n_system_blocks=None,
        n_system_sessions=5,
        ci_method=(
            "published 95% CI from 10,000 bootstrap resamples over individual sentence "
            "trials pooled across the final five evaluation sessions; exact sentence count not reported"
        ),
        n_trials=None,
        source=f"{CARD_SOURCE}; {CMUDICT_SOURCE}; {SUBTLEX_SOURCE}",
    ))

    frame = pd.DataFrame(rows).reindex(columns=COLUMNS)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False, float_format="%.12g")

    frontier = frequency_frontier(reference)
    args.frontier_output.parent.mkdir(parents=True, exist_ok=True)
    frontier.to_csv(args.frontier_output, index=False, float_format="%.12g")

    max_error = frame[["ovmi_check_abs_error_neural", "ovmi_check_abs_error_system"]].max().max()
    print(f"Wrote {len(frame)} rows to {args.output}")
    print(f"Wrote {len(frontier)} frontier points to {args.frontier_output}")
    print(f"Wrote MEG-MASC vocabulary to {args.meg_masc_vocabulary_output}")
    print(f"SUBTLEX-UK entropy after documented lexical normalization: {REFERENCE_ENTROPY:.12f} bits")
    print(f"Maximum |OVMI - C*I_invocab|: {max_error:.3g}")
    print(f"CMUdict proxy: {len(cmu_words):,} entries; coverage={frame.loc[frame.system_id == 'card_2024_v125k', 'coverage'].iloc[0]:.12f}")
    print(
        "Armeni plaintext top-50 cutoff: "
        f"{armeni_words[-1]}={armeni_counts[armeni_words[-1]]} occurrences"
    )
    print(
        "MEG-MASC top-50 cutoff: "
        f"{meg_masc_words[-1]}={meg_masc_counts[meg_masc_words[-1]]} occurrences"
    )
    for vocabulary_size in (50, 250, 1_000, 15_000):
        anchor = frontier.loc[frontier["V"] == vocabulary_size].iloc[0]
        print(
            f"Frontier V={vocabulary_size:,}: "
            f"C={anchor['coverage']:.6f}, H(p_S)={anchor['H_pS_bits']:.6f} bits"
        )
    flagged = frame.loc[frame["at_or_below_chance"] == True, "system_id"].tolist()  # noqa: E712
    print(f"At/below chance: {flagged or 'none'}")


REFERENCE_ENTROPY = float("nan")


if __name__ == "__main__":
    main()
