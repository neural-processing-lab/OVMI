#!/usr/bin/env python3
"""Figure 6: sensitivity of scalar OVMI to unobserved channel structure."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
import numpy as np  # noqa: E402


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
SCRIPT_DIR = PROJECT_ROOT / "scripts"
for directory in (SRC_DIR, SCRIPT_DIR):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from make_main_table import load_reference  # noqa: E402
from ovmi import ovmi  # noqa: E402


DEFAULT_REFERENCE = PROJECT_ROOT / "data/references/subtlex_uk.csv"
DEFAULT_OUTPUT_BASE = PROJECT_ROOT / "figures/figure6_estimator_sensitivity"
DEFAULT_CAPTION = PROJECT_ROOT / "figures/figure6_estimator_sensitivity_caption.md"
DEFAULT_CACHE = (
    PROJECT_ROOT / "data/simulations/figure6_estimator_sensitivity_samples.npz"
)
DEFAULT_SUMMARY = (
    PROJECT_ROOT / "data/simulations/figure6_estimator_sensitivity_summary.csv"
)
DEFAULT_ROBUSTNESS = (
    PROJECT_ROOT / "data/simulations/figure6_estimator_sensitivity_robustness.csv"
)

P_VALUES = np.asarray((0.25, 0.50, 0.90), dtype=np.float64)
P_COLORS = ("#0072B2", "#E69F00", "#009E73")
MAIN_VOCABULARY_SIZE = 50
DEFAULT_SAMPLES = 600
DEFAULT_ROBUSTNESS_SAMPLES = 500
DEFAULT_SEED = 20260306
SIGMA_LEVELS = np.concatenate(([0.0], np.geomspace(0.05, 16.0, 24)))
ALPHA_LEVELS = np.concatenate(([np.inf], np.geomspace(1e3, 1e-2, 24)))
ROBUST_SIGMAS = np.asarray((0.0, 1.5, 12.0))
ROBUST_ALPHAS = np.asarray((np.inf, 1.0, 0.03))
CHECK_TOLERANCE = 1e-10


@dataclass(frozen=True)
class ReferenceSlice:
    words: np.ndarray
    coverage: float
    conditional_probabilities: np.ndarray


@dataclass(frozen=True)
class SimulationResult:
    p_values: np.ndarray
    scalar_ovmi: np.ndarray
    a_parameters: np.ndarray
    a_x: np.ndarray
    a_relative_error: np.ndarray
    a_absolute_error_bits: np.ndarray
    a_ratio: np.ndarray
    a_align_high_relative_error: np.ndarray
    a_align_low_relative_error: np.ndarray
    a_frequency_correlation: np.ndarray
    b_parameters: np.ndarray
    b_x: np.ndarray
    b_relative_error: np.ndarray
    b_absolute_error_bits: np.ndarray
    b_ratio: np.ndarray
    coverage: float
    p_s: np.ndarray
    words: np.ndarray
    vocabulary_size: int
    samples: int
    seed: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--samples", type=int, default=DEFAULT_SAMPLES)
    parser.add_argument(
        "--robustness-samples", type=int, default=DEFAULT_ROBUSTNESS_SAMPLES,
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--output-base", type=Path, default=DEFAULT_OUTPUT_BASE)
    parser.add_argument("--caption-output", type=Path, default=DEFAULT_CAPTION)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--robustness-output", type=Path, default=DEFAULT_ROBUSTNESS)
    parser.add_argument(
        "--reuse-cache", action="store_true",
        help="Redraw from the saved NPZ without rerunning the main simulation.",
    )
    parser.add_argument(
        "--skip-robustness", action="store_true",
        help="Skip the optional V=250 and uniform-prior diagnostics.",
    )
    return parser.parse_args()


def entropy_rows(probabilities: np.ndarray) -> np.ndarray:
    """Entropy along the last axis, using 0 log 0 = 0."""
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if np.any(~np.isfinite(probabilities)):
        raise AssertionError("Probabilities must be finite")
    if np.any(probabilities < -1e-12):
        raise AssertionError("Probabilities must be non-negative")
    safe = np.clip(probabilities, 0.0, None)
    terms = np.zeros_like(safe)
    positive = safe > 0.0
    terms[positive] = safe[positive] * np.log2(safe[positive])
    return -np.sum(terms, axis=-1)


def load_reference_slice(path: Path, vocabulary_size: int) -> ReferenceSlice:
    reference = load_reference(path)
    total = float(sum(reference.values()))
    ranked = sorted(reference.items(), key=lambda item: (-item[1], item[0]))
    selected = ranked[:vocabulary_size]
    if len(selected) != vocabulary_size:
        raise ValueError(f"Reference contains fewer than {vocabulary_size} words")
    words = np.asarray([word for word, _ in selected], dtype=str)
    weights = np.asarray([weight for _, weight in selected], dtype=np.float64)
    coverage = float(weights.sum() / total)
    p_s = weights / weights.sum()
    if not np.isclose(p_s.sum(), 1.0, atol=1e-14):
        raise AssertionError("Restricted reference does not sum to one")
    return ReferenceSlice(words, coverage, p_s)


def scalar_ovmi(coverage: float, p_s: np.ndarray, probability: float) -> float:
    vocabulary_size = len(p_s)
    error = (1.0 - probability) / (vocabulary_size - 1)
    output = error + p_s * (probability - error)
    row_entropy = entropy_rows(
        np.asarray([[probability, *([error] * (vocabulary_size - 1))]])
    )[0]
    return float(coverage * (entropy_rows(output[None, :])[0] - row_entropy))


def exact_uniform_offdiagonal(
    coverage: float, p_s: np.ndarray, accuracies: np.ndarray,
) -> np.ndarray:
    """Exact OVMI for row-specific accuracies and uniform within-row errors."""
    vocabulary_size = len(p_s)
    errors = (1.0 - accuracies) / (vocabulary_size - 1)
    if not np.allclose(
        accuracies + (vocabulary_size - 1) * errors, 1.0, atol=2e-13,
    ):
        raise AssertionError("Panel-A channel rows do not sum to one")
    base = np.sum(p_s[None, :] * errors, axis=1, keepdims=True)
    output = base + p_s[None, :] * (accuracies - errors)
    row_entropy = np.zeros_like(accuracies)
    positive_accuracy = accuracies > 0.0
    row_entropy[positive_accuracy] -= (
        accuracies[positive_accuracy] * np.log2(accuracies[positive_accuracy])
    )
    remaining = 1.0 - accuracies
    positive_error = remaining > 0.0
    row_entropy[positive_error] -= remaining[positive_error] * np.log2(
        remaining[positive_error] / (vocabulary_size - 1)
    )
    conditional_entropy = np.sum(row_entropy * p_s[None, :], axis=1)
    return coverage * (entropy_rows(output) - conditional_entropy)


def _sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def heterogeneous_accuracies(
    rng: np.random.Generator, samples: int, vocabulary_size: int,
    probability: float, sigma: float,
) -> np.ndarray:
    """Random accuracies in [0,1] with arithmetic mean exactly P per channel."""
    if sigma == 0.0:
        return np.full((samples, vocabulary_size), probability, dtype=np.float64)
    directions = rng.standard_normal((samples, vocabulary_size))
    directions -= directions.mean(axis=1, keepdims=True)
    directions /= directions.std(axis=1, keepdims=True)

    low = np.full(samples, -80.0 - sigma * 5.0)
    high = np.full(samples, 80.0 + sigma * 5.0)
    for _ in range(90):
        midpoint = (low + high) / 2.0
        realised = _sigmoid(midpoint[:, None] + sigma * directions).mean(axis=1)
        too_high = realised > probability
        high = np.where(too_high, midpoint, high)
        low = np.where(too_high, low, midpoint)
    intercept = (low + high) / 2.0
    accuracies = _sigmoid(intercept[:, None] + sigma * directions)
    accuracies += probability - accuracies.mean(axis=1, keepdims=True)
    if np.any(accuracies < -1e-12) or np.any(accuracies > 1.0 + 1e-12):
        raise AssertionError("Generated per-word accuracies left [0,1]")
    accuracies = np.clip(accuracies, 0.0, 1.0)
    if not np.allclose(accuracies.mean(axis=1), probability, atol=2e-13):
        raise AssertionError("Generated macro accuracy does not equal P")
    return accuracies


def relative_metrics(exact: np.ndarray, scalar: float) -> tuple[np.ndarray, ...]:
    relative = 100.0 * (exact - scalar) / scalar
    absolute_bits = np.abs(exact - scalar)
    ratio = exact / scalar
    return relative, absolute_bits, ratio


def simulate_panel_a(
    reference: ReferenceSlice, p_values: np.ndarray, sigma_levels: np.ndarray,
    samples: int, seed: int,
) -> tuple[np.ndarray, ...]:
    shape = (len(p_values), len(sigma_levels), samples)
    realised_x = np.empty(shape)
    relative = np.empty(shape)
    absolute_bits = np.empty(shape)
    ratio = np.empty(shape)
    align_high = np.empty(shape)
    align_low = np.empty(shape)
    frequency_correlation = np.empty(shape)
    scalar_scores = np.asarray([
        scalar_ovmi(reference.coverage, reference.conditional_probabilities, p)
        for p in p_values
    ])

    for p_index, probability in enumerate(p_values):
        scale = np.sqrt(probability * (1.0 - probability))
        for level_index, sigma in enumerate(sigma_levels):
            rng = np.random.default_rng(seed + 10_000 + p_index * 1_000 + level_index)
            accuracies = heterogeneous_accuracies(
                rng, samples, len(reference.words), probability, float(sigma),
            )
            realised_x[p_index, level_index] = accuracies.std(axis=1) / scale
            if sigma == 0.0:
                frequency_correlation[p_index, level_index] = 0.0
            else:
                centred_accuracy = accuracies - accuracies.mean(axis=1, keepdims=True)
                log_frequency = np.log(reference.conditional_probabilities)
                centred_frequency = log_frequency - log_frequency.mean()
                frequency_variation = np.sum(centred_frequency ** 2)
                if frequency_variation <= 1e-24:
                    frequency_correlation[p_index, level_index] = 0.0
                else:
                    numerator = np.sum(
                        centred_accuracy * centred_frequency[None, :], axis=1,
                    )
                    denominator = np.sqrt(
                        np.sum(centred_accuracy ** 2, axis=1)
                        * frequency_variation
                    )
                    frequency_correlation[p_index, level_index] = (
                        numerator / denominator
                    )
            exact = exact_uniform_offdiagonal(
                reference.coverage, reference.conditional_probabilities, accuracies,
            )
            metrics = relative_metrics(exact, scalar_scores[p_index])
            relative[p_index, level_index] = metrics[0]
            absolute_bits[p_index, level_index] = metrics[1]
            ratio[p_index, level_index] = metrics[2]

            sorted_accuracies = np.sort(accuracies, axis=1)
            exact_high = exact_uniform_offdiagonal(
                reference.coverage, reference.conditional_probabilities,
                sorted_accuracies[:, ::-1],
            )
            exact_low = exact_uniform_offdiagonal(
                reference.coverage, reference.conditional_probabilities,
                sorted_accuracies,
            )
            align_high[p_index, level_index] = relative_metrics(
                exact_high, scalar_scores[p_index],
            )[0]
            align_low[p_index, level_index] = relative_metrics(
                exact_low, scalar_scores[p_index],
            )[0]

    return (
        realised_x, relative, absolute_bits, ratio, align_high, align_low,
        frequency_correlation,
    )


def _dirichlet_error_rows(
    rng: np.random.Generator, batch: int, vocabulary_size: int, alpha: float,
) -> np.ndarray:
    offdiagonal = rng.dirichlet(
        np.full(vocabulary_size - 1, alpha, dtype=np.float64),
        size=batch * vocabulary_size,
    ).reshape(batch, vocabulary_size, vocabulary_size - 1)
    errors = np.zeros((batch, vocabulary_size, vocabulary_size), dtype=np.float64)
    mask = ~np.eye(vocabulary_size, dtype=bool)
    errors[:, mask] = offdiagonal.reshape(batch, -1)
    if np.any(errors < 0.0):
        raise AssertionError("Dirichlet errors must be non-negative")
    if not np.allclose(errors.sum(axis=2), 1.0, atol=2e-13):
        raise AssertionError("Dirichlet error rows do not sum to one")
    return errors


def exact_dirichlet_channels(
    reference: ReferenceSlice, probability: float, alpha: float,
    samples: int, seed: int, batch_size: int = 64,
) -> tuple[np.ndarray, np.ndarray]:
    vocabulary_size = len(reference.words)
    if np.isinf(alpha):
        concentration = np.zeros(samples)
        exact = np.full(
            samples,
            scalar_ovmi(
                reference.coverage, reference.conditional_probabilities, probability,
            ),
        )
        return concentration, exact

    rng = np.random.default_rng(seed)
    concentration_parts = []
    exact_parts = []
    binary_entropy = entropy_rows(
        np.asarray([[probability, 1.0 - probability]])
    )[0]
    normaliser = np.log2(vocabulary_size - 1)
    for start in range(0, samples, batch_size):
        batch = min(batch_size, samples - start)
        error_rows = _dirichlet_error_rows(rng, batch, vocabulary_size, alpha)
        row_error_entropy = entropy_rows(error_rows)
        concentration_parts.append(1.0 - row_error_entropy.mean(axis=1) / normaliser)

        output = (
            probability * reference.conditional_probabilities[None, :]
            + (1.0 - probability)
            * np.einsum(
                "x,nxy->ny", reference.conditional_probabilities, error_rows,
                optimize=True,
            )
        )
        conditional_entropy = binary_entropy + (1.0 - probability) * np.sum(
            row_error_entropy * reference.conditional_probabilities[None, :],
            axis=1,
        )
        exact_parts.append(
            reference.coverage * (entropy_rows(output) - conditional_entropy)
        )

        channel_row_sums = probability + (1.0 - probability) * error_rows.sum(axis=2)
        if not np.allclose(channel_row_sums, 1.0, atol=2e-13):
            raise AssertionError("Constructed channel rows do not sum to one")
        diagonal = np.diagonal(error_rows, axis1=1, axis2=2)
        realised_diagonal = probability + (1.0 - probability) * diagonal
        if not np.allclose(realised_diagonal.mean(axis=1), probability, atol=2e-13):
            raise AssertionError("Channel diagonal accuracy changed")

    return np.concatenate(concentration_parts), np.concatenate(exact_parts)


def simulate_panel_b(
    reference: ReferenceSlice, p_values: np.ndarray, alpha_levels: np.ndarray,
    samples: int, seed: int,
) -> tuple[np.ndarray, ...]:
    shape = (len(p_values), len(alpha_levels), samples)
    realised_x = np.empty(shape)
    relative = np.empty(shape)
    absolute_bits = np.empty(shape)
    ratio = np.empty(shape)
    scalar_scores = np.asarray([
        scalar_ovmi(reference.coverage, reference.conditional_probabilities, p)
        for p in p_values
    ])
    batch_size = 64 if len(reference.words) <= 50 else 24

    for p_index, probability in enumerate(p_values):
        for level_index, alpha in enumerate(alpha_levels):
            concentration, exact = exact_dirichlet_channels(
                reference, float(probability), float(alpha), samples,
                seed + 20_000 + p_index * 1_000 + level_index,
                batch_size=batch_size,
            )
            realised_x[p_index, level_index] = concentration
            metrics = relative_metrics(exact, scalar_scores[p_index])
            relative[p_index, level_index] = metrics[0]
            absolute_bits[p_index, level_index] = metrics[1]
            ratio[p_index, level_index] = metrics[2]
    return realised_x, relative, absolute_bits, ratio


def validate_homogeneous(reference: ReferenceSlice, p_values: np.ndarray) -> None:
    for probability in p_values:
        scalar = scalar_ovmi(
            reference.coverage, reference.conditional_probabilities, float(probability),
        )
        accuracies = np.full((1, len(reference.words)), probability)
        exact = exact_uniform_offdiagonal(
            reference.coverage, reference.conditional_probabilities, accuracies,
        )[0]
        if abs(exact - scalar) > CHECK_TOLERANCE:
            raise AssertionError(f"Homogeneous validation failed at P={probability}")

        reference_mapping = {
            word: float(weight)
            for word, weight in zip(
                reference.words, reference.conditional_probabilities,
            )
        }
        library_score = float(
            ovmi(reference_mapping, reference.words, accuracy=float(probability))
        )
        expected_without_coverage = scalar / reference.coverage
        if abs(library_score - expected_without_coverage) > CHECK_TOLERANCE:
            raise AssertionError("Scalar implementation disagrees with ovmi helper")

    chance = 1.0 / len(reference.words)
    chance_score = scalar_ovmi(
        reference.coverage, reference.conditional_probabilities, chance,
    )
    chance_exact = exact_uniform_offdiagonal(
        reference.coverage, reference.conditional_probabilities,
        np.full((1, len(reference.words)), chance),
    )[0]
    if abs(chance_score) > CHECK_TOLERANCE or abs(chance_exact) > CHECK_TOLERANCE:
        raise AssertionError("Chance channel should have zero OVMI")
    print(
        f"CHECK homogeneous scalar=exact for V={len(reference.words)}; "
        f"chance score={chance_score:.3g} bits"
    )


def run_simulation(
    reference: ReferenceSlice, samples: int, seed: int,
    sigma_levels: np.ndarray = SIGMA_LEVELS,
    alpha_levels: np.ndarray = ALPHA_LEVELS,
) -> SimulationResult:
    if samples < 1:
        raise ValueError("samples must be positive")
    validate_homogeneous(reference, P_VALUES)
    scalar_scores = np.asarray([
        scalar_ovmi(reference.coverage, reference.conditional_probabilities, p)
        for p in P_VALUES
    ])
    a = simulate_panel_a(reference, P_VALUES, sigma_levels, samples, seed)
    b = simulate_panel_b(reference, P_VALUES, alpha_levels, samples, seed)
    result = SimulationResult(
        p_values=P_VALUES.copy(), scalar_ovmi=scalar_scores,
        a_parameters=np.asarray(sigma_levels), a_x=a[0],
        a_relative_error=a[1], a_absolute_error_bits=a[2], a_ratio=a[3],
        a_align_high_relative_error=a[4], a_align_low_relative_error=a[5],
        a_frequency_correlation=a[6],
        b_parameters=np.asarray(alpha_levels), b_x=b[0],
        b_relative_error=b[1], b_absolute_error_bits=b[2], b_ratio=b[3],
        coverage=reference.coverage,
        p_s=reference.conditional_probabilities.copy(), words=reference.words.copy(),
        vocabulary_size=len(reference.words), samples=samples, seed=seed,
    )
    validate_simulation_result(result)
    return result


def validate_simulation_result(result: SimulationResult) -> None:
    expected_a_shape = (
        len(result.p_values), len(result.a_parameters), result.samples,
    )
    expected_b_shape = (
        len(result.p_values), len(result.b_parameters), result.samples,
    )
    if result.a_x.shape != expected_a_shape or result.b_x.shape != expected_b_shape:
        raise AssertionError("Simulation arrays have inconsistent shapes")
    finite_arrays = (
        result.p_values, result.scalar_ovmi, result.a_parameters, result.a_x,
        result.a_relative_error, result.a_absolute_error_bits, result.a_ratio,
        result.a_align_high_relative_error, result.a_align_low_relative_error,
        result.a_frequency_correlation,
        result.b_x, result.b_relative_error, result.b_absolute_error_bits,
        result.b_ratio, result.p_s,
    )
    if not all(np.isfinite(values).all() for values in finite_arrays):
        raise AssertionError("Simulation contains a non-finite value")
    if not (
        np.isinf(result.b_parameters[0])
        and np.isfinite(result.b_parameters[1:]).all()
    ):
        raise AssertionError("Panel-B parameter grid must start at uniform errors")
    if np.any(result.a_x < -1e-12) or np.any(result.a_x > 1.0 + 1e-12):
        raise AssertionError("Accuracy heterogeneity left [0,1]")
    if np.any(result.b_x < -1e-12) or np.any(result.b_x > 1.0 + 1e-12):
        raise AssertionError("Error concentration left [0,1]")
    if np.any(result.scalar_ovmi <= 0.0) or np.any(result.a_ratio <= 0.0):
        raise AssertionError("OVMI scores and ratios must be positive")
    if np.any(result.b_ratio <= 0.0):
        raise AssertionError("OVMI ratios must be positive")
    if not np.allclose(result.a_relative_error[:, 0], 0.0, atol=CHECK_TOLERANCE):
        raise AssertionError("Panel A does not start at exact scalar agreement")
    if not np.allclose(result.b_relative_error[:, 0], 0.0, atol=CHECK_TOLERANCE):
        raise AssertionError("Panel B does not start at exact scalar agreement")
    if result.samples >= 500:
        mean_correlations = np.mean(result.a_frequency_correlation[:, 1:], axis=2)
        if np.max(np.abs(mean_correlations)) > 0.025:
            raise AssertionError(
                "Random accuracy perturbations correlate systematically with frequency"
            )


def save_cache(result: SimulationResult, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        p_values=result.p_values, scalar_ovmi=result.scalar_ovmi,
        a_parameters=result.a_parameters, a_x=result.a_x,
        a_relative_error=result.a_relative_error,
        a_absolute_error_bits=result.a_absolute_error_bits,
        a_ratio=result.a_ratio,
        a_align_high_relative_error=result.a_align_high_relative_error,
        a_align_low_relative_error=result.a_align_low_relative_error,
        a_frequency_correlation=result.a_frequency_correlation,
        b_parameters=result.b_parameters, b_x=result.b_x,
        b_relative_error=result.b_relative_error,
        b_absolute_error_bits=result.b_absolute_error_bits,
        b_ratio=result.b_ratio,
        coverage=np.asarray(result.coverage), p_s=result.p_s, words=result.words,
        vocabulary_size=np.asarray(result.vocabulary_size),
        samples=np.asarray(result.samples), seed=np.asarray(result.seed),
    )
    print(f"Saved simulation cache to {path}")


def load_cache(path: Path) -> SimulationResult:
    with np.load(path, allow_pickle=False) as archive:
        result = SimulationResult(
            p_values=archive["p_values"], scalar_ovmi=archive["scalar_ovmi"],
            a_parameters=archive["a_parameters"], a_x=archive["a_x"],
            a_relative_error=archive["a_relative_error"],
            a_absolute_error_bits=archive["a_absolute_error_bits"],
            a_ratio=archive["a_ratio"],
            a_align_high_relative_error=archive["a_align_high_relative_error"],
            a_align_low_relative_error=archive["a_align_low_relative_error"],
            a_frequency_correlation=archive["a_frequency_correlation"],
            b_parameters=archive["b_parameters"], b_x=archive["b_x"],
            b_relative_error=archive["b_relative_error"],
            b_absolute_error_bits=archive["b_absolute_error_bits"],
            b_ratio=archive["b_ratio"], coverage=float(archive["coverage"]),
            p_s=archive["p_s"], words=archive["words"],
            vocabulary_size=int(archive["vocabulary_size"]),
            samples=int(archive["samples"]), seed=int(archive["seed"]),
        )
    validate_simulation_result(result)
    return result


def curve_summary(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, ...]:
    x_median = np.median(x, axis=1)
    y_median = np.median(y, axis=1)
    y_low, y_high = np.quantile(y, (0.05, 0.95), axis=1)
    order = np.argsort(x_median)
    return x_median[order], y_median[order], y_low[order], y_high[order]


def write_summary_csv(result: SimulationResult, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = (
        "panel", "P", "parameter", "x_median", "relative_median",
        "relative_p05", "relative_p95", "absolute_bits_median",
        "absolute_bits_p95", "ratio_median", "alignment_high_median",
        "alignment_low_median", "frequency_correlation_mean",
    )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for panel in ("A", "B"):
            parameters = result.a_parameters if panel == "A" else result.b_parameters
            x_values = result.a_x if panel == "A" else result.b_x
            relative = (
                result.a_relative_error if panel == "A" else result.b_relative_error
            )
            absolute = (
                result.a_absolute_error_bits
                if panel == "A" else result.b_absolute_error_bits
            )
            ratios = result.a_ratio if panel == "A" else result.b_ratio
            for p_index, probability in enumerate(result.p_values):
                for level_index, parameter in enumerate(parameters):
                    row = {
                        "panel": panel, "P": probability,
                        "parameter": parameter,
                        "x_median": np.median(x_values[p_index, level_index]),
                        "relative_median": np.median(relative[p_index, level_index]),
                        "relative_p05": np.quantile(relative[p_index, level_index], 0.05),
                        "relative_p95": np.quantile(relative[p_index, level_index], 0.95),
                        "absolute_bits_median": np.median(absolute[p_index, level_index]),
                        "absolute_bits_p95": np.quantile(absolute[p_index, level_index], 0.95),
                        "ratio_median": np.median(ratios[p_index, level_index]),
                        "alignment_high_median": "",
                        "alignment_low_median": "",
                        "frequency_correlation_mean": "",
                    }
                    if panel == "A":
                        row["alignment_high_median"] = np.median(
                            result.a_align_high_relative_error[p_index, level_index]
                        )
                        row["alignment_low_median"] = np.median(
                            result.a_align_low_relative_error[p_index, level_index]
                        )
                        row["frequency_correlation_mean"] = np.mean(
                            result.a_frequency_correlation[p_index, level_index]
                        )
                    writer.writerow(row)
    print(f"Wrote simulation summary to {path}")


def configure_style() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "mathtext.fontset": "dejavusans",
        "font.size": 9.0,
        "axes.labelsize": 10.5,
        "axes.titlesize": 10.5,
        "axes.titleweight": "semibold",
        "axes.linewidth": 1.0,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "legend.fontsize": 8.5,
        "lines.linewidth": 2.4,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def draw_figure(result: SimulationResult):
    configure_style()
    figure, axes = plt.subplots(
        1, 2, figsize=(8.0, 3.0), sharey=True,
        gridspec_kw={"wspace": 0.18},
    )
    figure.subplots_adjust(left=0.086, right=0.987, bottom=0.20, top=0.77)
    panels = (
        (
            axes[0], result.a_x, result.a_relative_error,
            "Per-word accuracy heterogeneity", "Per-word accuracy heterogeneity",
        ),
        (
            axes[1], result.b_x, result.b_relative_error,
            "Concentration of decoding errors", "Error concentration",
        ),
    )
    global_low = np.inf
    global_high = -np.inf
    for _axis, _x, y, _title, _xlabel in panels:
        global_low = min(global_low, float(np.quantile(y, 0.005)))
        global_high = max(global_high, float(np.quantile(y, 0.995)))

    for axis, x_values, y_values, title, xlabel in panels:
        axis.axhline(
            0.0, color="#6B7280", linestyle=(0, (3.0, 2.2)),
            linewidth=1.05, zorder=1,
        )
        for p_index, (probability, color) in enumerate(
            zip(result.p_values, P_COLORS)
        ):
            x, median, low, high = curve_summary(
                x_values[p_index], y_values[p_index],
            )
            axis.fill_between(x, low, high, color=color, alpha=0.16, linewidth=0)
            axis.plot(x, median, color=color, linewidth=2.2, zorder=3)
        axis.set_xlim(left=0.0)
        axis.set_xlabel(xlabel)
        axis.set_title(title, pad=7)
        axis.grid(axis="y", color="#D5D9DE", linewidth=0.65, alpha=0.72)
        axis.set_axisbelow(True)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.tick_params(length=3.2, width=0.9, color="#4B5563")

    span = global_high - global_low
    padding = max(2.0, 0.08 * span)
    axes[0].set_ylim(global_low - padding, global_high + padding)
    axes[0].set_ylabel("Scalar-OVMI estimation error (%)")
    axes[0].text(
        -0.17, 1.08, "a", transform=axes[0].transAxes, fontsize=12.0,
        fontweight="bold", ha="left", va="bottom",
    )
    axes[1].text(
        -0.12, 1.08, "b", transform=axes[1].transAxes, fontsize=12.0,
        fontweight="bold", ha="left", va="bottom",
    )
    handles = [
        Line2D([], [], color=color, linewidth=2.4, label=f"P = {p:.0%}")
        for p, color in zip(result.p_values, P_COLORS)
    ]
    figure.legend(
        handles=handles, loc="upper center", bbox_to_anchor=(0.53, 0.995),
        ncol=3, frameon=False, handlelength=2.6, columnspacing=1.7,
    )
    return figure


def save_figure(figure, output_base: Path) -> None:
    output_base.parent.mkdir(parents=True, exist_ok=True)
    for extension in ("pdf", "png"):
        output = output_base.with_suffix(f".{extension}")
        figure.savefig(output, dpi=400 if extension == "png" else None)
        print(f"Saved {output}")


def _moderate_and_strong_summary(
    x_values: np.ndarray, relative: np.ndarray,
) -> tuple[int, int, float, float, float, float]:
    level_x = np.median(x_values, axis=1)
    nonzero = np.flatnonzero(level_x > 1e-8)
    moderate_index = nonzero[np.argmin(np.abs(level_x[nonzero] - 0.35))]
    strong_index = int(np.argmax(level_x))
    moderate_error = float(np.median(np.abs(relative[moderate_index])))
    strong_error = float(np.quantile(np.abs(relative[strong_index]), 0.95))
    return (
        int(moderate_index), strong_index, float(level_x[moderate_index]),
        float(level_x[strong_index]), moderate_error, strong_error,
    )


def report_main_findings(result: SimulationResult) -> None:
    print("Main simulation summary (signed curves; absolute errors below):")
    for p_index, probability in enumerate(result.p_values):
        a = _moderate_and_strong_summary(
            result.a_x[p_index], result.a_relative_error[p_index],
        )
        b = _moderate_and_strong_summary(
            result.b_x[p_index], result.b_relative_error[p_index],
        )
        print(
            f"  P={probability:.0%}: scalar={result.scalar_ovmi[p_index]:.6f} bits; "
            f"A median |error|={a[4]:.2f}% at heterogeneity={a[2]:.2f}, "
            f"A strong p95 |error|={a[5]:.2f}% at {a[3]:.2f}; "
            f"B median |error|={b[4]:.2f}% at concentration={b[2]:.2f}, "
            f"B strong p95 |error|={b[5]:.2f}% at {b[3]:.2f}"
        )
    maximum_mean_correlation = np.max(np.abs(np.mean(
        result.a_frequency_correlation[:, 1:], axis=2,
    )))
    print(
        "CHECK random accuracy-frequency assignment: "
        f"max |mean Pearson r|={maximum_mean_correlation:.4f}"
    )


def run_optional_robustness(
    reference_path: Path, samples: int, seed: int, output: Path,
) -> None:
    configurations = []
    top_250 = load_reference_slice(reference_path, 250)
    configurations.append(("SUBTLEX top-250", top_250))
    top_50 = load_reference_slice(reference_path, 50)
    uniform_50 = ReferenceSlice(
        top_50.words.copy(), top_50.coverage,
        np.full(50, 1.0 / 50.0),
    )
    configurations.append(("uniform p_S, V=50", uniform_50))

    output.parent.mkdir(parents=True, exist_ok=True)
    fields = (
        "configuration", "panel", "P", "intensity", "parameter",
        "x_median", "relative_abs_median", "relative_abs_p95",
        "absolute_bits_median", "ratio_median",
    )
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for config_index, (name, reference) in enumerate(configurations):
            validate_homogeneous(reference, P_VALUES)
            a = simulate_panel_a(
                reference, P_VALUES, ROBUST_SIGMAS, samples,
                seed + 100_000 + config_index * 10_000,
            )
            b = simulate_panel_b(
                reference, P_VALUES, ROBUST_ALPHAS, samples,
                seed + 200_000 + config_index * 10_000,
            )
            for panel, parameters, x_values, relative, absolute, ratios in (
                ("A", ROBUST_SIGMAS, a[0], a[1], a[2], a[3]),
                ("B", ROBUST_ALPHAS, b[0], b[1], b[2], b[3]),
            ):
                for p_index, probability in enumerate(P_VALUES):
                    for level_index, intensity in ((1, "moderate"), (2, "strong")):
                        writer.writerow({
                            "configuration": name, "panel": panel,
                            "P": probability, "intensity": intensity,
                            "parameter": parameters[level_index],
                            "x_median": np.median(x_values[p_index, level_index]),
                            "relative_abs_median": np.median(
                                np.abs(relative[p_index, level_index])
                            ),
                            "relative_abs_p95": np.quantile(
                                np.abs(relative[p_index, level_index]), 0.95
                            ),
                            "absolute_bits_median": np.median(
                                absolute[p_index, level_index]
                            ),
                            "ratio_median": np.median(ratios[p_index, level_index]),
                        })
    print(f"Wrote optional robustness checks to {output}")


def write_caption(path: Path, result: SimulationResult) -> None:
    caption = (
        "Sensitivity of scalar-OVMI to unobserved decoder-channel structure for "
        f"the top-{result.vocabulary_size} SUBTLEX-UK vocabulary "
        f"($C(S)={result.coverage:.3f}$). Curves show the median signed relative "
        "error $(\mathrm{OVMI}_{\rm exact}-\mathrm{OVMI}_{\rm scalar})/"
        "\mathrm{OVMI}_{\rm scalar}$ across random channels; bands show the "
        "5th--95th percentiles. (a) Per-word accuracies vary while their arithmetic "
        "mean remains exactly $P$ and each row's errors remain uniform. The x-axis "
        "is $\mathrm{sd}(a_x)/\sqrt{P(1-P)}$. (b) Every word has accuracy exactly "
        "$P$, while off-diagonal errors follow independent symmetric Dirichlet "
        "distributions. Error concentration is one minus mean row-error entropy "
        "normalised by $\log_2(V-1)$. Zero on either x-axis is the homogeneous "
        "symmetric channel and therefore has zero estimation error. The reference "
        "distribution is treated as fixed; uncertainty bands describe synthetic "
        "channel variation, not sampling uncertainty. Frequency-aligned adversarial "
        "assignments for panel (a), absolute bit errors, and exact/scalar ratios are "
        "saved with the simulation data but omitted visually."
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(caption + "\n", encoding="utf-8")
    print(f"Wrote caption to {path}")


def main() -> None:
    args = parse_args()
    if args.reuse_cache:
        result = load_cache(args.cache)
        print(f"Loaded simulation cache from {args.cache}")
    else:
        reference = load_reference_slice(args.reference, MAIN_VOCABULARY_SIZE)
        result = run_simulation(reference, args.samples, args.seed)
        save_cache(result, args.cache)
        write_summary_csv(result, args.summary)
    report_main_findings(result)
    if not args.skip_robustness:
        run_optional_robustness(
            args.reference, args.robustness_samples, args.seed,
            args.robustness_output,
        )
    figure = draw_figure(result)
    save_figure(figure, args.output_base)
    plt.close(figure)
    write_caption(args.caption_output, result)


if __name__ == "__main__":
    main()
