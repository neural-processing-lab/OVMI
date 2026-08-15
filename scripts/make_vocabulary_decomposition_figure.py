#!/usr/bin/env python3
"""Plot noiseless top-V information and its exact loss decomposition."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import math
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402
import numpy as np  # noqa: E402


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
SCRIPT_DIR = PROJECT_ROOT / "scripts"
for directory in (SRC_DIR, SCRIPT_DIR):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from ovmi import ovmi  # noqa: E402
import make_main_table as table  # noqa: E402


DEFAULT_REFERENCE = PROJECT_ROOT / "data/references/subtlex_uk.csv"
DEFAULT_OUTPUT_BASE = PROJECT_ROOT / "figures/vocabulary_decomposition"
DEFAULT_CAPTION_OUTPUT = PROJECT_ROOT / "figures/vocabulary_decomposition_caption.md"
REPRESENTATIVE_SIZES = (50, 250, 1_000, 5_000, 15_000)
MAX_V = max(REPRESENTATIVE_SIZES)
CHECK_TOLERANCE = 1e-10

COLORS = {
    "log_capacity": "#4B5563",
    "conditional_entropy": "#E69F00",
    "retained": "#0072B2",
    "coverage_loss": "#D55E00",
    "nonuniformity": "#9CA3AF",
}


@dataclass(frozen=True)
class VocabularyMetrics:
    sizes: np.ndarray
    ranked_words: tuple[str, ...]
    coverage: np.ndarray
    conditional_entropy: np.ndarray
    retained_ovmi: np.ndarray
    log_capacity: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--max-v", type=int, default=MAX_V)
    parser.add_argument("--output-base", type=Path, default=DEFAULT_OUTPUT_BASE)
    parser.add_argument(
        "--caption-output", type=Path, default=DEFAULT_CAPTION_OUTPUT
    )
    return parser.parse_args()


def compute_metrics(
    reference: dict[str, float], max_v: int = MAX_V,
) -> VocabularyMetrics:
    if max_v < max(REPRESENTATIVE_SIZES):
        raise ValueError(
            f"max_v must be at least {max(REPRESENTATIVE_SIZES):,}"
        )
    if max_v > len(reference):
        raise ValueError(
            f"max_v={max_v:,} exceeds the {len(reference):,}-word reference"
        )

    ranked = sorted(reference.items(), key=lambda item: (-item[1], item[0]))
    words = tuple(word for word, _ in ranked[:max_v])
    weights = np.asarray([weight for _, weight in ranked], dtype=np.float64)
    if np.any(~np.isfinite(weights)) or np.any(weights <= 0):
        raise ValueError("Reference weights must be positive and finite")
    probabilities = weights / weights.sum()
    selected = probabilities[:max_v]

    sizes = np.arange(1, max_v + 1, dtype=np.int64)
    coverage = np.cumsum(selected)
    partial_entropy = np.cumsum(-selected * np.log2(selected))
    conditional_entropy = partial_entropy / coverage + np.log2(coverage)
    log_capacity = np.log2(sizes.astype(np.float64))
    retained_ovmi = coverage * conditional_entropy

    metrics = VocabularyMetrics(
        sizes=sizes,
        ranked_words=words,
        coverage=coverage,
        conditional_entropy=conditional_entropy,
        retained_ovmi=retained_ovmi,
        log_capacity=log_capacity,
    )
    validate_metrics(reference, metrics)
    return metrics


def components_at(
    metrics: VocabularyMetrics,
    vocabulary_sizes: tuple[int, ...] = REPRESENTATIVE_SIZES,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    indices = np.asarray(vocabulary_sizes, dtype=int) - 1
    coverage = metrics.coverage[indices]
    conditional_entropy = metrics.conditional_entropy[indices]
    retained = metrics.retained_ovmi[indices]
    coverage_loss = (1.0 - coverage) * conditional_entropy
    nonuniformity = metrics.log_capacity[indices] - conditional_entropy
    totals = metrics.log_capacity[indices]
    return retained, coverage_loss, nonuniformity, totals


def validate_metrics(
    reference: dict[str, float], metrics: VocabularyMetrics,
) -> None:
    if not np.all(np.diff(metrics.sizes) == 1):
        raise AssertionError("Vocabulary sizes must be consecutive")
    if np.any(np.diff(metrics.coverage) <= 0):
        raise AssertionError("Top-V coverage must increase strictly")
    if np.any(metrics.conditional_entropy < -CHECK_TOLERANCE):
        raise AssertionError("Conditional entropy cannot be negative")
    if np.any(metrics.conditional_entropy - metrics.log_capacity > CHECK_TOLERANCE):
        raise AssertionError("H(p_S) cannot exceed log2(V)")
    if np.any(metrics.retained_ovmi - metrics.conditional_entropy > CHECK_TOLERANCE):
        raise AssertionError("C(S)H(p_S) cannot exceed H(p_S)")

    retained, coverage_loss, nonuniformity, totals = components_at(metrics)
    components = np.vstack([retained, coverage_loss, nonuniformity])
    if np.any(components < -CHECK_TOLERANCE):
        raise AssertionError("Decomposition components must be non-negative")
    error = np.max(np.abs(components.sum(axis=0) - totals))
    if error > CHECK_TOLERANCE:
        raise AssertionError(
            f"Decomposition does not sum to log2(V); maximum error={error}"
        )

    for vocabulary_size in REPRESENTATIVE_SIZES:
        vocabulary = metrics.ranked_words[:vocabulary_size]
        direct = float(ovmi(reference, vocabulary, accuracy=1.0))
        computed = float(metrics.retained_ovmi[vocabulary_size - 1])
        if not math.isclose(direct, computed, rel_tol=0.0, abs_tol=CHECK_TOLERANCE):
            raise AssertionError(
                f"P=1 OVMI mismatch at V={vocabulary_size}: {direct} != {computed}"
            )


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
        "legend.fontsize": 8.2,
        "lines.linewidth": 2.4,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def format_vocabulary_tick(value: float, _position: int | None = None) -> str:
    integer = int(round(value))
    if integer >= 1_000 and integer % 1_000 == 0:
        return f"{integer // 1_000}k"
    return f"{integer:,}"


def draw_figure(metrics: VocabularyMetrics):
    configure_style()
    figure, (left, right) = plt.subplots(
        1, 2, figsize=(8.0, 3.0), sharey=True,
        gridspec_kw={"width_ratios": (1.12, 1.0), "wspace": 0.18},
    )
    figure.subplots_adjust(left=0.078, right=0.985, bottom=0.19, top=0.86)

    representative_indices = np.asarray(REPRESENTATIVE_SIZES) - 1
    line_specs = (
        (
            metrics.log_capacity, COLORS["log_capacity"], (0, (5, 2.4)), "s",
            r"Wolpaw (uniform prior)  $\log_2 V$",
        ),
        (
            metrics.conditional_entropy, COLORS["conditional_entropy"],
            (0, (1.2, 1.5)), "^", r"In-vocabulary MI  $H(p_S)$",
        ),
        (
            metrics.retained_ovmi, COLORS["retained"], "-", "o",
            r"OVMI  $C(S)H(p_S)$",
        ),
    )
    for values, color, linestyle, marker, label in line_specs:
        left.plot(
            metrics.sizes[1:], values[1:], color=color, linestyle=linestyle,
            label=label,
        )
        left.scatter(
            metrics.sizes[representative_indices], values[representative_indices],
            s=24, marker=marker, facecolor=color, edgecolor="white",
            linewidth=0.6, zorder=4,
        )

    tick_sizes = (10, 50, 250, 1_000, 5_000, 15_000)
    left.set_xscale("log")
    left.set_xlim(2, metrics.sizes[-1] * 1.08)
    left.set_xticks(tick_sizes)
    left.xaxis.set_major_formatter(FuncFormatter(format_vocabulary_tick))
    left.minorticks_off()
    left.set_xlabel(r"Top-$V$ vocabulary size")
    left.set_ylabel("Bits")
    left.set_title("Ideal decoder (top-frequency vocabulary)", pad=7)
    left.legend(loc="upper left", frameon=False, handlelength=2.8)

    retained, coverage_loss, nonuniformity, _totals = components_at(metrics)
    positions = np.arange(len(REPRESENTATIVE_SIZES))
    bar_width = 0.68
    right.bar(
        positions, retained, width=bar_width, color=COLORS["retained"],
        edgecolor="white", linewidth=0.8,
        label=r"OVMI  $C(S)H(p_S)$",
    )
    right.bar(
        positions, coverage_loss, bottom=retained, width=bar_width,
        color=COLORS["coverage_loss"], edgecolor="white", linewidth=0.8,
        hatch="///", label=r"Coverage gap  $(1-C)H(p_S)$",
    )
    right.bar(
        positions, nonuniformity, bottom=retained + coverage_loss,
        width=bar_width, color=COLORS["nonuniformity"], edgecolor="white",
        linewidth=0.8, hatch="...",
        label=r"Non-uniformity  $\log_2V-H(p_S)$",
    )
    right.set_xticks(
        positions,
        [format_vocabulary_tick(value) for value in REPRESENTATIVE_SIZES],
    )
    right.set_xlabel(r"Top-$V$ vocabulary size")
    right.set_title("Sources of overestimation", pad=7)
    handles, labels = right.get_legend_handles_labels()
    right.legend(
        handles[::-1], labels[::-1], loc="upper left", frameon=False,
        handlelength=2.6,
    )
    right.tick_params(axis="y", labelleft=False)

    y_max = math.ceil(float(metrics.log_capacity[-1])) + 0.6
    left.set_ylim(0.0, y_max)
    for axis in (left, right):
        axis.grid(axis="both", color="#D5D9DE", linewidth=0.65, alpha=0.72)
        axis.set_axisbelow(True)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.tick_params(length=3.2, width=0.9, color="#4B5563")

    left.text(
        -0.14, 1.08, "a", transform=left.transAxes, fontsize=12.0,
        fontweight="bold", ha="left", va="bottom",
    )
    right.text(
        -0.10, 1.08, "b", transform=right.transAxes, fontsize=12.0,
        fontweight="bold", ha="left", va="bottom",
    )
    return figure


def save_figure(figure, output_base: Path) -> None:
    output_base.parent.mkdir(parents=True, exist_ok=True)
    for extension in ("pdf", "png"):
        path = output_base.with_suffix(f".{extension}")
        figure.savefig(path, dpi=400 if extension == "png" else None)
        print(f"Saved {path}")


def write_caption(path: Path) -> None:
    caption = (
        "Noiseless top-frequency vocabulary decomposition under SUBTLEX-UK. "
        "(a) With per-symbol accuracy fixed at $P=1$, in-vocabulary mutual "
        "information equals $H(p_S)$ and OVMI equals $C(S)H(p_S)$. The uniform "
        "Wolpaw measure $\\log_2 V$ under a uniform prior ignores both the "
        "non-uniform reference "
        "distribution and intent outside the decoder vocabulary. (b) Each bar "
        "is the exact identity $\\log_2V=C(S)H(p_S)+(1-C(S))H(p_S)+"
        "[\\log_2V-H(p_S)]$. Thus the blue portion is OVMI, the orange portion "
        "is the coverage gap, and the grey portion is the "
        "correction for non-uniform in-vocabulary word probabilities. Vocabularies "
        "contain the top $V$ SUBTLEX-UK words by frequency."
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(caption + "\n", encoding="utf-8")
    print(f"Wrote caption to {path}")


def report(metrics: VocabularyMetrics) -> None:
    retained, coverage_loss, nonuniformity, totals = components_at(metrics)
    print("Representative exact decompositions (bits):")
    for index, vocabulary_size in enumerate(REPRESENTATIVE_SIZES):
        coverage = metrics.coverage[vocabulary_size - 1]
        entropy_value = metrics.conditional_entropy[vocabulary_size - 1]
        print(
            f"  V={vocabulary_size:>6,}: C={coverage:.6f}; "
            f"H(p_S)={entropy_value:.6f}; retained={retained[index]:.6f}; "
            f"no coverage={coverage_loss[index]:.6f}; "
            f"non-uniformity={nonuniformity[index]:.6f}; "
            f"total={totals[index]:.6f}"
        )
    error = np.max(np.abs(retained + coverage_loss + nonuniformity - totals))
    print(f"CHECK stacked total equals log2(V): max error={error:.3g} bits")
    print("CHECK P=1 OVMI equals C(S)H(p_S) at all representative V")


def main() -> None:
    args = parse_args()
    reference = table.load_reference(args.reference)
    metrics = compute_metrics(reference, args.max_v)
    report(metrics)
    figure = draw_figure(metrics)
    save_figure(figure, args.output_base)
    plt.close(figure)
    write_caption(args.caption_output)


if __name__ == "__main__":
    main()
