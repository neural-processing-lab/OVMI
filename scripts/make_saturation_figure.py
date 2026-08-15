#!/usr/bin/env python3
"""Plot where lexical OVMI saturates and what the axis leaves out."""

from __future__ import annotations

import argparse
import contextlib
import io
import math
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
SCRIPT_DIR = PROJECT_ROOT / "scripts"
for directory in (SRC_DIR, SCRIPT_DIR):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from ovmi import ovmi  # noqa: E402
from ovmi.core import entropy  # noqa: E402
import make_main_table as table  # noqa: E402
from make_contour_figure import COLORS  # noqa: E402


DEFAULT_OUTPUT_BASE = PROJECT_ROOT / "figures/saturation"
DEFAULT_CAPTION_OUTPUT = PROJECT_ROOT / "figures/saturation_caption.md"
DEFAULT_CONTEXT = PROJECT_ROOT / "data/saturation_context.csv"
CURVE_VOCABULARY_SIZES = (50, 250, 1_000, 15_000, 125_000)
CURVE_COLORS = {
    50: "#0072B2",
    250: "#E69F00",
    1_000: "#009E73",
    15_000: "#D55E00",
    125_000: "#6F4C9B",
}
SATURATION_FRACTION = 0.95
CHECK_TOLERANCE = 1e-10


@dataclass(frozen=True)
class SaturationCurve:
    vocabulary_size: int
    vocabulary: list[str]
    probabilities: np.ndarray
    scores: np.ndarray
    asymptote: float
    p95: float
    color: str


@dataclass(frozen=True)
class PlottedPoint:
    point: table.SystemPoint
    label: str
    system_name: str
    color: str
    invasive: bool
    score: float
    p_ci: tuple[float, float] | None
    score_ci: tuple[float, float] | None
    frequency_curve_score: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--systems", type=Path, default=table.DEFAULT_SYSTEMS)
    parser.add_argument("--references-dir", type=Path, default=table.DEFAULT_REFERENCES)
    parser.add_argument("--predictions-dir", type=Path, default=table.DEFAULT_PREDICTIONS)
    parser.add_argument("--cmudict", type=Path, default=table.DEFAULT_CMUDICT)
    parser.add_argument("--armeni-text", type=Path, default=table.DEFAULT_ARMENI_TEXT)
    parser.add_argument(
        "--meg-masc-vocabulary", type=Path,
        default=table.DEFAULT_MEG_MASC_VOCABULARY,
    )
    parser.add_argument("--context", type=Path, default=DEFAULT_CONTEXT)
    parser.add_argument("--main-table", type=Path, default=table.DEFAULT_OUTPUT)
    parser.add_argument(
        "--appendix-table", type=Path, default=table.DEFAULT_APPENDIX_OUTPUT
    )
    parser.add_argument("--output-base", type=Path, default=DEFAULT_OUTPUT_BASE)
    parser.add_argument("--caption-output", type=Path, default=DEFAULT_CAPTION_OUTPUT)
    return parser.parse_args()


def load_context(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, keep_default_na=False)
    expected = [
        "system_id", "reported_wpm", "wpm_ci_low", "wpm_ci_high",
        "training_hours_to_first_125k_result", "p_ci_low", "p_ci_high",
        "source",
    ]
    if list(frame.columns) != expected:
        raise ValueError(f"{path} must contain exactly {expected}")
    if set(frame["system_id"]) != {"card_2024_v125k", "willett_2023_v125k"}:
        raise ValueError(f"{path} must contain exactly Card and Willett-125k")
    for column in (
        "reported_wpm", "training_hours_to_first_125k_result",
        "p_ci_low", "p_ci_high",
    ):
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    for column in ("wpm_ci_low", "wpm_ci_high"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if not bool((frame["p_ci_low"] < frame["p_ci_high"]).all()):
        raise ValueError("Context probability intervals must be ordered")
    return frame.set_index("system_id")


def load_pipeline(args: argparse.Namespace):
    systems = pd.read_csv(args.systems)
    systems = systems.loc[systems["plot_eligible"].astype(bool)].copy()
    references, entropies = table.load_references(args.references_dir)
    vocabularies = table.reconstruct_vocabularies(
        systems,
        references,
        args.predictions_dir,
        args.cmudict,
        args.armeni_text,
        args.meg_masc_vocabulary,
    )
    points = table.expand_system_points(systems)
    scores = table.score_all(points, vocabularies, references, entropies)
    table.validate_scores(points, scores)
    return systems, references, entropies, vocabularies, points, scores


def assert_main_table_agreement(
    main_path: Path,
    appendix_path: Path,
    points: list[table.SystemPoint],
    scores: dict[tuple[int, str], table.CellScore],
    entropies: dict[str, float],
    vocabularies: dict[str, list[str]],
    references: dict[str, dict[str, float]],
) -> None:
    uncertainties = table.score_uncertainties(
        points, vocabularies, references, entropies
    )
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        generated_main = root / "main.tex"
        generated_appendix = root / "appendix.tex"
        with contextlib.redirect_stdout(io.StringIO()):
            table.render_table(
                points, scores, uncertainties, entropies, generated_main
            )
            table.render_individual_target_appendix(
                points, scores, uncertainties, entropies["individual"],
                generated_appendix,
            )
        for generated, existing in (
            (generated_main, main_path),
            (generated_appendix, appendix_path),
        ):
            if generated.read_text(encoding="utf-8") != existing.read_text(
                encoding="utf-8"
            ):
                raise AssertionError(
                    f"Saturation pipeline disagrees with {existing}; regenerate the table"
                )
    print("CHECK main-table agreement: exact regeneration passed")


def frequency_vocabulary(reference: dict[str, float], size: int) -> list[str]:
    if size > len(reference):
        raise ValueError(f"Reference has {len(reference)} types; cannot select V={size}")
    return sorted(reference, key=lambda word: (-reference[word], word))[:size]


def scalar_curve(
    reference: dict[str, float], vocabulary: list[str], probabilities: np.ndarray,
) -> np.ndarray:
    """Vectorised Corollary-1 curve with bounded working memory."""

    reference_total = float(sum(reference.values()))
    raw = np.asarray([reference.get(word, 0.0) for word in vocabulary], dtype=float)
    coverage = float(raw.sum() / reference_total)
    conditional_weights = raw / raw.sum()
    vocabulary_size = len(vocabulary)
    scores = np.empty_like(probabilities, dtype=float)
    for start in range(0, len(probabilities), 32):
        stop = min(start + 32, len(probabilities))
        correct = probabilities[start:stop]
        error = (1.0 - correct) / (vocabulary_size - 1)
        output = error[:, None] + conditional_weights[None, :] * (
            correct - error
        )[:, None]
        with np.errstate(divide="ignore", invalid="ignore"):
            output_terms = np.where(output > 0, output * np.log2(output), 0.0)
            correct_terms = np.where(correct > 0, correct * np.log2(correct), 0.0)
            error_terms = np.where(error > 0, error * np.log2(error), 0.0)
        output_entropy = -output_terms.sum(axis=1)
        conditional_entropy = -correct_terms - (vocabulary_size - 1) * error_terms
        scores[start:stop] = coverage * (output_entropy - conditional_entropy)
    return np.maximum(scores, 0.0)


def probability_at_fraction_of_asymptote(
    reference: dict[str, float], vocabulary: list[str], asymptote: float,
    fraction: float = SATURATION_FRACTION,
) -> float:
    low, high = 1.0 / len(vocabulary), 1.0
    target = fraction * asymptote
    for _ in range(64):
        midpoint = (low + high) / 2.0
        score = float(ovmi(reference, vocabulary, accuracy=midpoint))
        if score >= target:
            high = midpoint
        else:
            low = midpoint
    return high


def build_curves(reference: dict[str, float]) -> dict[int, SaturationCurve]:
    probabilities = np.linspace(0.0, 1.0, 501)
    curves: dict[int, SaturationCurve] = {}
    for size in CURVE_VOCABULARY_SIZES:
        vocabulary = frequency_vocabulary(reference, size)
        asymptote = float(ovmi(reference, vocabulary, accuracy=1.0))
        p95 = probability_at_fraction_of_asymptote(
            reference, vocabulary, asymptote
        )
        scores = scalar_curve(reference, vocabulary, probabilities)
        direct = np.asarray(
            [float(ovmi(reference, vocabulary, accuracy=p)) for p in (0.1, 0.5, 1.0)]
        )
        vectorised = scalar_curve(
            reference, vocabulary, np.asarray([0.1, 0.5, 1.0])
        )
        if float(np.max(np.abs(direct - vectorised))) > CHECK_TOLERANCE:
            raise AssertionError(f"Vectorised curve disagrees with OVMI for V={size}")
        curves[size] = SaturationCurve(
            size, vocabulary, probabilities, scores, asymptote, p95,
            CURVE_COLORS[size],
        )
    return curves


def point_label(point: table.SystemPoint) -> str:
    if point.system_id == "meg_masc_2023_v50":
        return "MEG-MASC"
    if point.system_id == "dascoli_libribrain100_s0_v50":
        return "LibriBrain100"
    if point.system_id == "armeni_2022_v50":
        return "Armeni"
    vocabulary = "125k" if point.vocabulary_size == 125_000 else str(point.vocabulary_size)
    if point.system_id.startswith("willett_"):
        suffix = "+LM" if point.probability_source == "w" else "isolated"
        return f"Willett {vocabulary} {suffix}"
    if point.system_id == "moses_2021_v50":
        suffix = "+LM" if point.probability_source == "w" else "isolated"
        return f"Moses {suffix}"
    if point.system_id == "card_2024_v125k":
        return "Card 125k +LM"
    return point.display_name


def build_plotted_points(
    systems: pd.DataFrame,
    reference: dict[str, float],
    vocabularies: dict[str, list[str]],
    points: list[table.SystemPoint],
    context: pd.DataFrame,
    curves: dict[int, SaturationCurve],
) -> list[PlottedPoint]:
    plotted: list[PlottedPoint] = []
    own_curve_errors = []
    for point in points:
        row = systems.loc[systems["system_id"] == point.system_id].iloc[0]
        vocabulary = vocabularies[point.system_id]
        score = float(ovmi(reference, vocabulary, accuracy=point.probability))
        own_curve_errors.append(abs(score - point.csv_ovmi))
        p_ci: tuple[float, float] | None = None
        score_ci: tuple[float, float] | None = None
        neural_probability = pd.to_numeric(row["P_neural"], errors="coerce")
        if math.isfinite(neural_probability) and math.isclose(
            point.probability, float(neural_probability), abs_tol=1e-12
        ):
            low = pd.to_numeric(row["P_neural_ci_low"], errors="coerce")
            high = pd.to_numeric(row["P_neural_ci_high"], errors="coerce")
            if math.isfinite(low) and math.isfinite(high):
                p_ci = (float(low), float(high))
        if point.system_id in context.index:
            context_row = context.loc[point.system_id]
            p_ci = (float(context_row["p_ci_low"]), float(context_row["p_ci_high"]))
        if p_ci is not None:
            score_ci = (
                float(ovmi(reference, vocabulary, accuracy=p_ci[0])),
                float(ovmi(reference, vocabulary, accuracy=p_ci[1])),
            )
        frequency_score = float(
            ovmi(
                reference,
                curves[point.vocabulary_size].vocabulary,
                accuracy=point.probability,
            )
        )
        plotted.append(PlottedPoint(
            point=point,
            label=point_label(point),
            system_name=str(row["system_name"]),
            color=COLORS.get(str(row["system_name"]), "#374151"),
            invasive=str(row["invasiveness"]) == "invasive",
            score=score,
            p_ci=p_ci,
            score_ci=score_ci,
            frequency_curve_score=frequency_score,
        ))
    maximum_error = max(own_curve_errors, default=0.0)
    if maximum_error > CHECK_TOLERANCE:
        raise AssertionError("A system point does not lie on its own-vocabulary curve")
    print(
        "CHECK own-vocabulary curve agreement: "
        f"{len(plotted)} points; max absolute error={maximum_error:.3g} bits"
    )
    return plotted


def configure_style() -> None:
    matplotlib.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 9.0,
        "axes.labelsize": 10.5,
        "axes.titlesize": 10.5,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "axes.linewidth": 0.8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


POINT_OFFSETS = {
    "MEG-MASC": (5, 8, "left"),
    "Armeni": (5, 16, "left"),
    "LibriBrain100": (5, 27, "left"),
    "Moses isolated": (5, -12, "left"),
    "Moses +LM": (-5, 13, "right"),
    "Willett 50 +LM": (-5, 20, "right"),
    "Willett 50 isolated": (-5, 34, "right"),
    "Willett 125k +LM": (-7, 9, "right"),
    "Card 125k +LM": (-7, -13, "right"),
}


def draw_main_panel(
    axis,
    curves: dict[int, SaturationCurve],
    plotted: list[PlottedPoint],
    reference_entropy: float,
) -> None:
    thresholds = np.asarray([curve.p95 for curve in curves.values()])
    first_saturated = float(thresholds.min())
    all_saturated = float(thresholds.max())
    axis.axvspan(0.0, first_saturated, color="#DCEAF4", alpha=0.34, zorder=-5)
    axis.axvspan(first_saturated, all_saturated, color="#F3E6C5", alpha=0.55, zorder=-5)
    axis.axvspan(all_saturated, 1.0, color="#D9DCE1", alpha=0.70, zorder=-5)

    curve_handles = []
    for size, curve in curves.items():
        before = curve.probabilities <= curve.p95
        after = curve.probabilities >= curve.p95
        axis.plot(
            curve.probabilities[before], curve.scores[before],
            color=curve.color, linewidth=1.8, solid_capstyle="round", zorder=1,
        )
        axis.plot(
            curve.probabilities[after], curve.scores[after],
            color=curve.color, linewidth=1.5, linestyle=(0, (3, 2)),
            alpha=0.68, zorder=1,
        )
        label = f"V={size:,}" if size != 125_000 else "V=125k"
        curve_handles.append(Line2D([], [], color=curve.color, lw=1.8, label=label))

    axis.axhline(
        reference_entropy, color="#4B5563", linewidth=1.0,
        linestyle=(0, (4, 2.5)), zorder=0,
    )
    axis.text(
        0.985, reference_entropy + 0.10, rf"$H(p)={reference_entropy:.2f}$ bits",
        ha="right", va="bottom", fontsize=8.4, color="#374151",
    )
    axis.axvline(first_saturated, color="#9A7B36", lw=0.75, ls=(0, (2, 2)), zorder=0)
    axis.axvline(all_saturated, color="#6B7280", lw=0.75, ls=(0, (2, 2)), zorder=0)

    for item in plotted:
        x = item.point.probability
        y = item.score
        axis.plot(
            [x, x], [item.frequency_curve_score, y],
            color=item.color, linewidth=0.65, linestyle=(0, (1.3, 1.8)),
            alpha=0.35, zorder=2,
        )
        axis.scatter(
            x, y, s=38, marker="o",
            facecolor=item.color if item.invasive else "white",
            edgecolor=item.color, linewidth=1.35, zorder=5,
        )
        dx, dy, alignment = POINT_OFFSETS[item.label]
        axis.annotate(
            item.label, (x, y), xytext=(dx, dy), textcoords="offset points",
            ha=alignment, va="center", color=item.color,
            fontsize=7.1, fontweight="semibold", zorder=6,
        )

    legend = axis.legend(
        handles=curve_handles,
        title="Top-frequency vocabulary",
        loc="upper left", bbox_to_anchor=(0.015, 0.925),
        ncol=2, frameon=False, borderaxespad=0.0,
        columnspacing=1.0, handlelength=2.0, handletextpad=0.5,
        fontsize=7.6, title_fontsize=8.0,
    )
    axis.add_artist(legend)
    group_handles = [
        Line2D([], [], marker="o", linestyle="none", markersize=5.2,
               markerfacecolor="white", markeredgecolor="#374151",
               label="perceived / non-invasive"),
        Line2D([], [], marker="o", linestyle="none", markersize=5.2,
               markerfacecolor="#6B7280", markeredgecolor="#374151",
               label="attempted / invasive"),
    ]
    axis.legend(
        handles=group_handles, loc="upper left", bbox_to_anchor=(0.39, 0.925),
        frameon=False, borderaxespad=0.0, fontsize=7.4,
        handletextpad=0.35, labelspacing=0.35,
    )

    axis.text(
        0.045, 6.85,
        "Current field  (OVMI / $H(p)$)\n"
        "non-invasive: 0.4–2.4%\n"
        "invasive, V=50: 2.4–6.7%\n"
        "invasive, V=125k: 72.0–93.7%",
        ha="left", va="top", fontsize=7.8, color="#374151",
        bbox={"boxstyle": "round,pad=0.3", "fc": "white", "ec": "#CBD0D6", "lw": 0.7},
        zorder=7,
    )
    axis.text(
        0.70, 9.38, "all retain >5% headroom",
        ha="center", va="center", fontsize=7.2, color="#476579",
    )
    axis.text(
        (first_saturated + all_saturated) / 2.0, 0.22, "mixed",
        ha="center", va="bottom", rotation=90, fontsize=6.8, color="#795F27",
    )
    axis.text(
        (all_saturated + 1.0) / 2.0, 0.22, "all ≥95%",
        ha="center", va="bottom", rotation=90, fontsize=6.8, color="#4B5563",
    )
    axis.text(
        -0.055, 1.035, "a", transform=axis.transAxes,
        fontsize=11.5, fontweight="bold", ha="left", va="bottom",
    )
    axis.set_xlabel(r"Per-symbol accuracy $P$")
    axis.set_ylabel("OVMI (bits)")
    axis.set_xlim(0.0, 1.01)
    axis.set_ylim(0.0, 10.25)
    axis.set_xticks(np.linspace(0.0, 1.0, 6))
    axis.set_yticks(np.arange(0.0, 10.1, 2.0))
    axis.grid(axis="y", color="#D5D9DE", linewidth=0.55, alpha=0.65, zorder=-4)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.tick_params(length=3, color="#4B5563")


def _metric_bars(axis, values, colors, xlim, xlabel) -> None:
    y = np.asarray([1.0, 0.0])
    axis.barh(y, values, height=0.34, color=colors, alpha=0.84, zorder=2)
    for value, y_position, color in zip(values, y, colors):
        axis.text(
            value + 0.025 * xlim, y_position, f"{value:g}",
            ha="left", va="center", fontsize=8.0, color=color,
            fontweight="semibold",
        )
    axis.set_yticks(y, ["Card", "Willett"])
    axis.set_xlim(0.0, xlim)
    axis.set_xlabel(xlabel, fontsize=8.5)
    axis.grid(axis="x", color="#D5D9DE", linewidth=0.55, alpha=0.7, zorder=0)
    for spine in ("top", "right", "left"):
        axis.spines[spine].set_visible(False)
    axis.tick_params(axis="y", length=0, pad=4)
    axis.tick_params(axis="x", length=2.5)


def draw_context_panel(
    info_axis, wpm_axis, training_axis,
    context: pd.DataFrame,
    interval_report: dict[str, tuple[float, float, float]],
) -> None:
    card_color = COLORS["Card et al."]
    willett_color = COLORS["Willett et al."]
    colors = [card_color, willett_color]
    card_score = interval_report["card_2024_v125k"][0]
    willett_score = interval_report["willett_2023_v125k"][0]
    info_axis.axis("off")
    info_axis.text(
        -0.05, 1.12, "b", transform=info_axis.transAxes,
        fontsize=11.5, fontweight="bold", ha="left", va="bottom",
    )
    info_axis.text(
        0.04, 1.12, "What lexical OVMI omits", transform=info_axis.transAxes,
        fontsize=10.2, fontweight="semibold", ha="left", va="bottom",
    )
    info_axis.text(
        0.0, 0.68,
        "OVMI from published WER estimates",
        transform=info_axis.transAxes, fontsize=8.0, color="#374151",
        ha="left", va="top",
    )
    info_axis.text(
        0.0, 0.38,
        f"Card   {card_score:.2f} bits",
        transform=info_axis.transAxes, fontsize=7.7, color=card_color,
        ha="left", va="top", fontweight="semibold",
    )
    info_axis.text(
        0.0, 0.08,
        f"Willett {willett_score:.2f} bits",
        transform=info_axis.transAxes, fontsize=7.7, color=willett_color,
        ha="left", va="top", fontweight="semibold",
    )

    card = context.loc["card_2024_v125k"]
    willett = context.loc["willett_2023_v125k"]
    wpm_values = np.asarray([card["reported_wpm"], willett["reported_wpm"]], dtype=float)
    _metric_bars(
        wpm_axis, wpm_values, colors, 70.0, "Reported words per minute",
    )
    training_values = np.asarray([
        card["training_hours_to_first_125k_result"],
        willett["training_hours_to_first_125k_result"],
    ], dtype=float)
    _metric_bars(
        training_axis, training_values, colors, 19.0,
        "Training data before first\nreported 125k result (h)",
    )


def draw_figure(
    curves: dict[int, SaturationCurve],
    plotted: list[PlottedPoint],
    reference_entropy: float,
    context: pd.DataFrame,
    interval_report: dict[str, tuple[float, float, float]],
):
    configure_style()
    figure = plt.figure(figsize=(9.0, 3.8))
    outer = figure.add_gridspec(
        1, 2, width_ratios=(3.15, 1.15), wspace=0.30,
        left=0.075, right=0.985, top=0.91, bottom=0.17,
    )
    main_axis = figure.add_subplot(outer[0, 0])
    right = outer[0, 1].subgridspec(3, 1, height_ratios=(0.62, 1.0, 1.0), hspace=0.66)
    info_axis = figure.add_subplot(right[0, 0])
    wpm_axis = figure.add_subplot(right[1, 0])
    training_axis = figure.add_subplot(right[2, 0])
    draw_main_panel(main_axis, curves, plotted, reference_entropy)
    draw_context_panel(info_axis, wpm_axis, training_axis, context, interval_report)
    return figure


def save_figure(figure, output_base: Path) -> None:
    output_base.parent.mkdir(parents=True, exist_ok=True)
    for extension in ("pdf", "png"):
        output = output_base.with_suffix(f".{extension}")
        figure.savefig(output, dpi=400 if extension == "png" else None)
        print(f"Saved {output}")


def interval_report(
    systems: pd.DataFrame,
    reference: dict[str, float],
    vocabularies: dict[str, list[str]],
    context: pd.DataFrame,
) -> dict[str, tuple[float, float, float]]:
    report = {}
    for system_id in ("card_2024_v125k", "willett_2023_v125k"):
        row = context.loc[system_id]
        probability = float(
            systems.loc[systems["system_id"] == system_id, "P_system"].iloc[0]
        )
        vocabulary = vocabularies[system_id]
        report[system_id] = (
            float(ovmi(reference, vocabulary, accuracy=probability)),
            float(ovmi(reference, vocabulary, accuracy=float(row["p_ci_low"]))),
            float(ovmi(reference, vocabulary, accuracy=float(row["p_ci_high"]))),
        )
    return report


def field_percentages(
    plotted: list[PlottedPoint], reference_entropy: float,
    reference: dict[str, float], vocabularies: dict[str, list[str]],
) -> dict[str, tuple[float, float, float, float]]:
    groups = {
        "non-invasive": [item for item in plotted if not item.invasive],
        "invasive V=50": [
            item for item in plotted
            if item.invasive and item.point.vocabulary_size == 50
        ],
        "invasive V=125k": [
            item for item in plotted
            if item.invasive and item.point.vocabulary_size == 125_000
        ],
    }
    result = {}
    for name, members in groups.items():
        global_percent = [100.0 * item.score / reference_entropy for item in members]
        own_percent = [
            100.0 * item.score / float(
                ovmi(
                    reference, vocabularies[item.point.system_id], accuracy=1.0
                )
            )
            for item in members
        ]
        result[name] = (
            min(global_percent), max(global_percent),
            min(own_percent), max(own_percent),
        )
    return result


def write_caption(
    path: Path,
    curves: dict[int, SaturationCurve],
    reference_entropy: float,
    percentages: dict[str, tuple[float, float, float, float]],
    intervals: dict[str, tuple[float, float, float]],
) -> None:
    first = min(curve.p95 for curve in curves.values())
    last = max(curve.p95 for curve in curves.values())
    card_score = intervals["card_2024_v125k"][0]
    willett_score = intervals["willett_2023_v125k"][0]
    gap = card_score - willett_score
    noninvasive = percentages["non-invasive"]
    invasive50 = percentages["invasive V=50"]
    invasive125 = percentages["invasive V=125k"]
    caption = (
        "Lexical OVMI states its own expiry date. (a) Curves apply Corollary 1 "
        "to top-frequency SUBTLEX-UK vocabularies. Each curve's asymptote is "
        "$C(S)H(p_S)$; as V grows this approaches the full-reference entropy "
        f"$H(p)={reference_entropy:.2f}$ bits. "
        "Saturation is defined before looking at the plotted systems: a curve is "
        "saturated once it reaches 95% of its own asymptote, leaving at most 5% "
        "lexical headroom. Solid segments retain more than 5% headroom and dashed "
        "segments do not. Pale blue marks the range in which every curve retains "
        f"more than 5% headroom ($P<{first:.3f}$), amber is the vocabulary-dependent "
        f"transition, and grey marks the range in which every curve is saturated "
        f"($P>{last:.3f}$). Real points use each system's documented vocabulary; "
        "faint vertical stems show their offset from the representative top-frequency "
        "curve. Numerical validation is therefore against each system's own-vocabulary "
        "curve, not the representative frequency-selected curve. Under the full "
        f"reference, non-invasive systems occupy {noninvasive[0]:.1f}--{noninvasive[1]:.1f}% "
        f"of $H(p)$, invasive V=50 points {invasive50[0]:.1f}--{invasive50[1]:.1f}% "
        f"({invasive50[2]:.1f}--{invasive50[3]:.1f}% of their own vocabulary ceiling), "
        f"and invasive V=125k systems {invasive125[0]:.1f}--{invasive125[1]:.1f}%. "
        "(b) Quantities omitted by lexical OVMI can reverse the practical comparison: "
        "Willett reports 62 WPM and Card 31.6 WPM, while the training data preceding "
        "their first reported 125k results were 16.8 h and 1.9 h, respectively. This "
        "training quantity is a reproducible milestone, not a matched time-to-equal-accuracy "
        "measure. Their current OVMIs are not near-identical: Card exceeds Willett by "
        f"{gap:.3f} bits. Figure points omit uncertainty for visual clarity; propagated "
        "intervals remain reported in the main comparison table. Thus the present data "
        "still discriminate these systems, "
        "while the theoretical frontier identifies where rate, latency, calibration, and "
        "usability must take over. Rate sources: Card et al. (2024), "
        "https://doi.org/10.1056/NEJMoa2314132; Willett et al. (2023), "
        "https://doi.org/10.1038/s41586-023-06377-x."
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(caption + "\n", encoding="utf-8")
    print(f"Wrote caption to {path}")


def main() -> None:
    args = parse_args()
    systems, references, entropies, vocabularies, points, scores = load_pipeline(args)
    assert_main_table_agreement(
        args.main_table, args.appendix_table, points, scores, entropies,
        vocabularies, references,
    )
    reference = references["subtlex"]
    computed_entropy = entropy(np.asarray(list(reference.values()), dtype=float))
    if abs(computed_entropy - entropies["subtlex"]) > CHECK_TOLERANCE:
        raise AssertionError("SUBTLEX entropy disagreement")
    curves = build_curves(reference)
    context = load_context(args.context)
    plotted = build_plotted_points(
        systems, reference, vocabularies, points, context, curves
    )
    intervals = interval_report(systems, reference, vocabularies, context)
    percentages = field_percentages(
        plotted, entropies["subtlex"], reference, vocabularies
    )

    print("P at 95% of each frequency-selected curve's own asymptote:")
    for size, curve in curves.items():
        print(
            f"  V={size:,}: P95={curve.p95:.6f}; "
            f"asymptote={curve.asymptote:.6f} bits"
        )
    print("Offsets from representative frequency-selected V-curve:")
    for item in plotted:
        difference = item.score - item.frequency_curve_score
        status = "within tolerance" if abs(difference) <= CHECK_TOLERANCE else "own vocabulary differs"
        print(f"  {item.label}: {difference:+.6f} bits ({status})")

    card = intervals["card_2024_v125k"]
    willett = intervals["willett_2023_v125k"]
    gap = card[0] - willett[0]
    nonoverlap = card[1] > willett[2] or willett[1] > card[2]
    print(
        "Card vs Willett-125k: "
        f"difference={gap:.6f} bits; Card CI=[{card[1]:.6f}, {card[2]:.6f}]; "
        f"Willett CI=[{willett[1]:.6f}, {willett[2]:.6f}]; "
        f"intervals overlap={'no' if nonoverlap else 'yes'}"
    )
    for name, values in percentages.items():
        print(
            f"Field position {name}: {values[0]:.1f}-{values[1]:.1f}% H(p); "
            f"{values[2]:.1f}-{values[3]:.1f}% own ceiling"
        )

    figure = draw_figure(
        curves, plotted, entropies["subtlex"], context, intervals
    )
    save_figure(figure, args.output_base)
    plt.close(figure)
    write_caption(
        args.caption_output, curves, entropies["subtlex"], percentages, intervals
    )


if __name__ == "__main__":
    main()
