#!/usr/bin/env python3
"""Plot speech decoders on the coverage x in-vocabulary-information plane."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.transforms import ScaledTranslation
import numpy as np
import pandas as pd

from ovmi import ovmi


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_ROOT / "data/systems.csv"
DEFAULT_FRONTIER_INPUT = PROJECT_ROOT / "data/noiseless_frequency_frontier.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "figures"
CONTOUR_LEVELS = (0.01, 0.1, 1.0, 3.0, 10.0)
REFERENCE_STEMS = {"subtlex-uk": "subtlex"}

COLORS = {
    "MEG-MASC": "#0072B2",
    "LibriBrain100": "#E69F00",
    "Armeni et al.": "#6F4C9B",
    "Tang et al.": "#56B4E9",
    "Moses et al.": "#D55E00",
    "Willett et al.": "#009E73",
    "Card et al.": "#CC79A7",
}
STYLED_POINT_STYLES = {
    ("meg_masc_2023_v50", "neural"): ("o", "MEG-MASC 2023 (V=50)"),
    ("dascoli_libribrain100_s0_v50", "neural"): ("s", "LibriBrain100 2025 (V=50)"),
    ("armeni_2022_v50", "neural"): ("D", "Armeni 2022 (V=50)"),
    ("tang_2023_v6867", "system"): ("p", "Tang 2023 +LM (V=6,867)"),
    ("moses_2021_v50", "neural"): ("^", "Moses 2021 isolated (V=50)"),
    ("moses_2021_v50", "system"): ("v", "Moses 2021 +LM (V=50)"),
    ("willett_2023_v50", "neural"): ("P", "Willett 2023 isolated (V=50)"),
    ("willett_2023_v50", "system"): ("X", "Willett 2023 +LM (V=50)"),
    ("willett_2023_v125k", "system"): ("h", "Willett 2023 +LM (V=125k)"),
    ("card_2024_v125k", "system"): ("*", "Card 2024 +LM (V=125k)"),
}
LABEL_OFFSETS = {
    ("meg_masc_2023_v50", "neural"): (7, -12),
    ("dascoli_libribrain100_s0_v50", "neural"): (7, 12),
    ("armeni_2022_v50", "neural"): (7, 2),
    ("tang_2023_v6867", "system"): (7, 8),
    ("moses_2021_v50", "neural"): (8, -14),
    ("moses_2021_v50", "system"): (8, -12),
    ("willett_2023_v50", "neural"): (-9, 17),
    ("willett_2023_v50", "system"): (8, 4),
    ("willett_2023_v125k", "system"): (-12, -18),
    ("card_2024_v125k", "system"): (-12, -5),
}
LABEL_ALIGNMENTS = {
    ("willett_2023_v50", "neural"): "right",
    ("willett_2023_v125k", "system"): "right",
    ("card_2024_v125k", "system"): "right",
}
LINEAR_LABEL_OFFSETS = {
    ("meg_masc_2023_v50", "neural"): (16, 8),
    ("dascoli_libribrain100_s0_v50", "neural"): (6, 40),
    ("armeni_2022_v50", "neural"): (7, 14),
    ("moses_2021_v50", "neural"): (-8, -5),
    ("moses_2021_v50", "system"): (-8, 0),
}
LINEAR_LABEL_ALIGNMENTS = {
    ("moses_2021_v50", "neural"): "right",
    ("moses_2021_v50", "system"): "right",
}
POINT_DODGE_POINTS = {
    ("willett_2023_v50", "neural"): -5.0,
    ("willett_2023_v50", "system"): 5.0,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--frontier-input", type=Path, default=DEFAULT_FRONTIER_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--reference", default="subtlex-uk")
    parser.add_argument(
        "--panels",
        action="store_true",
        help="Emit one panel for every reference present in the CSV.",
    )
    parser.add_argument(
        "--neural-only",
        action="store_true",
        help="Suppress LM-assisted P_system coordinates and show only P_neural.",
    )
    parser.add_argument(
        "--styled",
        action="store_true",
        help=(
            "Emit a separate _styled variant matching the typography and visual "
            "hierarchy of the vocabulary-decomposition figure."
        ),
    )
    return parser.parse_args()


def configure_matplotlib(styled: bool = False) -> None:
    settings = {
        "font.family": "sans-serif",
        "font.sans-serif": [
            "Source Sans 3", "Arial", "Helvetica", "DejaVu Sans",
        ],
        "mathtext.fontset": "dejavusans",
        "font.size": 10.5,
        "axes.labelsize": 12.0,
        "axes.labelweight": "semibold",
        "axes.titlesize": 11.0,
        "legend.fontsize": 10.0,
        "xtick.labelsize": 10.5,
        "ytick.labelsize": 10.5,
        "axes.linewidth": 1.15,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.bbox": None,
    }
    if styled:
        settings.update({
            "font.size": 9.0,
            "axes.labelsize": 10.5,
            "axes.labelweight": "normal",
            "axes.titlesize": 10.5,
            "axes.titleweight": "semibold",
            "legend.fontsize": 9.0,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "axes.linewidth": 1.0,
            "lines.linewidth": 2.4,
        })
    mpl.rcParams.update(settings)


def load_systems(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {
        "system_id", "system_name", "label", "reference", "coverage",
        "I_invocab_neural_bits", "I_invocab_system_bits", "OVMI_neural_bits",
        "OVMI_system_bits", "reference_entropy_bits", "H_pS_bits", "trajectory",
        "operating_point", "plot_eligible", "speech_condition", "invasiveness",
        "P_system_is_lower_bound", "V",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {sorted(missing)}")
    for column in ("trajectory", "operating_point", "plot_eligible", "P_system_is_lower_bound"):
        frame[column] = frame[column].fillna(False).astype(bool)
    return frame


def load_frontier(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"reference", "V", "coverage", "H_pS_bits", "selection"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Frontier CSV is missing required columns: {sorted(missing)}")
    return frame.sort_values(["reference", "V"]).reset_index(drop=True)


def selected_references(frame: pd.DataFrame, requested: str, panels: bool) -> list[str]:
    available = frame["reference"].dropna().drop_duplicates().tolist()
    aliases = {"subtlex": "subtlex-uk", "subtlex_uk": "subtlex-uk"}
    requested = aliases.get(requested.casefold(), requested)
    if requested not in available:
        raise ValueError(f"Reference {requested!r} is absent; choose from {available}")
    return available if panels else [requested]


def axis_limits(
    data: pd.DataFrame,
    frontier: pd.DataFrame,
    neural_only: bool,
    entropy_value: float,
    scale: str,
):
    eligible = data.loc[
        data["plot_eligible"] & np.isfinite(data["coverage"]) & (data["coverage"] > 0)
    ]
    columns = ["I_invocab_neural_bits"]
    if not neural_only:
        columns.append("I_invocab_system_bits")
    information = pd.concat([eligible[column] for column in columns], ignore_index=True)
    information = information.loc[np.isfinite(information) & (information > 0)]
    if information.empty:
        raise ValueError("No finite, plot-eligible coordinates for the selected metric")
    if scale == "log":
        positive_frontier = frontier.loc[frontier["coverage"] > 0, "coverage"]
        x_min = min(float(eligible["coverage"].min()), float(positive_frontier.min()))
        x_limits = (max(0.01, x_min * 0.88), 1.04)
        y_limits = (max(0.01, float(information.min()) * 0.43), entropy_value * 1.25)
    else:
        x_limits = (0.0, 1.02)
        y_limits = (0.0, entropy_value * 1.15)
    return x_limits, y_limits


def draw_contours(axis, x_limits, y_limits, scale: str) -> None:
    x_start = max(x_limits[0], 1e-4)
    if scale == "log":
        x = np.geomspace(x_start, x_limits[1], 800)
    else:
        x = np.linspace(x_start, x_limits[1], 800)

    for index, level in enumerate(CONTOUR_LEVELS):
        y = level / x
        visible = (y >= y_limits[0]) & (y <= y_limits[1])
        axis.plot(x[visible], y[visible], color="#A7ADB3", linewidth=1.25, zorder=0)
        if not np.any(visible):
            continue
        visible_indices = np.flatnonzero(visible)
        if scale == "log":
            # Fixed anchors keep contour labels out of the dense system clusters.
            target_x = {
                0.01: 0.13,
                0.1: 0.65,
                1.0: 0.38,
                3.0: 0.35,
                10.0: 0.84,
            }[level]
            label_index = visible_indices[
                np.argmin(np.abs(x[visible_indices] - target_x))
            ]
        elif level in {0.01, 0.1}:
            label_fraction = {0.01: 0.82, 0.1: 0.72}[level]
        else:
            label_fraction = 0.55 if level == 10 else 0.25 + 0.11 * (index % 3)
        if scale != "log":
            label_index = visible_indices[int(label_fraction * (len(visible_indices) - 1))]
        rotation = -31 if scale == "log" and level < 10 else 0
        vertical_alignment = "top" if scale == "log" and level >= 3 else "bottom"
        axis.text(
            x[label_index], y[label_index], f"{level:g} bit" + ("s" if level != 1 else ""),
            color="#626A73", fontsize=9.0, fontweight="semibold", rotation=rotation,
            ha="center", va=vertical_alignment, clip_on=True,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.72, "pad": 0.35},
            zorder=1,
        )


def draw_noiseless_frontier(axis, frontier: pd.DataFrame) -> None:
    frontier = frontier.loc[
        np.isfinite(frontier["coverage"]) & np.isfinite(frontier["H_pS_bits"])
    ].sort_values("coverage")
    axis.plot(
        frontier["coverage"], frontier["H_pS_bits"],
        color="#6B7280", linestyle=(0, (3.0, 2.3)), linewidth=1.15,
        zorder=2,
    )


def point_transform(axis, system_id: str, status: str):
    """Return a data transform with an optional display-only horizontal dodge."""

    horizontal_points = POINT_DODGE_POINTS.get((system_id, status), 0.0)
    if horizontal_points == 0:
        return axis.transData
    return axis.transData + ScaledTranslation(
        horizontal_points / 72.0, 0.0, axis.figure.dpi_scale_trans,
    )


def draw_static_points(
    axis,
    data: pd.DataFrame,
    neural_only: bool,
    scale: str,
    styled: bool = False,
) -> None:
    points = data.loc[data["plot_eligible"] & np.isfinite(data["coverage"])].copy()
    for _, row in points.iterrows():
        color = COLORS.get(row["system_name"], "#333333")
        invasive = row["invasiveness"] == "invasive"
        facecolor = color if invasive else "white"
        variants = []
        if np.isfinite(row["I_invocab_neural_bits"]):
            variants.append(("neural", row["I_invocab_neural_bits"]))
        if not neural_only and np.isfinite(row["I_invocab_system_bits"]):
            variants.append(("system", row["I_invocab_system_bits"]))
        if not variants:
            continue

        for status, information in variants:
            marker_transform = point_transform(axis, row["system_id"], status)
            point_size = 58 if len(variants) == 2 else 76
            marker = "o"
            if styled:
                marker = STYLED_POINT_STYLES[(row["system_id"], status)][0]
                if marker == "*":
                    point_size *= 1.35
            axis.scatter(
                row["coverage"], information, s=point_size, marker=marker,
                facecolor=facecolor, edgecolor=color, linewidth=1.8, zorder=4,
                transform=marker_transform,
            )
            if styled:
                continue
            if len(variants) == 2:
                task_label = "isolated 50-way" if status == "neural" else "continuous +LM"
                label = f"{row['label']}\n{task_label}"
            else:
                label = row["label"]
                if scale == "linear" and row["system_id"] == "meg_masc_2023_v50":
                    label = "MEG-MASC 2023\n(V=50)"
            if len(variants) != 2 and status == "system":
                label += " (+LM)"
            offsets = LINEAR_LABEL_OFFSETS if scale == "linear" else LABEL_OFFSETS
            offset = offsets.get(
                (row["system_id"], status),
                LABEL_OFFSETS.get((row["system_id"], status), (6, 6)),
            )
            label_arrow = None
            if scale == "linear" and row["system_id"] == "dascoli_libribrain100_s0_v50":
                label_arrow = {
                    "arrowstyle": "-", "color": color, "lw": 1.15,
                    "alpha": 0.75, "shrinkA": 4, "shrinkB": 5,
                }
            alignments = LINEAR_LABEL_ALIGNMENTS if scale == "linear" else LABEL_ALIGNMENTS
            axis.annotate(
                label, (row["coverage"], information), xytext=offset,
                xycoords=marker_transform, textcoords="offset points",
                fontsize=8.5, fontweight="semibold", color=color,
                ha=alignments.get(
                    (row["system_id"], status),
                    LABEL_ALIGNMENTS.get((row["system_id"], status), "left"),
                ),
                va="center", zorder=6, arrowprops=label_arrow,
            )


def styled_system_legend_handles(
    data: pd.DataFrame, neural_only: bool,
) -> list[Line2D]:
    handles = []
    rows = {
        str(row["system_id"]): row
        for _, row in data.loc[data["plot_eligible"]].iterrows()
    }
    for (system_id, status), (marker, label) in STYLED_POINT_STYLES.items():
        if neural_only and status == "system":
            continue
        row = rows.get(system_id)
        if row is None:
            continue
        information_column = (
            "I_invocab_neural_bits" if status == "neural"
            else "I_invocab_system_bits"
        )
        if not np.isfinite(row[information_column]):
            continue
        color = COLORS.get(row["system_name"], "#333333")
        invasive = row["invasiveness"] == "invasive"
        handles.append(Line2D(
            [], [], marker=marker, markersize=6.8 if marker != "*" else 8.0,
            linestyle="none", markerfacecolor=color if invasive else "white",
            markeredgecolor=color, markeredgewidth=1.35, label=label,
        ))
    return handles


def add_legend(
    axis, data: pd.DataFrame, neural_only: bool, styled: bool = False,
) -> None:
    encoding_handles = [
        Line2D([], [], marker="o", markersize=7.5, linestyle="none", markerfacecolor="white", markeredgecolor="#374151", markeredgewidth=1.4, label="Perceived / non-invasive"),
        Line2D([], [], marker="o", markersize=7.5, linestyle="none", markerfacecolor="#6B7280", markeredgecolor="#374151", markeredgewidth=1.4, label="Attempted / invasive"),
        Line2D([], [], color="#A7ADB3", linewidth=1.25, label=r"Iso-OVMI (slope $-1$)"),
        Line2D([], [], color="#6B7280", linestyle=(0, (3.0, 2.3)), linewidth=1.15, label="Noiseless frequency-\nselected decoder"),
    ]
    if styled:
        system_legend = axis.legend(
            handles=styled_system_legend_handles(data, neural_only),
            loc="upper left", bbox_to_anchor=(1.025, 1.0),
            borderaxespad=0.0, frameon=False, fontsize=7.7,
            handletextpad=0.55, labelspacing=0.15,
        )
        axis.add_artist(system_legend)
        axis.legend(
            handles=encoding_handles, loc="lower left",
            bbox_to_anchor=(1.025, 0.0), borderaxespad=0.0,
            frameon=False, handletextpad=0.6, labelspacing=0.30,
            fontsize=7.8,
        )
    else:
        axis.legend(
            handles=encoding_handles, loc="center left",
            bbox_to_anchor=(1.025, 0.5), borderaxespad=0.0,
            frameon=False, handletextpad=0.8, labelspacing=0.65,
        )


def draw_panel(
    axis, data: pd.DataFrame, frontier: pd.DataFrame, reference: str, scale: str,
    neural_only: bool, show_legend: bool, styled: bool = False,
) -> None:
    reference_data = data.loc[data["reference"] == reference]
    reference_frontier = frontier.loc[frontier["reference"] == reference]
    if reference_frontier.empty:
        raise ValueError(f"No noiseless frontier data for {reference}")
    entropy_values = reference_data["reference_entropy_bits"].dropna().unique()
    if len(entropy_values) != 1:
        raise ValueError(f"Expected one entropy for {reference}, got {entropy_values}")
    entropy_value = float(entropy_values[0])
    x_limits, y_limits = axis_limits(
        reference_data, reference_frontier, neural_only, entropy_value, scale,
    )

    draw_contours(axis, x_limits, y_limits, scale)
    draw_noiseless_frontier(axis, reference_frontier)
    draw_static_points(axis, reference_data, neural_only, scale, styled=styled)
    if scale == "log":
        axis.set_xscale("log")
        axis.set_yscale("log")
    axis.set_xlim(x_limits)
    axis.set_ylim(y_limits)
    axis.set_xlabel(
        r"Lexical coverage $C(S)$", labelpad=3 if styled else -2,
    )
    axis.set_ylabel(r"$I(X;Y\mid X\in S)$ (bits)")
    axis.grid(True, which="major", color="#E2E5E9", linewidth=0.7, zorder=-2)
    axis.tick_params(which="major", direction="out", length=4.5, width=1.1, pad=5)
    axis.tick_params(axis="x", which="major", pad=2)
    axis.tick_params(which="minor", direction="out", length=2.5, width=0.8)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    if show_legend:
        add_legend(axis, reference_data, neural_only, styled=styled)


def validate_and_report(
    frame: pd.DataFrame,
    frontier: pd.DataFrame,
    neural_only: bool,
    *,
    ceiling_tolerance: float = 1e-10,
) -> None:
    metric_pairs = [("I_invocab_neural_bits", "OVMI_neural_bits")]
    if not neural_only:
        metric_pairs.append(("I_invocab_system_bits", "OVMI_system_bits"))
    errors = []
    for information_column, ovmi_column in metric_pairs:
        finite = frame.loc[
            frame["plot_eligible"]
            & np.isfinite(frame[information_column])
            & np.isfinite(frame[ovmi_column])
        ]
        errors.extend(np.abs(finite["coverage"] * finite[information_column] - finite[ovmi_column]))
    max_error = float(max(errors))
    if max_error > 1e-10:
        raise AssertionError(f"C * I_invocab != OVMI; maximum error {max_error}")

    plotted_groups = set(
        frame.loc[
            frame["plot_eligible"], ["speech_condition", "invasiveness"]
        ].itertuples(index=False, name=None)
    )
    expected_groups = {("perceived", "non-invasive"), ("attempted", "invasive")}
    if plotted_groups != expected_groups:
        raise AssertionError(
            "Combined glyph legend requires the plotted task/access pairs to be "
            f"exactly {sorted(expected_groups)}; got {sorted(plotted_groups)}"
        )

    ceiling_violations = []
    checked_points = 0
    for _, row in frame.loc[frame["plot_eligible"]].iterrows():
        ceiling = row["H_pS_bits"]
        for information_column, _ in metric_pairs:
            information = row[information_column]
            if not np.isfinite(information):
                continue
            checked_points += 1
            if not np.isfinite(ceiling) or information > ceiling + ceiling_tolerance:
                ceiling_violations.append(
                    (row["system_id"], information_column, information, ceiling)
                )
    if ceiling_violations:
        for system_id, metric, information, ceiling in ceiling_violations:
            print(
                "Sanity violation: "
                f"{system_id} {metric}={information:.12g} > H(p_S)={ceiling:.12g}"
            )
        raise AssertionError(
            f"{len(ceiling_violations)} in-vocabulary information value(s) exceed "
            "their own vocabulary entropy ceiling"
        )

    for reference, group in frontier.groupby("reference", sort=False):
        endpoint = group.sort_values("V").iloc[-1]
        reference_entropy = frame.loc[
            frame["reference"] == reference, "reference_entropy_bits"
        ].dropna().unique()
        if len(reference_entropy) != 1:
            raise AssertionError(f"Cannot validate frontier endpoint for {reference}")
        if not math.isclose(endpoint["coverage"], 1.0, abs_tol=1e-12):
            raise AssertionError(f"Noiseless frontier for {reference} does not terminate at C=1")
        if not math.isclose(
            endpoint["H_pS_bits"], float(reference_entropy[0]), abs_tol=1e-10,
        ):
            raise AssertionError(
                f"Noiseless frontier for {reference} does not terminate at H(p)"
            )

    uniform_vocabulary = [f"word-{index}" for index in range(50)]
    probability = 0.6
    result = ovmi(
        {word: 1.0 for word in uniform_vocabulary}, uniform_vocabulary,
        accuracy=probability, return_details=True,
    )
    wolpaw = (
        math.log2(len(uniform_vocabulary))
        + probability * math.log2(probability)
        + (1.0 - probability) * math.log2((1.0 - probability) / (len(uniform_vocabulary) - 1))
    )
    if not math.isclose(result.in_vocab_information, wolpaw, rel_tol=0.0, abs_tol=1e-12):
        raise AssertionError("Uniform p_S does not recover Wolpaw's formula")

    print(f"Sanity: max |OVMI - C*I_invocab| = {max_error:.3g}")
    print(f"Sanity: uniform-p_S Wolpaw error = {abs(result.in_vocab_information - wolpaw):.3g}")
    print(
        f"Sanity: {checked_points} plotted point(s) satisfy "
        f"I_invocab <= own-vocabulary H(p_S) + {ceiling_tolerance:g}"
    )
    print("Sanity: task/access pairs form exactly two combined glyph groups")
    chance = frame.loc[frame["at_or_below_chance"] == True, "system_id"].tolist()  # noqa: E712
    print(f"Sanity: at/below chance = {chance or 'none'}")
    for system_id in ("willett_2023_v125k", "card_2024_v125k"):
        row = frame.loc[frame["system_id"] == system_id].iloc[0]
        ratio = row["I_invocab_system_bits"] / row["H_pS_bits"]
        print(f"Sanity: {row['label']} I_invocab/H(p_S) = {ratio:.3f}")

    excluded = frame.loc[~frame["plot_eligible"] & frame["exclusion_reason"].notna(), ["system_id", "exclusion_reason"]]
    for _, row in excluded.iterrows():
        print(f"Excluded: {row['system_id']}: {row['exclusion_reason']}")


def output_stem(
    references: list[str], panels: bool, neural_only: bool, styled: bool = False,
) -> str:
    if panels:
        stem = "contour_panels"
    else:
        stem = f"contour_{REFERENCE_STEMS.get(references[0], references[0].replace('-', '_'))}"
    if neural_only:
        stem += "_neural"
    if styled:
        stem += "_styled"
    return stem


def main() -> None:
    args = parse_args()
    configure_matplotlib(args.styled)
    frame = load_systems(args.input)
    frontier = load_frontier(args.frontier_input)
    references = selected_references(frame, args.reference, args.panels)
    validate_and_report(frame, frontier, args.neural_only)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for scale in ("log", "linear"):
        reserve_legend_column = args.styled and len(references) == 1
        figure, axes = plt.subplots(
            1, len(references), figsize=(8.0 * len(references), 3.0),
            squeeze=False, constrained_layout=not reserve_legend_column,
        )
        if reserve_legend_column:
            figure.subplots_adjust(
                left=0.080, right=0.700, bottom=0.205, top=0.975,
            )
        for index, (axis, reference) in enumerate(zip(axes[0], references)):
            draw_panel(
                axis, frame, frontier, reference, scale, args.neural_only,
                show_legend=index == len(references) - 1, styled=args.styled,
            )
        stem = output_stem(
            references, args.panels, args.neural_only, styled=args.styled,
        )
        if scale == "linear":
            stem += "_linear"
        for extension in ("pdf", "png"):
            output = args.output_dir / f"{stem}.{extension}"
            figure.savefig(output, dpi=400 if extension == "png" else None)
            print(f"Wrote {output}")
        plt.close(figure)


if __name__ == "__main__":
    main()
