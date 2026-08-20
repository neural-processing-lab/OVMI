#!/usr/bin/env python3
"""Plot reported speech-decoder OVMI operating points by publication year."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.transforms import ScaledTranslation  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = PROJECT_ROOT / "scripts"
for directory in (SCRIPT_DIR,):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from make_contour_figure import COLORS  # noqa: E402


DEFAULT_SYSTEMS = PROJECT_ROOT / "data/systems.csv"
DEFAULT_OUTPUT_BASE = PROJECT_ROOT / "figures/progress_over_time"
DEFAULT_CAPTION_OUTPUT = PROJECT_ROOT / "figures/progress_over_time_caption.md"
SPLIT_OUTPUT_BASE = PROJECT_ROOT / "figures/progress_over_time_split"
SPLIT_CAPTION_OUTPUT = PROJECT_ROOT / "figures/progress_over_time_split_caption.md"
CHECK_TOLERANCE = 1e-10


@dataclass(frozen=True)
class ProgressPoint:
    point_id: str
    system_id: str
    variant: str
    label: str
    year: int
    ovmi_bits: float
    reference_entropy_bits: float
    percentage: float
    invasive: bool
    marker: str
    color: str
    source: str
    horizontal_offset_points: float = 0.0
    label_offset_points: tuple[float, float] = (0.0, 10.0)


POINT_SPECS = (
    (
        "moses_2021_v50", "neural", "Moses isolated (50)", 2021,
        "^", COLORS["Moses et al."], -9.0, (-5.0, 9.0),
    ),
    (
        "moses_2021_v50", "system", "Moses +LM (50)", 2021,
        "v", COLORS["Moses et al."], 0.0, (5.0, 9.0),
    ),
    (
        "willett_2023_v50", "neural", "Willett isolated (50)", 2023,
        "P", COLORS["Willett et al."], -8.0, (-13.0, 9.0),
    ),
    (
        "willett_2023_v50", "system", "Willett +LM (50)", 2023,
        "X", COLORS["Willett et al."], 8.0, (13.0, 9.0),
    ),
    (
        "willett_2023_v125k", "system", "Willett +LM (125k)", 2023,
        "h", COLORS["Willett et al."], 0.0, (0.0, 9.0),
    ),
    (
        "tang_2023_v6867", "system", "Tang +LM", 2023,
        "p", COLORS["Tang et al."], 18.0, (2.0, 9.0),
    ),
    (
        "card_2024_v125k", "system", "Card +LM (125k)", 2024,
        "*", COLORS["Card et al."], 0.0, (0.0, -15.0),
    ),
    (
        "dascoli_libribrain100_s0_v50", "neural",
        "d'Ascoli–LibriBrain", 2025,
        "s", COLORS["LibriBrain100"], -8.0, (-10.0, 9.0),
    ),
    (
        "armeni_2022_v50", "neural",
        "d'Ascoli–Armeni", 2023,
        "D", COLORS["Armeni et al."], 8.0, (11.0, 22.0),
    ),
    (
        "meg_masc_2023_v50", "neural",
        "MEG-XL–MEG-MASC", 2023,
        "o", COLORS["MEG-MASC"], 0.0, (0.0, 9.0),
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--systems", type=Path, default=DEFAULT_SYSTEMS)
    parser.add_argument("--output-base", type=Path, default=DEFAULT_OUTPUT_BASE)
    parser.add_argument(
        "--caption-output", type=Path, default=DEFAULT_CAPTION_OUTPUT,
    )
    parser.add_argument(
        "--split", action="store_true",
        help="Emit the separate invasive/non-invasive two-panel version.",
    )
    return parser.parse_args()


def load_progress_points(
    systems_path: Path = DEFAULT_SYSTEMS,
) -> list[ProgressPoint]:
    systems = pd.read_csv(systems_path)
    points = []
    for (
        system_id, variant, label, year, marker, color,
        horizontal_offset, label_offset,
    ) in POINT_SPECS:
        matches = systems.loc[systems["system_id"] == system_id]
        if len(matches) != 1:
            raise ValueError(f"Expected exactly one row for {system_id}")
        row = matches.iloc[0]
        ovmi_column = f"OVMI_{variant}_bits"
        score = float(row[ovmi_column])
        reference_entropy = float(row["reference_entropy_bits"])
        if not np.isfinite(score) or not np.isfinite(reference_entropy):
            raise ValueError(f"Missing OVMI or reference entropy for {system_id}")
        points.append(ProgressPoint(
            point_id=f"{system_id}:{variant}",
            system_id=system_id,
            variant=variant,
            label=label,
            year=year,
            ovmi_bits=score,
            reference_entropy_bits=reference_entropy,
            percentage=100.0 * score / reference_entropy,
            invasive=str(row["invasiveness"]) == "invasive",
            marker=marker,
            color=color,
            source=str(row["source"]),
            horizontal_offset_points=horizontal_offset,
            label_offset_points=label_offset,
        ))

    points.sort(key=lambda point: point.year)
    validate_points(points)
    return points


def validate_points(points: list[ProgressPoint]) -> None:
    expected_ids = [f"{spec[0]}:{spec[1]}" for spec in POINT_SPECS]
    if {point.point_id for point in points} != set(expected_ids):
        raise AssertionError("Progress figure does not contain every reported point")
    if len({point.label for point in points}) != len(points):
        raise AssertionError("Progress labels must be unique")
    if any(
        not np.isfinite(point.ovmi_bits)
        or not np.isfinite(point.reference_entropy_bits)
        or not np.isfinite(point.percentage)
        for point in points
    ):
        raise AssertionError("Progress metrics must be finite")
    if any(point.ovmi_bits < 0 for point in points):
        raise AssertionError("OVMI cannot be negative")
    if any(point.percentage > 100.0 + CHECK_TOLERANCE for point in points):
        raise AssertionError("Normalised OVMI cannot exceed 100%")
    entropies = np.asarray([point.reference_entropy_bits for point in points])
    if np.ptp(entropies) > CHECK_TOLERANCE:
        raise AssertionError("All points must use the same SUBTLEX-UK entropy")
    method_years = {point.point_id: point.year for point in points}
    if method_years["dascoli_libribrain100_s0_v50:neural"] != 2025:
        raise AssertionError("d'Ascoli must use the 2025 method year")
    if method_years["armeni_2022_v50:neural"] != 2023:
        raise AssertionError("Armeni must use the 2023 dataset year")
    if method_years["meg_masc_2023_v50:neural"] != 2023:
        raise AssertionError("MEG-MASC must use the 2023 dataset year")


def configure_style() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "mathtext.fontset": "dejavusans",
        "font.size": 9.0,
        "axes.labelsize": 10.5,
        "axes.linewidth": 1.0,
        "xtick.labelsize": 9.0,
        "ytick.labelsize": 9.0,
        "legend.fontsize": 9.0,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def system_handles(
    points: list[ProgressPoint], marker_scale: float = 1.0,
) -> list[Line2D]:
    handles = []
    for point in points:
        handles.append(Line2D(
            [], [], marker=point.marker,
            markersize=marker_scale * (7.3 if point.marker != "*" else 8.5),
            linestyle="none",
            markerfacecolor=point.color if point.invasive else "white",
            markeredgecolor=point.color, markeredgewidth=1.5,
            label=point.label,
        ))
    return handles


def access_handles() -> list[Line2D]:
    return [
        Line2D(
            [], [], marker="o", markersize=7.3, linestyle="none",
            markerfacecolor="#6B7280", markeredgecolor="#374151",
            markeredgewidth=1.4, label="Attempted / invasive",
        ),
        Line2D(
            [], [], marker="o", markersize=7.3, linestyle="none",
            markerfacecolor="white", markeredgecolor="#374151",
            markeredgewidth=1.4, label="Perceived / non-invasive",
        ),
    ]


def best_invasive_by_year(points: list[ProgressPoint]) -> list[ProgressPoint]:
    """Return the highest-normalised invasive operating point in each year."""
    invasive_years = sorted({point.year for point in points if point.invasive})
    return [
        max(
            (point for point in points if point.invasive and point.year == year),
            key=lambda point: point.percentage,
        )
        for year in invasive_years
    ]


def point_transform(axis, figure, point: ProgressPoint):
    return axis.transData + ScaledTranslation(
        point.horizontal_offset_points / 72.0, 0.0,
        figure.dpi_scale_trans,
    )


def draw_low_ovmi_inset(figure, points: list[ProgressPoint]):
    """Repeat the low-scoring points on a compact 0--8% detail axis."""
    inset = figure.add_axes([0.625, 0.135, 0.355, 0.255])
    for point in points:
        if point.percentage > 8.0:
            continue
        marker_transform = point_transform(inset, figure, point)
        inset.scatter(
            point.year, point.percentage,
            s=34 if point.marker != "*" else 42,
            marker=point.marker,
            facecolor=point.color if point.invasive else "white",
            edgecolor=point.color, linewidth=1.25, zorder=3,
            transform=marker_transform,
        )

    inset.set_xlim(2020.6, 2026.4)
    inset.set_ylim(-0.35, 8.15)
    inset.set_xticks((2021, 2023, 2025))
    inset.set_yticks((0, 4, 8))
    inset.set_title("0–8% detail", fontsize=7.4, pad=2.0)
    inset.grid(color="#D5D9DE", linewidth=0.48, alpha=0.72)
    inset.set_axisbelow(True)
    inset.tick_params(
        axis="both", labelsize=6.5, length=2.1, width=0.65,
        color="#4B5563", pad=1.5,
    )
    for spine in inset.spines.values():
        spine.set_color("#6B7280")
        spine.set_linewidth(0.7)
    return inset


def draw_figure(points: list[ProgressPoint]):
    configure_style()
    figure, axis = plt.subplots(figsize=(8.0, 3.0))
    figure.subplots_adjust(left=0.090, right=0.560, bottom=0.205, top=0.965)

    invasive_frontier = best_invasive_by_year(points)
    axis.plot(
        [point.year for point in invasive_frontier],
        [point.percentage for point in invasive_frontier],
        color="#4B5563", linewidth=2.0, solid_capstyle="round",
        solid_joinstyle="round", zorder=2,
    )

    for point in points:
        size = 110 if point.marker != "*" else 145
        marker_transform = point_transform(axis, figure, point)
        axis.scatter(
            point.year, point.percentage, s=size, marker=point.marker,
            facecolor=point.color if point.invasive else "white",
            edgecolor=point.color, linewidth=1.9, zorder=4,
            transform=marker_transform,
        )
        offset = point.label_offset_points
        vertical_alignment = "top" if point.percentage > 90 else "bottom"
        axis.annotate(
            f"{point.percentage:.1f}%", (point.year, point.percentage),
            xycoords=marker_transform, xytext=offset,
            textcoords="offset points", ha="center",
            va=vertical_alignment, color=point.color, fontsize=8.5,
            fontweight="semibold", zorder=5,
        )

    axis.set_xlim(2020.6, 2026.4)
    axis.set_ylim(0.0, 102.0)
    axis.set_xticks(np.arange(2021, 2027))
    axis.set_yticks(np.arange(0, 101, 20))
    axis.set_xlabel("Method publication year")
    axis.set_ylabel(r"OVMI / $H(p)$ (\%)")
    axis.grid(axis="both", color="#D5D9DE", linewidth=0.65, alpha=0.72)
    axis.set_axisbelow(True)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.tick_params(length=3.2, width=0.9, color="#4B5563")

    system_legend = axis.legend(
        handles=system_handles(points), loc="upper left",
        bbox_to_anchor=(1.035, 0.875), borderaxespad=0.0,
        frameon=False, handletextpad=0.45, labelspacing=0.22,
        columnspacing=0.9, ncol=2, fontsize=7.2,
    )
    axis.add_artist(system_legend)
    axis.legend(
        handles=access_handles(), loc="upper left",
        bbox_to_anchor=(1.035, 1.0), borderaxespad=0.0,
        frameon=False, handletextpad=0.45, labelspacing=0.15,
        columnspacing=1.0, ncol=2, fontsize=7.2,
    )
    draw_low_ovmi_inset(figure, points)
    return figure


def _style_axis(axis, *, invasive: bool) -> None:
    axis.set_xlim(2020.6 if invasive else 2022.6, 2024.4 if invasive else 2026.4)
    axis.set_xticks((2021, 2022, 2023, 2024) if invasive else (2023, 2024, 2025, 2026))
    if invasive:
        axis.set_ylim(0.0, 102.0)
        axis.set_yticks(np.arange(0, 101, 20))
        axis.set_title("Invasive", loc="left", fontsize=11, fontweight="semibold")
    else:
        axis.set_ylim(0.0, 8.0)
        axis.set_yticks(np.arange(0, 9, 2))
        axis.set_title("Non-invasive", loc="left", fontsize=11, fontweight="semibold")
    axis.set_xlabel("Study publication year")
    axis.set_ylabel(r"OVMI / $H(p)$ (\%)")
    axis.grid(axis="both", color="#D5D9DE", linewidth=0.65, alpha=0.72)
    axis.set_axisbelow(True)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.tick_params(length=3.2, width=0.9, color="#4B5563")


def draw_split_figure(points: list[ProgressPoint]):
    """Draw invasive and non-invasive progress on independently scaled axes."""
    configure_style()
    figure, axes = plt.subplots(1, 2, figsize=(8.0, 3.25), sharex=False)
    figure.subplots_adjust(left=0.075, right=0.985, bottom=0.44, top=0.87, wspace=0.28)

    for axis, invasive in zip(axes, (True, False)):
        panel_points = [point for point in points if point.invasive == invasive]
        if invasive:
            frontier = best_invasive_by_year(panel_points)
            axis.plot(
                [point.year for point in frontier],
                [point.percentage for point in frontier],
                color="#4B5563", linewidth=2.0, solid_capstyle="round",
                solid_joinstyle="round", zorder=2,
            )
        for point in panel_points:
            size = 110 if point.marker != "*" else 145
            marker_transform = point_transform(axis, figure, point)
            axis.scatter(
                point.year, point.percentage, s=size, marker=point.marker,
                facecolor=point.color if point.invasive else "white",
                edgecolor=point.color, linewidth=1.9, zorder=4,
                transform=marker_transform,
            )
            offset = point.label_offset_points
            if not invasive:
                offset = {
                    "tang_2023_v6867": (-18.0, 8.0),
                    "armeni_2022_v50": (18.0, -4.0),
                    "meg_masc_2023_v50": (-10.0, 8.0),
                }.get(point.system_id, offset)
            vertical_alignment = "top" if point.percentage > 90 else "bottom"
            axis.annotate(
                f"{point.percentage:.1f}%", (point.year, point.percentage),
                xycoords=marker_transform, xytext=offset,
                textcoords="offset points", ha="center",
                va=vertical_alignment, color=point.color, fontsize=8.0,
                fontweight="semibold", zorder=5,
            )
        _style_axis(axis, invasive=invasive)
        axis.legend(
            handles=system_handles(panel_points, marker_scale=1.18), loc="upper center",
            bbox_to_anchor=(0.5, -0.42), borderaxespad=0.0,
            frameon=False, handlelength=1.1, handletextpad=0.9,
            labelspacing=0.8, columnspacing=1.8, ncol=2, fontsize=7.5,
        )

    return figure


def save_figure(figure, output_base: Path) -> None:
    output_base.parent.mkdir(parents=True, exist_ok=True)
    for extension in ("pdf", "png"):
        output = output_base.with_suffix(f".{extension}")
        figure.savefig(output, dpi=400 if extension == "png" else None)
        print(f"Saved {output}")


def write_caption(path: Path, points: list[ProgressPoint]) -> None:
    entropy_value = points[0].reference_entropy_bits
    caption = (
        "Reported speech-decoder progress under a fixed SUBTLEX-UK "
        f"communication target ($H(p)={entropy_value:.2f}$ bits). Each point is "
        "a reported operating point and gives OVMI normalised by the full "
        "reference entropy; isolated and LM-assisted configurations are retained "
        "as separate points. Filled markers denote attempted/invasive systems; "
        "open markers denote perceived/non-invasive systems. Each non-invasive "
        "point is a method--dataset pair: Tang's 2023 participant-mean fMRI "
        "semantic decoder result, local $V=50$ evaluations of d'Ascoli's method "
        "on LibriBrain100 subject 0 and Armeni, and MEG-XL on MEG-MASC. The local "
        "points are positioned by source dataset publication year: Armeni and "
        "MEG-MASC in 2023, and LibriBrain100 in 2025. "
        "Small horizontal offsets only separate configurations sharing the same "
        "publication year. The solid line connects the highest invasive OVMI "
        "operating point reported in each year; it is a best-achieved frontier, "
        "not a within-system trajectory. The inset enlarges the 0--8\% region "
        "without changing scale or values. The remaining points are not joined: the "
        "systems differ in vocabulary, task, access method, and language-model use, "
        "so the plot is a historical comparison rather than a controlled trajectory. "
        "Uncertainty is omitted from the figure for clarity."
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(caption + "\n", encoding="utf-8")
    print(f"Wrote caption to {path}")


def write_split_caption(path: Path, points: list[ProgressPoint]) -> None:
    entropy_value = points[0].reference_entropy_bits
    caption = (
        "Reported speech-decoder progress under a fixed SUBTLEX-UK "
        f"communication target ($H(p)={entropy_value:.2f}$ bits), split into "
        "invasive and non-invasive panels with independently scaled y-axes. "
        "Each point is a reported operating point and gives OVMI normalised by "
        "the full reference entropy; isolated and LM-assisted configurations are "
        "retained as separate points. Filled markers denote attempted/invasive "
        "systems; open markers denote perceived/non-invasive systems. The solid "
        "line in the invasive panel connects the highest reported operating point "
        "in each year. Non-invasive points are shown at their method publication "
        "years. Uncertainty is omitted from the figure for clarity."
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(caption + "\n", encoding="utf-8")
    print(f"Wrote caption to {path}")


def report(points: list[ProgressPoint]) -> None:
    print("Progress points under SUBTLEX-UK:")
    for point in points:
        print(
            f"  {point.year} {point.label}: {point.ovmi_bits:.6f} bits; "
            f"OVMI/H(p)={point.percentage:.3f}%"
        )
    print("CHECK all points share one fixed SUBTLEX-UK reference entropy")
    print("CHECK non-invasive points use source dataset years")


def main() -> None:
    args = parse_args()
    points = load_progress_points(
        args.systems,
    )
    report(points)
    if args.split:
        output_base = args.output_base if args.output_base != DEFAULT_OUTPUT_BASE else SPLIT_OUTPUT_BASE
        caption_output = (
            args.caption_output
            if args.caption_output != DEFAULT_CAPTION_OUTPUT
            else SPLIT_CAPTION_OUTPUT
        )
        figure = draw_split_figure(points)
    else:
        output_base = args.output_base
        caption_output = args.caption_output
        figure = draw_figure(points)
    save_figure(figure, output_base)
    plt.close(figure)
    if args.split:
        write_split_caption(caption_output, points)
    else:
        write_caption(caption_output, points)


if __name__ == "__main__":
    main()
