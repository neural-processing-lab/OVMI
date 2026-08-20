#!/usr/bin/env python3
"""Plot how speech-BCI rankings depend on the communication reference.

The four displayed references are exactly the columns retained in the main
paper table: SUBTLEX-UK, Switchboard, UCV, and Sherlock.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import FixedLocator, FuncFormatter  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import make_main_table as table  # noqa: E402
from make_contour_figure import COLORS  # noqa: E402


DEFAULT_OUTPUT_BASE = PROJECT_ROOT / "figures/reference_dependence"
DEFAULT_CAPTION_OUTPUT = PROJECT_ROOT / "figures/reference_dependence_caption.md"
REVERSAL_TOLERANCE = 1e-10


@dataclass(frozen=True)
class AxisSpec:
    key: str
    position: float
    short_label: str
    display_name: str


AXES = (
    AxisSpec("subtlex", 0.0, "Broad spoken\nSUBTLEX-UK", "broad spoken / SUBTLEX-UK"),
    AxisSpec("conversation", 1.0, "Conversation\nSwitchboard", "conversational / Switchboard"),
    AxisSpec("ucv", 2.0, "AAC / clinical\nUCV", "AAC/clinical / UCV"),
    AxisSpec("narrative", 3.0, "Narrative prose\nSherlock", "narrative prose / Sherlock"),
)


@dataclass(frozen=True)
class Profile:
    point_index: int
    point: table.SystemPoint
    percentages: np.ndarray
    color: str
    label: str


@dataclass(frozen=True)
class Reversal:
    left_label: str
    right_label: str
    left_higher: tuple[str, ...]
    right_higher: tuple[str, ...]


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
    parser.add_argument("--main-table", type=Path, default=table.DEFAULT_OUTPUT)
    parser.add_argument(
        "--appendix-table", type=Path, default=table.DEFAULT_APPENDIX_OUTPUT
    )
    parser.add_argument("--output-base", type=Path, default=DEFAULT_OUTPUT_BASE)
    parser.add_argument("--caption-output", type=Path, default=DEFAULT_CAPTION_OUTPUT)
    return parser.parse_args()


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
    table_scores = table.score_all(points, vocabularies, references, entropies)
    return systems, references, entropies, vocabularies, points, table_scores


def assert_main_table_agreement(
    args: argparse.Namespace,
    points: list[table.SystemPoint],
    scores: dict[tuple[int, str], table.CellScore],
    entropies: dict[str, float],
) -> None:
    """Regenerate both table fragments and require exact artifact agreement."""

    systems = pd.read_csv(args.systems)
    systems = systems.loc[systems["plot_eligible"].astype(bool)].copy()
    references, _ = table.load_references(args.references_dir)
    vocabularies = table.reconstruct_vocabularies(
        systems, references, args.predictions_dir, args.cmudict,
        args.armeni_text, args.meg_masc_vocabulary,
    )
    uncertainties = table.score_uncertainties(
        points, vocabularies, references, entropies
    )
    with tempfile.TemporaryDirectory() as temporary_directory:
        temporary_root = Path(temporary_directory)
        generated_main = temporary_root / "main.tex"
        generated_appendix = temporary_root / "appendix.tex"
        with contextlib.redirect_stdout(io.StringIO()):
            table.render_table(
                points, scores, uncertainties, entropies, generated_main
            )
            table.render_individual_target_appendix(
                points, scores, uncertainties, entropies["individual"],
                generated_appendix,
            )
        expected_pairs = (
            (generated_main, args.main_table),
            (generated_appendix, args.appendix_table),
        )
        for generated, existing in expected_pairs:
            if not existing.exists():
                raise FileNotFoundError(
                    f"Required table artifact is missing: {existing}; run "
                    "scripts/make_main_table.py first"
                )
            if generated.read_text(encoding="utf-8") != existing.read_text(encoding="utf-8"):
                raise AssertionError(
                    f"Figure values disagree with {existing}; regenerate the main table"
                )
    print("CHECK main-table agreement: exact regeneration passed (numerical tolerance <1e-10)")


def point_label(point: table.SystemPoint) -> str:
    vocabulary = "125k" if point.vocabulary_size == 125_000 else str(point.vocabulary_size)
    if point.system_id == "meg_masc_2023_v50":
        return f"MEG-MASC V={vocabulary}"
    if point.system_id.startswith("megxl_"):
        return f"MEG-XL V={vocabulary}"
    if point.system_id.startswith("willett_"):
        suffix = " +LM" if point.probability_source == "w" else " isolated"
        return f"Willett V={vocabulary}{suffix}"
    if point.system_id == "moses_2021_v50":
        return "Moses +LM" if point.probability_source == "w" else "Moses isolated"
    if point.system_id == "card_2024_v125k":
        return "Card +LM"
    return point.display_name


def system_color(point: table.SystemPoint, systems: pd.DataFrame) -> str:
    system_name = str(
        systems.loc[systems["system_id"] == point.system_id, "system_name"].iloc[0]
    )
    return COLORS.get(system_name, "#374151")


def build_profiles(
    systems: pd.DataFrame,
    references: dict[str, dict[str, float]],
    entropies: dict[str, float],
    vocabularies: dict[str, list[str]],
    points: list[table.SystemPoint],
) -> list[Profile]:
    profiles = []
    for point_index, point in enumerate(points):
        values = []
        for axis in AXES:
            cell = table.score_cell(
                references[axis.key],
                entropies[axis.key],
                vocabularies[point.system_id],
                point.probability,
            )
            values.append(cell.percentage)
        profiles.append(Profile(
            point_index=point_index,
            point=point,
            percentages=np.asarray(values, dtype=np.float64),
            color=system_color(point, systems),
            label=point_label(point),
        ))
    return profiles


def reverses(first: np.ndarray, second: np.ndarray) -> bool:
    difference = first - second
    return bool(
        np.any(difference > REVERSAL_TOLERANCE)
        and np.any(difference < -REVERSAL_TOLERANCE)
    )


def reversal_signature(
    profile_index: int, profiles: list[Profile], excluded_system_id: str
) -> frozenset[str]:
    profile = profiles[profile_index]
    return frozenset(
        other.label
        for other_index, other in enumerate(profiles)
        if other_index != profile_index
        and other.point.system_id != excluded_system_id
        and reverses(profile.percentages, other.percentages)
    )


def collapse_parallel_lm_variants(
    profiles: list[Profile],
) -> tuple[list[Profile], list[str]]:
    """Collapse ±LM pairs only when they have the same reversal signature."""

    groups: dict[str, list[int]] = {}
    for index, profile in enumerate(profiles):
        groups.setdefault(profile.point.system_id, []).append(index)
    remove: set[int] = set()
    collapsed: list[str] = []
    for system_id, indices in groups.items():
        if len(indices) != 2:
            continue
        first_index, second_index = indices
        first, second = profiles[first_index], profiles[second_index]
        same_signature = (
            reversal_signature(first_index, profiles, system_id)
            == reversal_signature(second_index, profiles, system_id)
        )
        if reverses(first.percentages, second.percentages) or not same_signature:
            continue
        preferred = next(
            (
                index for index in indices
                if profiles[index].point.probability_source == "w"
            ),
            indices[0],
        )
        omitted = second_index if preferred == first_index else first_index
        remove.add(omitted)
        collapsed.append(system_id)
    return (
        [profile for index, profile in enumerate(profiles) if index not in remove],
        collapsed,
    )


def find_reversals(profiles: list[Profile]) -> list[Reversal]:
    reversals = []
    for left_index, left in enumerate(profiles):
        for right in profiles[left_index + 1:]:
            difference = left.percentages - right.percentages
            left_higher = tuple(
                AXES[index].display_name
                for index in np.flatnonzero(difference > REVERSAL_TOLERANCE)
            )
            right_higher = tuple(
                AXES[index].display_name
                for index in np.flatnonzero(difference < -REVERSAL_TOLERANCE)
            )
            if left_higher and right_higher:
                reversals.append(Reversal(
                    left.label, right.label, left_higher, right_higher
                ))
    return reversals


def crossing_segments(profiles: list[Profile]) -> np.ndarray:
    highlighted = np.zeros((len(profiles), len(AXES) - 1), dtype=bool)
    for first_index, first in enumerate(profiles):
        for second_index in range(first_index + 1, len(profiles)):
            second = profiles[second_index]
            differences = first.percentages - second.percentages
            for segment in range(len(AXES) - 1):
                if differences[segment] * differences[segment + 1] < 0:
                    highlighted[first_index, segment] = True
                    highlighted[second_index, segment] = True
    return highlighted


def spearman_correlations(profiles: list[Profile]) -> pd.DataFrame:
    values = pd.DataFrame(
        np.vstack([profile.percentages for profile in profiles]),
        columns=[axis.key for axis in AXES],
    )
    return values.rank(method="average").corr(method="pearson")


def configure_style() -> None:
    matplotlib.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Source Sans 3", "Arial", "Helvetica", "DejaVu Sans"],
        "mathtext.fontset": "dejavusans",
        "font.size": 10.5,
        "axes.labelsize": 12.0,
        "axes.labelweight": "semibold",
        "xtick.labelsize": 9.2,
        "ytick.labelsize": 9.5,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.linewidth": 1.0,
    })


def line_style(profile: Profile):
    """Dash patterns supplement colour without changing visual weight."""
    key = (profile.point.system_id, profile.point.probability_source)
    return {
        ("card_2024_v125k", "w"): "-",
        ("willett_2023_v125k", "w"): (0, (6, 2.2)),
        ("willett_2023_v50", "w"): (0, (2.2, 1.6)),
        ("moses_2021_v50", "w"): (0, (5, 1.8, 1.2, 1.8)),
        ("tang_2023_v6867", "w"): (0, (4, 1.4, 1, 1.4, 1, 1.4)),
        ("dascoli_libribrain100_s0_v50", "b"): (0, (1.2, 1.5)),
        ("armeni_2022_v50", "b"): (0, (8, 2.2)),
        ("meg_masc_2023_v50", "a"): (0, (3, 1.5, 1, 1.5)),
    }[key]


def marker_style(profile: Profile) -> str:
    key = (profile.point.system_id, profile.point.probability_source)
    return {
        ("card_2024_v125k", "w"): "o",
        ("willett_2023_v125k", "w"): "s",
        ("willett_2023_v50", "w"): "^",
        ("moses_2021_v50", "w"): "D",
        ("tang_2023_v6867", "w"): "p",
        ("dascoli_libribrain100_s0_v50", "b"): "v",
        ("armeni_2022_v50", "b"): "P",
        ("meg_masc_2023_v50", "a"): "X",
    }[key]


def marker_facecolor(profile: Profile) -> str:
    """Use fill as a redundant invasive/non-invasive cue."""
    return profile.color if profile.point.group == "attempted" else "white"


def draw_parallel_coordinates(
    profiles: list[Profile], entropies: dict[str, float], scale: str
):
    configure_style()
    figure, axis = plt.subplots(figsize=(8.0, 3.0))
    x = np.asarray([item.position for item in AXES], dtype=np.float64)
    for axis_spec in AXES:
        axis.axvline(axis_spec.position, color="#C9CED4", linewidth=1.05, zorder=-2)

    legend_handles = []
    for profile_index, profile in enumerate(profiles):
        style = line_style(profile)
        marker = marker_style(profile)
        facecolor = marker_facecolor(profile)
        axis.plot(
            x, profile.percentages,
            color=profile.color, linestyle=style,
            linewidth=2.0, alpha=0.82, zorder=3,
            solid_capstyle="round",
        )
        axis.scatter(
            x, profile.percentages,
            s=27, marker=marker, facecolor=facecolor, edgecolor=profile.color,
            linewidth=1.35, alpha=1.0, zorder=5,
        )
        legend_handles.append(Line2D(
            [], [], color=profile.color, linestyle=style, linewidth=2.0,
            marker=marker, markersize=5.8, markerfacecolor=facecolor,
            markeredgecolor=profile.color, markeredgewidth=1.2,
            label=profile.label,
        ))

    axis.legend(
        handles=legend_handles,
        loc="center left", bbox_to_anchor=(1.015, 0.5),
        frameon=False, borderaxespad=0.0,
        fontsize=8.2, handlelength=2.8, handletextpad=0.7,
        labelspacing=0.65,
    )

    tick_labels = [
        f"{axis_spec.short_label}\n$H(p)={entropies[axis_spec.key]:.2f}$ bits"
        for axis_spec in AXES
    ]
    axis.set_xticks(x, tick_labels)
    axis.tick_params(axis="x", length=0, pad=8)
    axis.set_ylabel(r"OVMI / $H(p)$ (\%)")
    axis.set_xlim(-0.16, AXES[-1].position + 0.16)

    if scale == "log":
        axis.set_yscale("log")
        axis.set_ylim(0.34, 108.0)
        ticks = [0.5, 1, 2, 5, 10, 20, 50, 100]
        axis.yaxis.set_major_locator(FixedLocator(ticks))
        axis.yaxis.set_major_formatter(FuncFormatter(lambda value, _position: f"{value:g}"))
        axis.yaxis.set_minor_formatter(FuncFormatter(lambda _value, _position: ""))
    else:
        axis.set_ylim(0.0, 102.0)
        axis.yaxis.set_major_locator(FixedLocator([0, 20, 40, 60, 80, 100]))

    axis.grid(axis="y", color="#D8DCE1", linewidth=0.7, alpha=0.62, zorder=-3)
    for spine in ("top", "right", "bottom"):
        axis.spines[spine].set_visible(False)
    axis.spines["left"].set_color("#6B7280")
    axis.tick_params(axis="y", colors="#374151", length=3)

    figure.subplots_adjust(left=0.105, right=0.735, top=0.95, bottom=0.29)
    return figure


def save_figure(figure, output_stem: Path) -> None:
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    for extension in ("pdf", "png"):
        output = output_stem.with_suffix(f".{extension}")
        figure.savefig(output, dpi=400 if extension == "png" else None)
        print(f"Saved {output}")


def write_caption(
    path: Path, collapsed: list[str], reversals: list[Reversal]
) -> None:
    collapsed_names = ", ".join(
        "Moses 2021" if name == "moses_2021_v50" else "Willett 2023 (V=50)"
        for name in collapsed
    )
    caption = (
        "Reference dependence of speech-BCI rankings. Each line is OVMI divided "
        "by the full entropy of the stated reference; raw bits are therefore not "
        "compared across axes. Lines use a uniform visual weight throughout. The "
        "external legend combines colour with dash and marker patterns, so system "
        "identity does not depend on colour vision; filled markers indicate invasive "
        "attempted-speech systems and hollow markers indicate non-invasive perceived-"
        "speech systems. Rank reversals are read directly "
        "from crossings. Narrative systems on "
        "Sherlock and the caregiving-vocabulary systems on UCV are design-aligned "
        "and should not be read as general capability. The four axes are exactly the "
        "reference columns retained in the main paper table: SUBTLEX-UK, Switchboard, "
        "UCV, and Sherlock; the BNC robustness estimator and individual Moses target "
        "are not shown. "
        f"The isolated and +LM variants for {collapsed_names} never cross and have "
        "identical reversal partners, so only +LM is shown. "
        f"The plotted set contains {len(reversals)} reversing system pairs. The log "
        "version is the primary panel because it separates the non-invasive traces "
        "without changing any crossings; a linear twin is also provided."
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(caption + "\n", encoding="utf-8")
    print(f"Wrote caption to {path}")


def print_reversals(reversals: list[Reversal]) -> None:
    print(f"Rank reversals ({len(reversals)} unordered system pairs):")
    if not reversals:
        print("  none -- the figure is unlikely to earn its place")
        return
    for reversal in reversals:
        print(
            f"  ({reversal.left_label}, {reversal.right_label}; "
            f"{reversal.left_label} > {reversal.right_label} at "
            f"[{'; '.join(reversal.left_higher)}]; "
            f"{reversal.right_label} > {reversal.left_label} at "
            f"[{'; '.join(reversal.right_higher)}])"
        )


def main() -> None:
    args = parse_args()
    systems, references, entropies, vocabularies, points, table_scores = load_pipeline(args)
    table.validate_scores(points, table_scores)
    assert_main_table_agreement(args, points, table_scores, entropies)

    profiles = build_profiles(
        systems, references, entropies, vocabularies, points
    )
    profiles, collapsed = collapse_parallel_lm_variants(profiles)
    print(
        "Collapsed parallel ±LM variants (kept +LM): "
        + (", ".join(collapsed) if collapsed else "none")
    )
    reversals = find_reversals(profiles)
    print_reversals(reversals)

    correlations = spearman_correlations(profiles)
    print("Spearman rank correlations for plotted lines:")
    print(correlations.to_string(float_format=lambda value: f"{value:.3f}"))

    figures = {}
    for scale in ("linear", "log"):
        figure = draw_parallel_coordinates(profiles, entropies, scale)
        figures[scale] = figure
        save_figure(figure, args.output_base.with_name(f"{args.output_base.name}_{scale}"))
    # The log panel is primary: it makes the roughly two-decade dynamic range
    # legible while preserving the topology of every crossing.
    save_figure(figures["log"], args.output_base)
    for figure in figures.values():
        plt.close(figure)

    write_caption(
        args.caption_output,
        collapsed,
        reversals,
    )


if __name__ == "__main__":
    main()
