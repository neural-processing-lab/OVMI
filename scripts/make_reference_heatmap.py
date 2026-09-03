#!/usr/bin/env python3
"""Plot normalised OVMI for representative systems across four references."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LogNorm  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import make_main_table as table  # noqa: E402


DEFAULT_OUTPUT_BASE = PROJECT_ROOT / "figures/reference_heatmap"
DEFAULT_CAPTION_OUTPUT = PROJECT_ROOT / "figures/reference_heatmap_caption.md"
REFERENCE_KEYS = ("subtlex", "conversation", "ucv", "narrative")
REFERENCE_LABELS = (
    "Broad speech\n(SUBTLEX-UK)",
    "Conversation\n(Switchboard)",
    "AAC\n(UCV)",
    "Narrative\n(Sherlock)",
)
REFERENCE_CAPTION_LABELS = (
    "Broad speech / SUBTLEX-UK",
    "Conversation / Switchboard",
    "AAC / UCV",
    "Narrative / Sherlock",
)
CHECK_TOLERANCE = 1e-10


@dataclass(frozen=True)
class HeatmapRow:
    system_id: str
    variant: str
    label: str


@dataclass(frozen=True)
class HeatmapData:
    values: np.ndarray
    lows: np.ndarray
    highs: np.ndarray
    uncertainty_kinds: tuple[str, ...]
    entropies: dict[str, float]
    reference_sizes: dict[str, int]


ROW_SPECS = (
    HeatmapRow("card_2024_v125k", "system", "Card +LM"),
    HeatmapRow("willett_2023_v125k", "system", "Willett 125k +LM"),
    HeatmapRow("willett_2023_v50", "system", "Willett 50 +LM"),
    HeatmapRow("moses_2021_v50", "system", "Moses +LM"),
    HeatmapRow("moses_2021_v50", "neural", "Moses isolated"),
    HeatmapRow("tang_2023_v6867", "system", "Tang"),
    HeatmapRow(
        "dascoli_libribrain100_s0_v50", "neural",
        "LibriBrain100",
    ),
)


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
    parser.add_argument(
        "--armeni-text", type=Path, default=table.DEFAULT_ARMENI_TEXT,
    )
    parser.add_argument(
        "--meg-masc-vocabulary", type=Path,
        default=table.DEFAULT_MEG_MASC_VOCABULARY,
    )
    parser.add_argument("--output-base", type=Path, default=DEFAULT_OUTPUT_BASE)
    parser.add_argument(
        "--caption-output", type=Path, default=DEFAULT_CAPTION_OUTPUT,
    )
    return parser.parse_args()


def point_variant(point: table.SystemPoint) -> str:
    return "system" if point.probability_source == "w" else "neural"


def load_matrix(
    systems_path: Path = table.DEFAULT_SYSTEMS,
    references_dir: Path = table.DEFAULT_REFERENCES,
    predictions_dir: Path = table.DEFAULT_PREDICTIONS,
    cmudict_path: Path = table.DEFAULT_CMUDICT,
    armeni_text_path: Path = table.DEFAULT_ARMENI_TEXT,
    meg_masc_vocabulary_path: Path = table.DEFAULT_MEG_MASC_VOCABULARY,
) -> HeatmapData:
    systems = pd.read_csv(systems_path)
    systems = systems.loc[systems["plot_eligible"].astype(bool)].copy()
    references, entropies = table.load_references(references_dir)
    reference_sizes = {
        key: len(references[key]) for key in REFERENCE_KEYS
    }
    vocabularies = table.reconstruct_vocabularies(
        systems, references, predictions_dir, cmudict_path, armeni_text_path,
        meg_masc_vocabulary_path,
    )
    points = table.expand_system_points(systems)
    scores = table.score_all(points, vocabularies, references, entropies)
    uncertainties = table.score_uncertainties(
        points, vocabularies, references, entropies,
    )

    point_indices = {}
    for index, point in enumerate(points):
        point_indices[(point.system_id, point_variant(point))] = index

    values = np.empty((len(ROW_SPECS), len(REFERENCE_KEYS)), dtype=float)
    lows = np.empty_like(values)
    highs = np.empty_like(values)
    uncertainty_kinds = []
    for row_index, row in enumerate(ROW_SPECS):
        key = (row.system_id, row.variant)
        if key not in point_indices:
            raise ValueError(f"Missing main-table point {key}")
        point_index = point_indices[key]
        point = points[point_index]
        uncertainty_kinds.append(point.uncertainty_kind)
        for column_index, reference_key in enumerate(REFERENCE_KEYS):
            values[row_index, column_index] = scores[
                (point_index, reference_key)
            ].percentage
            uncertainty_key = (point_index, reference_key)
            if uncertainty_key not in uncertainties:
                raise ValueError(f"Missing uncertainty for {key} on {reference_key}")
            interval = uncertainties[uncertainty_key]
            lows[row_index, column_index] = interval.low.percentage
            highs[row_index, column_index] = interval.high.percentage

    if not all(np.isfinite(array).all() for array in (values, lows, highs)):
        raise AssertionError("Heatmap values and uncertainty must be finite")
    if np.any(values <= 0.0):
        raise AssertionError("Log-colour heatmap values must be finite and positive")
    if np.any(values > 100.0 + CHECK_TOLERANCE):
        raise AssertionError("Normalised OVMI cannot exceed 100%")
    if np.any(lows > values) or np.any(values > highs):
        raise AssertionError("Every heatmap point must lie within its uncertainty")
    return HeatmapData(
        values=values,
        lows=lows,
        highs=highs,
        uncertainty_kinds=tuple(uncertainty_kinds),
        entropies=entropies,
        reference_sizes=reference_sizes,
    )


def configure_style() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "mathtext.fontset": "dejavusans",
        "font.size": 10.5,
        "xtick.labelsize": 10.5,
        "ytick.labelsize": 10.5,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def cell_annotation(data: HeatmapData, row: int, column: int) -> tuple[str, str]:
    value = data.values[row, column]
    low = data.lows[row, column]
    high = data.highs[row, column]
    if data.uncertainty_kinds[row] in {"seed_sem", "participant_sem"}:
        sem = max(value - low, high - value)
        return f"{value:.2f}", rf"$\pm${sem:.2f}"
    return f"{value:.1f}", f"[{low:.1f}, {high:.1f}]"


def draw_figure(data: HeatmapData):
    configure_style()
    figure, axis = plt.subplots(figsize=(8.0, 3.25))
    figure.subplots_adjust(left=0.185, right=0.880, bottom=0.105, top=0.765)

    normalisation = LogNorm(vmin=1.0, vmax=100.0)
    image = axis.imshow(
        data.values, cmap="gist_heat", norm=normalisation,
        interpolation="nearest", aspect="auto",
    )

    axis.set_xticks(np.arange(len(REFERENCE_LABELS)), REFERENCE_LABELS)
    axis.set_yticks(
        np.arange(len(ROW_SPECS)), [row.label for row in ROW_SPECS],
    )
    axis.xaxis.tick_top()
    axis.tick_params(axis="x", which="major", length=0, pad=9)
    for label in axis.get_xticklabels():
        label.set_linespacing(1.30)
    for column_index, reference_key in enumerate(REFERENCE_KEYS):
        axis.text(
            column_index, -0.055, f"V={data.reference_sizes[reference_key]:,}",
            transform=axis.get_xaxis_transform(), ha="center", va="top",
            fontsize=9.0, color="#374151",
        )
    axis.tick_params(axis="y", which="major", length=0, pad=8, labelsize=10.0)

    axis.set_xticks(np.arange(-0.5, len(REFERENCE_LABELS), 1), minor=True)
    axis.set_yticks(np.arange(-0.5, len(ROW_SPECS), 1), minor=True)
    axis.grid(which="minor", color="white", linewidth=2.0)
    axis.tick_params(which="minor", bottom=False, left=False)
    for spine in axis.spines.values():
        spine.set_visible(False)

    for row_index in range(data.values.shape[0]):
        for column_index in range(data.values.shape[1]):
            value = data.values[row_index, column_index]
            red, green, blue, _ = image.cmap(normalisation(value))
            luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue
            text_color = "#111827" if luminance > 0.55 else "white"
            point_text, uncertainty_text = cell_annotation(
                data, row_index, column_index,
            )
            axis.text(
                column_index, row_index - 0.10, point_text,
                ha="center", va="center", color=text_color,
                fontsize=9.4, fontweight="semibold",
            )
            axis.text(
                column_index, row_index + 0.18, uncertainty_text,
                ha="center", va="center", color=text_color,
                fontsize=6.8,
            )

    colorbar = figure.colorbar(image, ax=axis, fraction=0.036, pad=0.035)
    colorbar.set_ticks((1, 3, 10, 30, 100))
    colorbar.set_ticklabels(("1", "3", "10", "30", "100"))
    colorbar.ax.tick_params(labelsize=8.5, length=2.5, width=0.7)
    colorbar.set_label(r"OVMI / $H(p)$ (\%)", fontsize=9.5, labelpad=6)
    colorbar.outline.set_linewidth(0.7)
    return figure


def save_figure(figure, output_base: Path) -> None:
    output_base.parent.mkdir(parents=True, exist_ok=True)
    for extension in ("pdf", "png"):
        output = output_base.with_suffix(f".{extension}")
        figure.savefig(output, dpi=400 if extension == "png" else None)
        print(f"Saved {output}")


def write_caption(
    path: Path, entropies: dict[str, float], reference_sizes: dict[str, int],
) -> None:
    entropy_text = ", ".join(
        f"{label} (V={reference_sizes[key]:,}), {entropies[key]:.2f} bits"
        for key, label in zip(REFERENCE_KEYS, REFERENCE_CAPTION_LABELS)
    )
    caption = (
        "Normalised OVMI across communication targets. Cell labels report "
        "OVMI/$H(p)$ as percentages; brackets give propagated 95\% intervals "
        "for invasive systems, while $\pm$ gives SEM across three training seeds "
        "for LibriBrain100 or three participants for Tang; neither is a confidence "
        "interval. Reference distributions "
        "are treated as fixed. Colour uses a logarithmic 1--100\% scale "
        "to retain contrast among the lower-scoring systems. The invasive rows "
        "show the selected LM-assisted operating points, with Moses isolated "
        "retained as the corresponding neural-only result. The non-invasive rows "
        "are Tang's participant-mean fMRI result and the d'Ascoli--LibriBrain100 "
        "method--dataset pair. Reference "
        "entropies are: "
        f"{entropy_text}."
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(caption + "\n", encoding="utf-8")
    print(f"Wrote {path}")


def report(data: HeatmapData) -> None:
    print("Normalised OVMI heatmap with uncertainty (%):")
    print("System\t" + "\t".join(REFERENCE_CAPTION_LABELS))
    for row_index, row in enumerate(ROW_SPECS):
        cells = []
        for column_index in range(len(REFERENCE_KEYS)):
            point, uncertainty = cell_annotation(data, row_index, column_index)
            cells.append(f"{point} {uncertainty}")
        print(row.label + "\t" + "\t".join(cells))


def main() -> None:
    args = parse_args()
    data = load_matrix(
        args.systems, args.references_dir, args.predictions_dir,
        args.cmudict, args.armeni_text, args.meg_masc_vocabulary,
    )
    report(data)
    figure = draw_figure(data)
    save_figure(figure, args.output_base)
    plt.close(figure)
    write_caption(args.caption_output, data.entropies, data.reference_sizes)


if __name__ == "__main__":
    main()
