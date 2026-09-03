import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import make_progress_over_time_figure as progress
import make_progress_over_time_table as progress_table


class ProgressOverTimeFigureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.points = progress.load_progress_points()
        cls.by_id = {point.point_id: point for point in cls.points}

    def test_every_reported_operating_point_is_present(self):
        self.assertEqual(len(self.points), 10)
        self.assertEqual(
            [point.year for point in self.points],
            [2021, 2021, 2023, 2023, 2023, 2023, 2023, 2023, 2024, 2025],
        )
        self.assertEqual(
            {point.point_id for point in self.points},
            {
                "moses_2021_v50:neural", "moses_2021_v50:system",
                "willett_2023_v50:neural", "willett_2023_v50:system",
                "willett_2023_v125k:system", "card_2024_v125k:system",
                "tang_2023_v6867:system",
                "dascoli_libribrain100_s0_v50:neural",
                "armeni_2022_v50:neural", "meg_masc_2023_v50:neural",
            },
        )

    def test_noninvasive_points_use_method_publication_year(self):
        self.assertEqual(
            self.by_id["dascoli_libribrain100_s0_v50:neural"].year, 2025
        )
        self.assertEqual(self.by_id["armeni_2022_v50:neural"].year, 2023)
        self.assertEqual(self.by_id["meg_masc_2023_v50:neural"].year, 2023)
        self.assertFalse(
            self.by_id["dascoli_libribrain100_s0_v50:neural"].invasive
        )
        self.assertFalse(self.by_id["armeni_2022_v50:neural"].invasive)
        self.assertFalse(self.by_id["meg_masc_2023_v50:neural"].invasive)

    def test_normalised_scores_are_stable(self):
        expected = {
            "moses_2021_v50:neural": 2.432,
            "moses_2021_v50:system": 4.699,
            "willett_2023_v50:neural": 6.713,
            "willett_2023_v50:system": 6.356,
            "willett_2023_v125k:system": 72.023,
            "card_2024_v125k:system": 93.695,
            "dascoli_libribrain100_s0_v50:neural": 2.430,
            "armeni_2022_v50:neural": 1.797,
            "meg_masc_2023_v50:neural": 0.340,
            "tang_2023_v6867:system": 3.650,
        }
        for system_id, percentage in expected.items():
            self.assertAlmostEqual(
                self.by_id[system_id].percentage, percentage, delta=0.001
            )

    def test_invasive_frontier_selects_best_point_in_each_year(self):
        frontier = progress.best_invasive_by_year(self.points)
        self.assertEqual(
            [point.point_id for point in frontier],
            [
                "moses_2021_v50:system",
                "willett_2023_v125k:system",
                "card_2024_v125k:system",
            ],
        )

    def test_figure_is_eight_by_three_inches(self):
        rendered = progress.draw_figure(self.points)
        try:
            self.assertEqual(tuple(rendered.get_size_inches()), (8.0, 3.0))
            self.assertEqual(len(rendered.axes), 2)
            self.assertLessEqual(rendered.axes[1].get_ylim()[1], 8.2)
            self.assertGreaterEqual(rendered.axes[1].get_ylim()[0], -0.4)
        finally:
            progress.plt.close(rendered)

    def test_fill_legend_explains_task_and_access(self):
        rendered = progress.draw_figure(self.points)
        try:
            labels = [
                text.get_text()
                for text in rendered.axes[0].get_legend().get_texts()
            ]
            self.assertEqual(
                labels,
                ["Attempted / invasive", "Perceived / non-invasive"],
            )
        finally:
            progress.plt.close(rendered)

    def test_outputs_and_caption_exist(self):
        for extension in ("pdf", "png"):
            path = progress.DEFAULT_OUTPUT_BASE.with_suffix(f".{extension}")
            self.assertTrue(path.exists(), msg=str(path))
            self.assertGreater(path.stat().st_size, 0, msg=str(path))
        caption = progress.DEFAULT_CAPTION_OUTPUT.read_text(encoding="utf-8")
        self.assertIn("method publication year (2025 and 2026)", caption)
        self.assertIn("The remaining points are not joined", caption)
        self.assertIn("method--dataset pair", caption)
        self.assertIn("inset enlarges the 0--8\\% region", caption)

    def test_simple_latex_table_uses_the_same_ten_points(self):
        rows = progress_table.build_rows()
        latex = progress_table.render_table(rows)
        self.assertEqual(len(rows), 10)
        self.assertIn("Tang +LM & 6,867", latex)
        self.assertIn("d'Ascoli–LibriBrain100 & 50", latex)
        self.assertIn("MEG-XL–MEG-MASC & 50", latex)
        self.assertIn(r"$\geq 74.4\%$", latex)
        self.assertIn("0.033 & 0.3\\%", latex)


if __name__ == "__main__":
    unittest.main()
