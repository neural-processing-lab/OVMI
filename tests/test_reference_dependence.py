import sys
import unittest
from argparse import Namespace
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import make_reference_dependence_figure as figure


class ReferenceDependenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.args = Namespace(
            systems=figure.table.DEFAULT_SYSTEMS,
            references_dir=figure.table.DEFAULT_REFERENCES,
            predictions_dir=figure.table.DEFAULT_PREDICTIONS,
            cmudict=figure.table.DEFAULT_CMUDICT,
            armeni_text=figure.table.DEFAULT_ARMENI_TEXT,
            meg_masc_vocabulary=figure.table.DEFAULT_MEG_MASC_VOCABULARY,
            main_table=figure.table.DEFAULT_OUTPUT,
            appendix_table=figure.table.DEFAULT_APPENDIX_OUTPUT,
            output_base=figure.DEFAULT_OUTPUT_BASE,
            caption_output=figure.DEFAULT_CAPTION_OUTPUT,
        )
        (
            cls.systems,
            cls.references,
            cls.entropies,
            cls.vocabularies,
            cls.points,
            cls.table_scores,
        ) = figure.load_pipeline(cls.args)
        cls.full_profiles = figure.build_profiles(
            cls.systems,
            cls.references,
            cls.entropies,
            cls.vocabularies,
            cls.points,
        )
        cls.profiles, cls.collapsed = figure.collapse_parallel_lm_variants(
            cls.full_profiles
        )

    def test_axes_match_main_table_columns_exactly(self):
        self.assertEqual(
            [axis.key for axis in figure.AXES],
            ["subtlex", "conversation", "ucv", "narrative"],
        )
        self.assertNotIn("bnc", [axis.key for axis in figure.AXES])
        self.assertNotIn("individual", [axis.key for axis in figure.AXES])

    def test_figure_pipeline_exactly_regenerates_main_table(self):
        figure.assert_main_table_agreement(
            self.args, self.points, self.table_scores, self.entropies
        )

    def test_only_parallel_lm_pairs_are_collapsed(self):
        self.assertEqual(len(self.full_profiles), 10)
        self.assertEqual(len(self.profiles), 8)
        self.assertEqual(
            set(self.collapsed), {"moses_2021_v50", "willett_2023_v50"}
        )
        self.assertFalse(any(profile.label.endswith("isolated") for profile in self.profiles))

    def test_reversing_pairs_and_crossing_segments(self):
        reversals = figure.find_reversals(self.profiles)
        self.assertGreater(len(reversals), 0)
        highlighted = figure.crossing_segments(self.profiles)
        self.assertEqual(highlighted.shape, (8, len(figure.AXES) - 1))
        self.assertTrue(bool(highlighted.any()))

    def test_reference_rank_correlations_are_well_formed(self):
        correlations = figure.spearman_correlations(self.profiles)
        self.assertEqual(correlations.shape, (4, 4))
        self.assertTrue(bool(np.isfinite(correlations.to_numpy()).all()))

    def test_figure_is_eight_by_three_inches(self):
        generated = figure.draw_parallel_coordinates(
            self.profiles, self.entropies, "log"
        )
        self.assertTrue(np.allclose(generated.get_size_inches(), [8.0, 3.0]))
        figure.plt.close(generated)

    def test_trace_identity_does_not_depend_on_colour(self):
        style_signatures = {
            (repr(figure.line_style(profile)), figure.marker_style(profile))
            for profile in self.profiles
        }
        self.assertEqual(len(style_signatures), len(self.profiles))

    def test_invasive_markers_are_filled_and_non_invasive_markers_are_hollow(self):
        for profile in self.profiles:
            expected = profile.color if profile.point.group == "attempted" else "white"
            self.assertEqual(figure.marker_facecolor(profile), expected)

    def test_all_requested_artifacts_exist(self):
        expected = [
            PROJECT_ROOT / "figures/reference_dependence.pdf",
            PROJECT_ROOT / "figures/reference_dependence.png",
            PROJECT_ROOT / "figures/reference_dependence_linear.pdf",
            PROJECT_ROOT / "figures/reference_dependence_linear.png",
            PROJECT_ROOT / "figures/reference_dependence_log.pdf",
            PROJECT_ROOT / "figures/reference_dependence_log.png",
            figure.DEFAULT_CAPTION_OUTPUT,
        ]
        for path in expected:
            self.assertTrue(path.exists(), msg=str(path))
            self.assertGreater(path.stat().st_size, 0, msg=str(path))
        caption = figure.DEFAULT_CAPTION_OUTPUT.read_text(encoding="utf-8")
        self.assertIn("SUBTLEX-UK, Switchboard, UCV, and Sherlock", caption)
        self.assertNotIn("same broad-spoken-English target", caption)


if __name__ == "__main__":
    unittest.main()
