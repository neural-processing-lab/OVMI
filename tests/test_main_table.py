import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts/make_main_table.py"


def load_table_module():
    spec = importlib.util.spec_from_file_location("make_main_table", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class MainTableTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = load_table_module()
        cls.systems = pd.read_csv(cls.module.DEFAULT_SYSTEMS)
        cls.systems = cls.systems.loc[cls.systems["plot_eligible"].astype(bool)].copy()
        cls.references, cls.entropies = cls.module.load_references(
            cls.module.DEFAULT_REFERENCES
        )
        cls.vocabularies = cls.module.reconstruct_vocabularies(
            cls.systems,
            cls.references,
            cls.module.DEFAULT_PREDICTIONS,
            cls.module.DEFAULT_CMUDICT,
            cls.module.DEFAULT_ARMENI_TEXT,
        )
        cls.points = cls.module.expand_system_points(cls.systems)
        cls.scores = cls.module.score_all(
            cls.points, cls.vocabularies, cls.references, cls.entropies
        )
        cls.uncertainties = cls.module.score_uncertainties(
            cls.points, cls.vocabularies, cls.references, cls.entropies
        )

    def test_reference_entropies_are_stable(self):
        expected = {
            "subtlex": 9.771728917643,
            "conversation": 8.268285074700,
            "ucv": 4.304625359994,
            "narrative": 8.443791992670,
            "individual": 4.274131918965,
        }
        for key, value in expected.items():
            self.assertAlmostEqual(self.entropies[key], value, places=10)

    def test_same_ten_points_and_variants_as_contour(self):
        self.assertEqual(len(self.points), 10)
        self.assertEqual(sum(point.group == "attempted" for point in self.points), 6)
        self.assertEqual(sum(point.group == "perceived" for point in self.points), 4)
        self.assertEqual(
            sum(point.system_id == "moses_2021_v50" for point in self.points), 2
        )
        self.assertEqual(
            sum(point.system_id == "willett_2023_v50" for point in self.points), 2
        )

    def test_subtlex_cells_match_contour_csv(self):
        errors = [
            abs(self.scores[(index, "subtlex")].score - point.csv_ovmi)
            for index, point in enumerate(self.points)
        ]
        self.assertLess(max(errors), 5e-10)

    def test_every_cell_satisfies_factorisation_and_ceilings(self):
        for index, _point in enumerate(self.points):
            for spec in self.module.REFERENCES:
                cell = self.scores[(index, spec.key)]
                self.assertAlmostEqual(
                    cell.score,
                    cell.coverage * cell.in_vocab_information,
                    places=12,
                )
                self.assertLessEqual(
                    cell.score, cell.coverage * cell.own_entropy + 1e-10
                )
                self.assertLessEqual(cell.percentage, 100.0 + 1e-10)

    def test_generated_latex_variants_are_current_and_have_nine_columns(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            generated = Path(temporary_directory) / "main_table.tex"
            self.module.render_table(
                self.points, self.scores, self.uncertainties,
                self.entropies, generated
            )
            self.assertEqual(
                generated.read_text(encoding="utf-8"),
                self.module.DEFAULT_OUTPUT.read_text(encoding="utf-8"),
            )
            generated_bits = Path(temporary_directory) / "main_table_bits_only.tex"
            self.module.render_table(
                self.points, self.scores, self.uncertainties,
                self.entropies, generated_bits, include_percentages=False,
            )
            self.assertEqual(
                generated_bits.read_text(encoding="utf-8"),
                self.module.DEFAULT_BITS_ONLY_OUTPUT.read_text(encoding="utf-8"),
            )
            generated_no_gap = (
                Path(temporary_directory) / "main_table_no_specialisation_gap.tex"
            )
            self.module.render_table(
                self.points, self.scores, self.uncertainties,
                self.entropies, generated_no_gap,
                include_specialisation_gap=False,
                include_design_alignment=False,
            )
            self.assertEqual(
                generated_no_gap.read_text(encoding="utf-8"),
                self.module.DEFAULT_NO_SPECIALISATION_GAP_OUTPUT.read_text(
                    encoding="utf-8"
                ),
            )
        table = self.module.DEFAULT_OUTPUT.read_text(encoding="utf-8")
        self.assertIn(r"\label{tab:main}", table)
        self.assertIn(r"\scriptsize", table)
        self.assertIn(r"\setlength{\tabcolsep}{1.1pt}", table)
        self.assertEqual(table.count(r"\addlinespace[1.2pt]"), 8)
        self.assertNotIn(r"\resizebox", table)
        self.assertEqual(table.count(r"$H(p)="), 4)
        self.assertNotIn("Individual target\\\\Moses set", table)
        for line in table.splitlines():
            if line.endswith(r"\\") and "multicolumn" not in line:
                self.assertEqual(line.count(" & "), 7, msg=line)

        no_gap_table = self.module.DEFAULT_NO_SPECIALISATION_GAP_OUTPUT.read_text(
            encoding="utf-8"
        )
        self.assertIn(r"\label{tab:main-no-specialisation-gap}", no_gap_table)
        self.assertNotIn("Specialisation", no_gap_table)
        self.assertNotIn("specialisation gap", no_gap_table)
        self.assertNotIn(r"\textbf{\shortstack", no_gap_table)
        self.assertNotIn(r"$^{\star}$", no_gap_table)
        self.assertNotIn(r"Bold $\star$ cells", no_gap_table)
        for line in no_gap_table.splitlines():
            if line.endswith(r"\\") and "multicolumn" not in line:
                self.assertEqual(line.count(" & "), 6, msg=line)

    def test_every_point_has_uncertainty(self):
        self.assertEqual(len(self.uncertainties), 10 * len(self.module.REFERENCES))
        missing = [
            point.display_name
            for index, point in enumerate(self.points)
            if (index, "subtlex") not in self.uncertainties
        ]
        self.assertEqual(missing, [])

    def test_uncertainty_notation_distinguishes_sampling_from_seed_sem(self):
        table = self.module.DEFAULT_OUTPUT.read_text(encoding="utf-8")
        self.assertIn(r"$0.238$ {\tiny (2.4\%)}\\$[0.230,0.245]$", table)
        self.assertIn(r"$0.237$ {\tiny (2.4\%)}\\$\pm 0.003$", table)
        self.assertNotIn(r"\ddagger", table)

    def test_key_subtlex_comparisons_use_propagated_endpoints(self):
        by_key = {
            (point.system_id, point.probability_source): index
            for index, point in enumerate(self.points)
        }
        libri_index = by_key[("dascoli_libribrain100_s0_v50", "b")]
        moses_index = by_key[("moses_2021_v50", "a")]
        libri = self.scores[(libri_index, "subtlex")]
        moses = self.scores[(moses_index, "subtlex")]
        libri_uncertainty = self.uncertainties[(libri_index, "subtlex")]
        moses_uncertainty = self.uncertainties[(moses_index, "subtlex")]
        difference = abs(libri.score - moses.score)
        self.assertLess(difference, libri.score - libri_uncertainty.low.score)
        self.assertLess(difference, moses.score - moses_uncertainty.low.score)

        willett_isolated = by_key[("willett_2023_v50", "a")]
        willett_lm = by_key[("willett_2023_v50", "w")]
        isolated_interval = self.uncertainties[(willett_isolated, "subtlex")]
        lm_interval = self.uncertainties[(willett_lm, "subtlex")]
        self.assertLessEqual(isolated_interval.low.score, lm_interval.high.score)

    def test_individual_target_is_in_appendix_table(self):
        appendix = self.module.DEFAULT_APPENDIX_OUTPUT.read_text(encoding="utf-8")
        self.assertIn(r"\label{tab:main-individual-target}", appendix)
        self.assertIn("Individual target\\\\Moses set", appendix)
        self.assertEqual(appendix.count(r"$H(p)="), 1)
        appendix_bits = self.module.DEFAULT_APPENDIX_BITS_ONLY_OUTPUT.read_text(
            encoding="utf-8"
        )
        self.assertIn(r"\label{tab:main-individual-target-bits-only}", appendix_bits)

    def test_not_all_reference_rankings_are_redundant(self):
        values = pd.DataFrame({
            spec.key: [
                self.scores[(index, spec.key)].score
                for index in range(len(self.points))
            ]
            for spec in self.module.REFERENCES
        })
        ranks = values.rank(method="average", ascending=False)
        correlations = ranks.corr()
        off_diagonal = correlations.to_numpy()[
            ~np.eye(len(correlations), dtype=bool)
        ]
        self.assertTrue(bool((off_diagonal < 1.0 - 1e-12).any()))


if __name__ == "__main__":
    unittest.main()
