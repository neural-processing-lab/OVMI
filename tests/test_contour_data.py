import unittest
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SYSTEMS_PATH = PROJECT_ROOT / "data/systems.csv"
FRONTIER_PATH = PROJECT_ROOT / "data/noiseless_frequency_frontier.csv"


class ContourDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.systems = pd.read_csv(SYSTEMS_PATH)
        cls.frontier = pd.read_csv(FRONTIER_PATH)

    def test_every_plotted_information_value_respects_own_vocabulary_entropy(self):
        eligible = self.systems.loc[self.systems["plot_eligible"]]
        for information_column in (
            "I_invocab_neural_bits",
            "I_invocab_system_bits",
        ):
            finite = eligible.loc[np.isfinite(eligible[information_column])]
            excess = finite[information_column] - finite["H_pS_bits"]
            self.assertTrue(
                bool((excess <= 1e-10).all()),
                msg=f"Own-vocabulary entropy violations:\n{finite.loc[excess > 1e-10]}",
            )

    def test_frontier_terminates_at_full_reference_entropy(self):
        endpoint = self.frontier.sort_values("V").iloc[-1]
        reference_entropy = self.systems["reference_entropy_bits"].dropna().iloc[0]
        self.assertAlmostEqual(endpoint["coverage"], 1.0, places=12)
        self.assertAlmostEqual(endpoint["H_pS_bits"], reference_entropy, places=10)

    def test_frontier_is_monotone_in_vocabulary_size_and_coverage(self):
        frontier = self.frontier.sort_values("V")
        self.assertTrue(bool((np.diff(frontier["V"]) > 0).all()))
        self.assertTrue(bool((np.diff(frontier["coverage"]) > 0).all()))

    def test_subtlex_frontier_matches_broad_zipfian_anchors(self):
        expected = {
            50: (0.45, 4.6),
            250: (0.65, 6.2),
            1_000: (0.80, 7.5),
            15_000: (0.95, 9.9),
        }
        for vocabulary_size, (coverage, entropy) in expected.items():
            row = self.frontier.loc[self.frontier["V"] == vocabulary_size].iloc[0]
            self.assertLess(abs(row["coverage"] - coverage), 0.05)
            self.assertLess(abs(row["H_pS_bits"] - entropy), 0.75)

    def test_dascoli_uses_libribrain_three_seed_result_and_exact_vocabulary(self):
        system = self.systems.loc[
            self.systems["system_id"] == "dascoli_libribrain100_s0_v50"
        ].iloc[0]
        self.assertEqual(system["V"], 50)
        self.assertAlmostEqual(system["P_neural"], 0.258, places=12)
        self.assertAlmostEqual(system["P_neural_ci_low"], 0.256, places=10)
        self.assertAlmostEqual(system["P_neural_ci_high"], 0.260, places=10)
        self.assertEqual(system["uncertainty_neural"], "seed_sem")
        self.assertAlmostEqual(system["coverage"], 0.389913010102, places=10)
        self.assertAlmostEqual(system["OVMI_neural_bits"], 0.237463086598, places=10)
        self.assertIn("LibriBrain100 subject 0", system["source"])
        self.assertIn("Sherlock test set", system["source"])

    def test_armeni_uses_three_seed_result_and_frequency_derived_vocabulary(self):
        system = self.systems.loc[
            self.systems["system_id"] == "armeni_2022_v50"
        ].iloc[0]
        self.assertEqual(system["V"], 50)
        self.assertAlmostEqual(system["P_neural"], 0.207666666667, places=10)
        self.assertAlmostEqual(system["P_neural_ci_low"], 0.204818665418, places=10)
        self.assertAlmostEqual(system["P_neural_ci_high"], 0.210514667915, places=10)
        self.assertEqual(system["uncertainty_neural"], "seed_sem")
        self.assertAlmostEqual(system["coverage"], 0.407596948556, places=10)
        self.assertAlmostEqual(system["OVMI_neural_bits"], 0.175573192157, places=10)
        self.assertIn("top-50 frequency-ranked tokens", system["vocabulary_kind"])
        self.assertIn("sherlock-holm.es/stories/plain-text/advs.txt", system["source"])

    def test_meg_masc_uses_three_seed_mean_sem_and_derived_vocabulary(self):
        system = self.systems.loc[
            self.systems["system_id"] == "meg_masc_2023_v50"
        ].iloc[0]
        self.assertEqual(system["V"], 50)
        self.assertAlmostEqual(system["P_neural"], np.mean([0.093, 0.083, 0.080]), places=12)
        expected_sem = np.std([0.093, 0.083, 0.080], ddof=1) / np.sqrt(3)
        self.assertAlmostEqual(
            system["P_neural_ci_low"], system["P_neural"] - expected_sem, places=12
        )
        self.assertAlmostEqual(
            system["P_neural_ci_high"], system["P_neural"] + expected_sem, places=12
        )
        self.assertAlmostEqual(system["coverage"], 0.397030280231, places=10)
        self.assertAlmostEqual(system["OVMI_neural_bits"], 0.033248154823, places=10)
        self.assertEqual(system["uncertainty_neural"], "seed_sem")
        self.assertEqual(system["seed_values_neural"], "0.093;0.083;0.080")
        self.assertEqual(system["n_seeds_neural"], 3)
        self.assertIn("four MEG-MASC stories", system["vocabulary_kind"])
        self.assertIn("top-1 accuracies 9.3%, 8.3%, and 8.0%", system["source"])

        vocabulary = pd.read_csv(
            PROJECT_ROOT / "data/vocabularies/meg_masc_2023_v50.csv",
            keep_default_na=False,
        )
        self.assertEqual(len(vocabulary), 50)
        self.assertEqual(vocabulary.iloc[0]["word"], "the")
        self.assertEqual(vocabulary.iloc[-1]["word"], "one")
        self.assertIn("'s", set(vocabulary["word"]))
        self.assertIn("n't", set(vocabulary["word"]))

    def test_tang_uses_participant_mean_sem_and_published_vocabulary(self):
        system = self.systems.loc[
            self.systems["system_id"] == "tang_2023_v6867"
        ].iloc[0]
        self.assertEqual(system["V"], 6867)
        self.assertAlmostEqual(system["P_system"], 0.0665333333333, places=12)
        self.assertAlmostEqual(system["P_system_ci_low"], 0.0617013795072, places=12)
        self.assertAlmostEqual(system["P_system_ci_high"], 0.0713652871595, places=12)
        self.assertEqual(system["uncertainty_system"], "participant_sem")
        self.assertAlmostEqual(system["coverage"], 0.850072343675, places=10)
        self.assertAlmostEqual(system["OVMI_system_bits"], 0.356638612910, places=10)
        self.assertIn("S1 94.07%, S2 93.54%, and S3 92.43%", system["source"])

    def test_all_propagated_interval_endpoints_respect_own_entropy(self):
        finite = self.systems.loc[self.systems["I_neural_ci_high_bits"].notna()]
        self.assertTrue(
            bool((finite["I_neural_ci_high_bits"] <= finite["H_pS_bits"] + 1e-10).all())
        )

    def test_invasive_sampling_inputs_and_counts_are_provenanced(self):
        rows = self.systems.set_index("system_id")
        moses = rows.loc["moses_2021_v50"]
        self.assertEqual(moses["n_trials"], 9000)
        self.assertEqual(moses["n_system_sentences"], 150)
        self.assertEqual(moses["n_system_blocks"], 15)
        self.assertAlmostEqual(moses["P_system_ci_low"], 0.629)
        self.assertAlmostEqual(moses["P_system_ci_high"], 0.829)

        willett50 = rows.loc["willett_2023_v50"]
        willett125 = rows.loc["willett_2023_v125k"]
        self.assertEqual(willett50["n_system_sentences"], 250)
        self.assertEqual(willett125["n_system_sentences"], 400)
        self.assertEqual(willett50["uncertainty_system"], "bootstrap95")
        self.assertEqual(willett125["uncertainty_system"], "bootstrap95")

        card = rows.loc["card_2024_v125k"]
        self.assertEqual(card["n_system_sessions"], 5)
        self.assertEqual(card["uncertainty_system"], "bootstrap95")
        self.assertIn("exact sentence count not reported", card["ci_method"])

    def test_plotted_task_and_access_groups_are_perfectly_confounded(self):
        plotted = self.systems.loc[self.systems["plot_eligible"]]
        groups = set(
            plotted[["speech_condition", "invasiveness"]].itertuples(
                index=False, name=None
            )
        )
        self.assertEqual(
            groups,
            {("perceived", "non-invasive"), ("attempted", "invasive")},
        )

    def test_every_plotted_label_uses_explicit_vocabulary_notation(self):
        plotted = self.systems.loc[self.systems["plot_eligible"]]
        self.assertTrue(bool(plotted["label"].str.contains("V=").all()))


if __name__ == "__main__":
    unittest.main()
