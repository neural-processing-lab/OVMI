import json
import math
import re
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SITE_ROOT = PROJECT_ROOT / "site"


class StaticSiteTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.payload = json.loads(
            (SITE_ROOT / "data/leaderboard.json").read_text(encoding="utf-8")
        )
        cls.html = (SITE_ROOT / "index.html").read_text(encoding="utf-8")
        cls.javascript = (SITE_ROOT / "app.js").read_text(encoding="utf-8")

    def test_benchmark_has_four_references_and_ten_rows(self):
        self.assertEqual(
            set(self.payload["references"]),
            {"subtlex", "conversation", "ucv", "narrative"},
        )
        self.assertEqual(len(self.payload["systems"]), 10)
        self.assertEqual(
            len({system["id"] for system in self.payload["systems"]}), 10
        )

    def test_reference_entropies_match_analysis(self):
        expected = {
            "subtlex": 9.771728917643,
            "conversation": 8.268285074700,
            "ucv": 4.304625359994,
            "narrative": 8.443791992670,
        }
        for key, entropy in expected.items():
            self.assertAlmostEqual(
                self.payload["references"][key]["entropy_bits"], entropy, places=10
            )

    def test_every_site_score_satisfies_ovmi_factorisation(self):
        reference_keys = set(self.payload["references"])
        for system in self.payload["systems"]:
            self.assertEqual(set(system["references"]), reference_keys)
            for result in system["references"].values():
                self.assertTrue(math.isfinite(result["ovmi_bits"]))
                self.assertAlmostEqual(
                    result["ovmi_bits"],
                    result["coverage"] * result["in_vocab_information_bits"],
                    places=11,
                )
                self.assertLessEqual(result["ovmi_percent"], 100.0 + 1e-10)

    def test_headline_values_match_paper_table(self):
        by_id = {system["id"]: system for system in self.payload["systems"]}
        card = by_id["card_2024_v125k-w"]["references"]["subtlex"]
        willett_50 = by_id["willett_2023_v50-w"]["references"]
        self.assertAlmostEqual(card["ovmi_bits"], 9.156, delta=0.0005)
        self.assertAlmostEqual(card["ovmi_percent"], 93.7, delta=0.05)
        self.assertAlmostEqual(
            willett_50["subtlex"]["ovmi_percent"], 6.4, delta=0.05
        )
        self.assertAlmostEqual(
            willett_50["ucv"]["ovmi_percent"], 40.4, delta=0.05
        )

    def test_wer_rows_are_marked_as_conservative_lower_bounds(self):
        wer_rows = [
            system for system in self.payload["systems"]
            if system["metric"]["type"] == "wer"
        ]
        self.assertEqual(len(wer_rows), 5)
        self.assertTrue(all(
            system["metric"]["p_is_lower_bound"] for system in wer_rows
        ))
        self.assertIn("conservative lower bound", self.javascript)

    def test_tang_mean_and_sem_use_the_published_decoder_vocabulary(self):
        tang_rows = [
            system for system in self.payload["systems"]
            if system["system_id"] == "tang_2023_v6867"
        ]
        self.assertEqual(len(tang_rows), 1)
        system = tang_rows[0]
        self.assertEqual(system["system"], "Tang 2023")
        self.assertEqual(system["group"], "perceived_noninvasive")
        self.assertEqual(system["modality"], "fMRI")
        self.assertEqual(system["vocabulary_size"], 6867)
        self.assertEqual(system["decoder_method"], "Tang et al. (2023) semantic decoder")
        self.assertAlmostEqual(system["metric"]["reported_value"], 0.9334666666666667)
        self.assertAlmostEqual(system["metric"]["reported_sem"], 0.004831953826122276)
        self.assertTrue(system["metric"]["p_is_lower_bound"])
        self.assertTrue(all(
            result["uncertainty"]["kind"] == "participant_sem"
            for result in system["references"].values()
        ))
        self.assertIn("SEM across participants", self.javascript)

    def test_noninvasive_decoder_provenance_is_explicit(self):
        by_id = {system["system_id"]: system for system in self.payload["systems"]}
        self.assertEqual(by_id["meg_masc_2023_v50"]["decoder_method"], "MEG-XL")
        for system_id in ("dascoli_libribrain100_s0_v50", "armeni_2022_v50"):
            self.assertEqual(
                by_id[system_id]["decoder_method"], "d’Ascoli et al. (2025)"
            )
        self.assertIn("decoder_method", self.javascript)
        self.assertIn("Decoder:", self.javascript)

    def test_site_uses_relative_asset_and_data_paths(self):
        for path in (
            "./styles.css", "./config.js", "./app.js",
            "./data/leaderboard.json", "./assets/favicon.svg",
        ):
            self.assertIn(path, self.html + self.javascript)
        self.assertIsNone(re.search(r'(?:src|href)="/(?!/)', self.html))

    def test_favicon_uses_the_ovmi_mark(self):
        favicon = (SITE_ROOT / "assets/favicon.svg").read_text(encoding="utf-8")
        self.assertIn('aria-label="OVMI mark"', favicon)
        self.assertIn('fill="#0f504c"', favicon)
        self.assertNotIn(">MI<", favicon)

    def test_required_interactive_regions_and_scientific_caveats_exist(self):
        for element_id in (
            "leaderboard", "reference-selector", "leaderboard-body",
            "methodology", "add-result", "use-ovmi",
        ):
            self.assertIn(f'id="{element_id}"', self.html)
        self.assertIn("OVMI Explorer", self.html)
        self.assertIn('id="interpretation"', self.html)
        self.assertIn("How to interpret the comparison", self.html)
        self.assertIn(
            "OVMI puts systems on a common communication scale, but it does not make their underlying experiments equivalent.",
            self.html,
        )
        self.assertIn(
            "An interactive comparison of published speech-BCI systems under "
            "different reference communication distributions.",
            self.html,
        )
        self.assertIn(
            "Open-Vocabulary Mutual Information (OVMI) evaluates each system against an explicit, common reference communication distribution",
            self.html,
        )
        self.assertNotIn(">Rank<", self.html)
        self.assertIn(
            "Historical results are estimated from reported accuracy or WER",
            self.html,
        )
        self.assertIn("Submit a result →", self.html)

    def test_paper_identity_is_prominent(self):
        self.assertIn(
            "A Common Measure of Communication for<br>"
            "Speech Brain–Computer Interfaces",
            self.html,
        )
        for author in ("Dulhan Jayalath", "Benjamin Ballyk", "Oiwi Parker Jones"):
            self.assertIn(author, self.html)
        self.assertIn("PNPL🍍, University of Oxford", self.html)
        self.assertIn('href="https://neural-processing-lab.github.io/"', self.html)
        self.assertIn("Research paper · interactive explorer", self.html)
        self.assertNotIn("Paper overview", self.html)
        self.assertNotIn("The metric in one line", self.html)
        self.assertEqual(self.html.count('class="button-icon"'), 4)

    def test_why_ovmi_defines_conditional_information_and_application_scope(self):
        self.assertIn(
            "Speech-BCI studies use different vocabularies and communication domains",
            self.html,
        )
        self.assertIn(
            "OVMI multiplies the speech BCI's vocabulary coverage of the reference distribution by its in-vocabulary information transfer.",
            self.html,
        )
        self.assertIn("I(X;Y | X∈S)", self.html)
        self.assertIn("A spelling system needs broad lexical coverage", self.html)

    def test_removed_long_form_sections_are_absent(self):
        for element_id in ("communication-plane", "progress-plot"):
            self.assertNotIn(f'id="{element_id}"', self.html)
        self.assertNotIn("renderProgress", self.javascript)

    def test_explorer_has_split_scatter_and_sort_linked_comparison_bars(self):
        self.assertIn('id="benchmark-scatter"', self.html)
        self.assertIn('id="group-filter"', self.html)
        self.assertIn('value="attempted_invasive" checked>Invasive', self.html)
        self.assertNotIn('value="all" checked>Both', self.html)
        self.assertIn('id="bar-metric-label"', self.html)
        self.assertIn("renderBenchmarkScatter", self.javascript)
        self.assertIn("nicePlotMaximum", self.javascript)
        self.assertIn("comparisonBarHtml", self.javascript)
        self.assertIn("score-interval", self.javascript)

    def test_bibtex_citation_matches_repository_entry(self):
        self.assertIn('@article{jayalath2026ovmi,', self.html)
        self.assertIn(
            "A Common Measure of Communication for Speech Brain--Computer Interfaces",
            self.html,
        )
        self.assertIn("arXiv preprint arXiv:PLACEHOLDER", self.html)

    def test_pages_workflow_and_submission_template_exist(self):
        workflow = (
            PROJECT_ROOT / ".github/workflows/pages.yml"
        ).read_text(encoding="utf-8")
        self.assertIn("path: site", workflow)
        self.assertIn("actions/deploy-pages@v4", workflow)
        self.assertTrue(
            (PROJECT_ROOT / ".github/ISSUE_TEMPLATE/add-ovmi-result.yml").exists()
        )

    def test_readme_links_to_public_site(self):
        readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
        self.assertIn(
            "https://neural-processing-lab.github.io/OVMI/", readme
        )


if __name__ == "__main__":
    unittest.main()
