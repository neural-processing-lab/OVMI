import sys
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import make_vocabulary_decomposition_figure as figure


class VocabularyDecompositionFigureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.reference = figure.table.load_reference(figure.DEFAULT_REFERENCE)
        cls.metrics = figure.compute_metrics(cls.reference)

    def test_representative_bars_sum_exactly_to_log_capacity(self):
        retained, coverage_loss, nonuniformity, totals = figure.components_at(
            self.metrics
        )
        self.assertTrue(bool((retained >= 0).all()))
        self.assertTrue(bool((coverage_loss >= 0).all()))
        self.assertTrue(bool((nonuniformity >= 0).all()))
        np.testing.assert_allclose(
            retained + coverage_loss + nonuniformity,
            totals,
            rtol=0.0,
            atol=figure.CHECK_TOLERANCE,
        )

    def test_noiseless_ovmi_equals_coverage_times_entropy(self):
        np.testing.assert_allclose(
            self.metrics.retained_ovmi,
            self.metrics.coverage * self.metrics.conditional_entropy,
            rtol=0.0,
            atol=1e-12,
        )
        for vocabulary_size in figure.REPRESENTATIVE_SIZES:
            direct = figure.ovmi(
                self.reference,
                self.metrics.ranked_words[:vocabulary_size],
                accuracy=1.0,
            )
            self.assertAlmostEqual(
                direct,
                self.metrics.retained_ovmi[vocabulary_size - 1],
                places=10,
            )

    def test_subtlex_anchor_values_are_stable(self):
        expected = {
            50: (0.455361, 5.182198, 2.359769),
            250: (0.675014, 6.801807, 4.591315),
            1_000: (0.802528, 7.848797, 6.298884),
            15_000: (0.972095, 9.413760, 9.151071),
        }
        for vocabulary_size, values in expected.items():
            index = vocabulary_size - 1
            actual = (
                self.metrics.coverage[index],
                self.metrics.conditional_entropy[index],
                self.metrics.retained_ovmi[index],
            )
            np.testing.assert_allclose(actual, values, rtol=0.0, atol=5e-7)

    def test_outputs_and_caption_exist(self):
        for extension in ("pdf", "png"):
            path = figure.DEFAULT_OUTPUT_BASE.with_suffix(f".{extension}")
            self.assertTrue(path.exists(), msg=str(path))
            self.assertGreater(path.stat().st_size, 0, msg=str(path))
        caption = figure.DEFAULT_CAPTION_OUTPUT.read_text(encoding="utf-8")
        self.assertIn("exact identity", caption)
        self.assertIn("$P=1$", caption)
        self.assertIn("is the coverage gap", caption)

    def test_panel_b_legend_mirrors_stack_from_top_to_bottom(self):
        rendered = figure.draw_figure(self.metrics)
        try:
            labels = [
                text.get_text() for text in rendered.axes[1].get_legend().get_texts()
            ]
            self.assertTrue(labels[0].startswith("Non-uniformity"))
            self.assertTrue(labels[1].startswith("Coverage gap"))
            self.assertTrue(labels[2].startswith("OVMI"))
        finally:
            figure.plt.close(rendered)


if __name__ == "__main__":
    unittest.main()
