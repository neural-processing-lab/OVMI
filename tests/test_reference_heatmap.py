import sys
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import make_reference_heatmap as heatmap


class ReferenceHeatmapTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = heatmap.load_matrix()

    def test_shape_and_card_anchor(self):
        self.assertEqual(self.data.values.shape, (7, 4))
        np.testing.assert_allclose(
            self.data.values[0], (93.7, 96.9, 97.5, 95.9), atol=0.05,
        )

    def test_values_are_log_compatible_percentages(self):
        self.assertTrue(np.all(self.data.values > 0.0))
        self.assertTrue(np.all(self.data.values <= 100.0))
        self.assertTrue(np.all(self.data.lows <= self.data.values))
        self.assertTrue(np.all(self.data.values <= self.data.highs))

    def test_uncertainty_kinds_are_not_pooled(self):
        self.assertEqual(heatmap.ROW_SPECS[-1].label, "LibriBrain100")
        self.assertEqual(
            self.data.uncertainty_kinds,
            (
                "bootstrap95", "bootstrap95", "bootstrap95", "published95",
                "wilson95", "participant_sem", "seed_sem",
            ),
        )
        point, uncertainty = heatmap.cell_annotation(self.data, 6, 0)
        self.assertEqual(point, "2.43")
        self.assertTrue(uncertainty.startswith(r"$\pm$"))

    def test_figure_is_eight_by_three_with_log_colour(self):
        figure = heatmap.draw_figure(self.data)
        try:
            self.assertEqual(tuple(figure.get_size_inches()), (8.0, 3.25))
            self.assertIsInstance(figure.axes[0].images[0].norm, heatmap.LogNorm)
            self.assertEqual(
                [label.get_text() for label in figure.axes[0].get_xticklabels()],
                [
                    "Broad speech\n(SUBTLEX-UK)",
                    "Conversation\n(Switchboard)",
                    "AAC\n(UCV)",
                    "Narrative\n(Sherlock)",
                ],
            )
        finally:
            heatmap.plt.close(figure)


if __name__ == "__main__":
    unittest.main()
