import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import figure6_estimator_sensitivity as figure6


class Figure6EstimatorSensitivityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.reference = figure6.load_reference_slice(figure6.DEFAULT_REFERENCE, 50)
        cls.result = figure6.run_simulation(
            cls.reference, samples=12, seed=1234,
            sigma_levels=np.asarray((0.0, 1.0, 6.0)),
            alpha_levels=np.asarray((np.inf, 10.0, 0.1)),
        )

    def test_reference_and_homogeneous_anchor(self):
        self.assertEqual(len(self.reference.words), 50)
        self.assertAlmostEqual(
            self.reference.conditional_probabilities.sum(), 1.0, places=13,
        )
        figure6.validate_homogeneous(self.reference, figure6.P_VALUES)
        np.testing.assert_allclose(self.result.a_relative_error[:, 0], 0.0, atol=1e-10)
        np.testing.assert_allclose(self.result.b_relative_error[:, 0], 0.0, atol=1e-10)

    def test_heterogeneous_accuracies_preserve_macro_p(self):
        accuracies = figure6.heterogeneous_accuracies(
            np.random.default_rng(5), 30, 50, 0.25, 8.0,
        )
        self.assertTrue(np.all(accuracies >= 0.0))
        self.assertTrue(np.all(accuracies <= 1.0))
        np.testing.assert_allclose(accuracies.mean(axis=1), 0.25, atol=2e-13)

    def test_simulation_shapes_and_finite_values(self):
        self.assertEqual(self.result.a_x.shape, (3, 3, 12))
        self.assertEqual(self.result.b_x.shape, (3, 3, 12))
        for values in (
            self.result.a_relative_error, self.result.a_absolute_error_bits,
            self.result.a_ratio, self.result.b_relative_error,
            self.result.b_absolute_error_bits, self.result.b_ratio,
            self.result.a_frequency_correlation,
        ):
            self.assertTrue(np.isfinite(values).all())

    def test_cache_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "simulation.npz"
            figure6.save_cache(self.result, path)
            loaded = figure6.load_cache(path)
            np.testing.assert_array_equal(
                loaded.a_relative_error, self.result.a_relative_error,
            )
            np.testing.assert_array_equal(loaded.b_x, self.result.b_x)
            self.assertEqual(loaded.samples, self.result.samples)

    def test_figure_is_eight_by_three(self):
        figure = figure6.draw_figure(self.result)
        try:
            self.assertEqual(tuple(figure.get_size_inches()), (8.0, 3.0))
            self.assertEqual(len(figure.axes), 2)
        finally:
            figure6.plt.close(figure)


if __name__ == "__main__":
    unittest.main()
