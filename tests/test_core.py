import math
import unittest
from unittest.mock import patch

import numpy as np

from ovmi import coverage, full_ovmi, ovmi, scalar_ovmi


class CoreTests(unittest.TestCase):
    def test_coverage_normalises_counts(self):
        reference = {"yes": 3, "no": 1, "water": 2}

        self.assertAlmostEqual(coverage(reference, ["yes", "water"]), 5 / 6)

    def test_scalar_ovmi_matches_symmetric_channel_full_ovmi(self):
        reference = {"yes": 3, "no": 1, "water": 2}
        vocabulary = ["yes", "no", "water"]
        accuracy = 0.7
        off_diagonal = (1 - accuracy) / (len(vocabulary) - 1)
        channel = np.full((3, 3), off_diagonal)
        np.fill_diagonal(channel, accuracy)

        scalar = scalar_ovmi(reference, vocabulary, accuracy=accuracy)
        full = full_ovmi(reference, vocabulary, confusion_matrix=channel)

        self.assertAlmostEqual(scalar, full)

    def test_scalar_ovmi_accepts_per_word_accuracies(self):
        reference = {"yes": 3, "no": 1, "water": 2}
        vocabulary = ["yes", "no", "water"]
        accuracies = {"yes": 0.9, "no": 0.6, "water": 0.75}
        channel = np.array([
            [0.9, 0.05, 0.05],
            [0.2, 0.6, 0.2],
            [0.125, 0.125, 0.75],
        ])

        scalar = scalar_ovmi(reference, vocabulary, accuracy=accuracies)
        full = full_ovmi(reference, vocabulary, confusion_matrix=channel)

        self.assertAlmostEqual(scalar, full)

    def test_scalar_ovmi_matches_plotting_macro_formula(self):
        reference = {"a": 7, "b": 5, "c": 3, "outside": 5}
        vocabulary = ["a", "b", "c"]
        accuracies = np.array([0.8, 0.6, 0.4])
        macro_accuracy = float(np.mean(accuracies))

        self.assertAlmostEqual(
            scalar_ovmi(reference, vocabulary, accuracy=macro_accuracy),
            _plotting_scalar_ovmi(reference, vocabulary, macro_accuracy),
        )

    def test_scalar_ovmi_matches_plotting_per_word_formula(self):
        reference = {"a": 7, "b": 5, "c": 3, "outside": 5}
        vocabulary = ["a", "b", "c"]
        accuracies = {"a": 0.8, "b": 0.6, "c": 0.4}

        self.assertAlmostEqual(
            scalar_ovmi(reference, vocabulary, accuracy=accuracies),
            _plotting_per_word_ovmi(reference, vocabulary, accuracies),
        )

    def test_scalar_ovmi_rejects_missing_per_word_accuracy(self):
        reference = {"yes": 1, "no": 1}

        with self.assertRaises(ValueError):
            scalar_ovmi(reference, ["yes", "no"], accuracy={"yes": 0.8})

    def test_full_ovmi_uses_labeled_submatrix(self):
        reference = {"a": 5, "b": 3, "c": 2}
        labels = ["a", "b", "c"]
        confusion = np.array([
            [8, 1, 1],
            [2, 7, 1],
            [1, 2, 7],
        ])

        result = full_ovmi(
            reference,
            ["a", "c"],
            confusion_matrix=confusion,
            labels=labels,
            return_details=True,
        )

        self.assertAlmostEqual(result.coverage, 0.7)
        self.assertEqual(result.vocabulary_size, 2)
        self.assertGreater(result.score, 0)

    def test_full_ovmi_requires_numpy_confusion_matrix(self):
        reference = {"a": 1, "b": 1}

        with self.assertRaises(TypeError):
            full_ovmi(reference, ["a", "b"], confusion_matrix=[[1, 0], [0, 1]])

    def test_dispatcher_defaults_to_scalar(self):
        reference = {"a": 1, "b": 1}

        self.assertAlmostEqual(
            ovmi(reference, ["a", "b"], accuracy=0.8),
            scalar_ovmi(reference, ["a", "b"], accuracy=0.8),
        )

    def test_dispatcher_supports_full_method(self):
        reference = {"a": 1, "b": 1}
        confusion = np.array([[8, 2], [1, 9]])

        self.assertAlmostEqual(
            ovmi(reference, ["a", "b"], method="full", confusion_matrix=confusion),
            full_ovmi(reference, ["a", "b"], confusion_matrix=confusion),
        )

    def test_omitted_reference_uses_default_reference(self):
        default = {"a": 3, "b": 2, "c": 1}

        with patch("ovmi.core.default_reference", return_value=default):
            self.assertAlmostEqual(
                ovmi(["a", "b"], accuracy=0.8),
                ovmi(default, ["a", "b"], accuracy=0.8),
            )
            self.assertAlmostEqual(coverage(["a", "b"]), 5 / 6)

    def test_invalid_reference_rejected(self):
        with self.assertRaises(ValueError):
            scalar_ovmi({"a": -1}, ["a"], accuracy=1)

    def test_one_word_scalar_channel_requires_perfect_accuracy(self):
        with self.assertRaises(ValueError):
            scalar_ovmi({"a": 1}, ["a"], accuracy=0.9)

        self.assertTrue(math.isclose(scalar_ovmi({"a": 1}, ["a"], accuracy=1.0), 0.0))


def _plotting_scalar_ovmi(reference, vocabulary, accuracy):
    """Formula used by brainstorm/plot_*ovmi*.py scalar-P_c paths."""
    weights = np.array([reference[word] for word in vocabulary], dtype=float)
    coverage = weights.sum() / sum(reference.values())
    p_x = weights / weights.sum()
    vocab_size = len(vocabulary)
    off_diagonal = (1.0 - accuracy) / (vocab_size - 1)
    q_y = off_diagonal + p_x * (accuracy - off_diagonal)
    h_y = -float(np.sum(q_y[q_y > 0] * np.log2(q_y[q_y > 0])))
    h_yx = -accuracy * np.log2(accuracy) - (1.0 - accuracy) * np.log2(off_diagonal)
    return coverage * max(0.0, h_y - h_yx)


def _plotting_per_word_ovmi(reference, vocabulary, accuracies):
    """Formula used by brainstorm/plot_*ovmi*.py per-word scalar paths."""
    weights = np.array([reference[word] for word in vocabulary], dtype=float)
    coverage = weights.sum() / sum(reference.values())
    p_x = weights / weights.sum()
    word_acc = np.array([accuracies[word] for word in vocabulary], dtype=float)
    vocab_size = len(vocabulary)
    error_mass = np.sum(p_x * (1.0 - word_acc))
    q_y = p_x * word_acc + (error_mass - p_x * (1.0 - word_acc)) / (vocab_size - 1)
    h_y = -float(np.sum(q_y[q_y > 0] * np.log2(q_y[q_y > 0])))
    pc = np.clip(word_acc, 1e-15, 1.0 - 1e-15)
    h_yx_per_word = -pc * np.log2(pc) - (1.0 - pc) * np.log2((1.0 - pc) / (vocab_size - 1))
    h_yx = float(np.sum(p_x * h_yx_per_word))
    return coverage * max(0.0, h_y - h_yx)


if __name__ == "__main__":
    unittest.main()
