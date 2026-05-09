import math
import unittest
from unittest.mock import patch

import numpy as np

from ovmi import coverage, full_ovmi, optimize_vocabulary, ovmi, scalar_ovmi
from ovmi import (
    confusion_matrix_from_embeddings,
    confusion_matrix_from_logits,
    optimize_vocabulary_from_embeddings,
    optimize_vocabulary_from_logits,
    optimize_vocabulary_range,
)


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

    def test_omitted_reference_uses_default_reference(self):
        default = {"a": 3, "b": 2, "c": 1}

        with patch("ovmi.core.default_reference", return_value=default):
            self.assertAlmostEqual(
                ovmi(["a", "b"], accuracy=0.8),
                ovmi(default, ["a", "b"], accuracy=0.8),
            )
            self.assertAlmostEqual(coverage(["a", "b"]), 5 / 6)

    def test_omitted_reference_works_for_optimisation(self):
        default = {"a": 9, "b": 8, "c": 1}

        with patch("ovmi.core.default_reference", return_value=default):
            selected = optimize_vocabulary(default.keys(), size=2, accuracy=0.75)

        self.assertEqual(selected, ["a", "b"])

    def test_greedy_optimisation_selects_best_scalar_vocab(self):
        reference = {"a": 9, "b": 8, "c": 1}

        selected = optimize_vocabulary(reference, reference.keys(), size=2, accuracy=0.75)

        self.assertEqual(selected, ["a", "b"])

    def test_scalar_optimisation_accepts_per_word_accuracy(self):
        reference = {"a": 10, "b": 8, "c": 7}
        accuracies = {"a": 0.3, "b": 0.9, "c": 0.9}

        selected = optimize_vocabulary(reference, reference.keys(), size=2, accuracy=accuracies)

        self.assertEqual(selected, ["a", "b"])
        self.assertAlmostEqual(
            scalar_ovmi(reference, ["a", "b"], accuracy=accuracies),
            scalar_ovmi(reference, ["a", "b"], accuracy=0.6),
        )

    def test_greedy_optimisation_supports_full_method(self):
        reference = {"a": 5, "b": 4, "c": 4}
        labels = ["a", "b", "c"]
        confusion = np.array([
            [9, 1, 0],
            [1, 8, 1],
            [4, 4, 2],
        ])

        selected = optimize_vocabulary(
            reference,
            labels,
            size=2,
            method="full",
            confusion_matrix=confusion,
            labels=labels,
        )

        self.assertEqual(len(selected), 2)
        self.assertNotIn("c", selected)

    def test_greedy_range_returns_each_size(self):
        reference = {"a": 9, "b": 8, "c": 1}

        path = optimize_vocabulary_range(reference, reference.keys(), max_size=3, accuracy=0.75)

        self.assertEqual([step.size for step in path], [1, 2, 3])
        self.assertEqual(path[0].vocabulary, ["a"])
        self.assertEqual(path[-1].vocabulary, ["a", "b", "c"])

    def test_confusion_matrix_from_embeddings_uses_nearest_cosine(self):
        vocabulary = ["a", "b"]
        labels = ["a", "b", "c"]
        true_words = ["a", "a", "b", "c"]
        predicted_embeddings = np.array([
            [0.9, 0.1],
            [0.7, 0.3],
            [0.1, 0.9],
            [-0.9, 0.1],
        ])
        target_embeddings = np.array([
            [1.0, 0.0],
            [0.0, 1.0],
            [-1.0, 0.0],
        ])

        confusion = confusion_matrix_from_embeddings(
            vocabulary,
            true_words=true_words,
            predicted_embeddings=predicted_embeddings,
            target_embeddings=target_embeddings,
            candidate_labels=labels,
        )

        np.testing.assert_allclose(confusion, np.array([[2.0, 0.0], [0.0, 1.0]]))

    def test_confusion_matrix_from_logits_masks_and_softmaxes(self):
        vocabulary = ["a", "b"]
        labels = ["a", "b", "c"]
        true_words = ["a", "a", "b", "c"]
        logits = np.array([
            [2.0, 0.0, 10.0],
            [0.0, 2.0, 10.0],
            [0.0, 2.0, 10.0],
            [10.0, 0.0, 2.0],
        ])

        confusion = confusion_matrix_from_logits(
            vocabulary,
            true_words=true_words,
            logits=logits,
            candidate_labels=labels,
        )

        first = np.exp([2.0, 0.0]) / np.exp([2.0, 0.0]).sum()
        second = np.exp([0.0, 2.0]) / np.exp([0.0, 2.0]).sum()
        expected = np.vstack([first + second, second])
        np.testing.assert_allclose(confusion, expected)

    def test_embedding_search_returns_greedy_path(self):
        reference = {"a": 9, "b": 8, "c": 1}
        labels = ["a", "b", "c"]
        true_words = ["a", "a", "b", "b", "c"]
        predicted_embeddings = np.array([
            [0.9, 0.1],
            [0.8, 0.2],
            [0.1, 0.9],
            [0.2, 0.8],
            [0.8, 0.2],
        ])
        target_embeddings = np.array([
            [1.0, 0.0],
            [0.0, 1.0],
            [-1.0, 0.0],
        ])

        path = optimize_vocabulary_from_embeddings(
            reference,
            labels,
            max_size=2,
            true_words=true_words,
            predicted_embeddings=predicted_embeddings,
            target_embeddings=target_embeddings,
            candidate_labels=labels,
        )

        self.assertEqual([step.size for step in path], [1, 2])
        self.assertEqual(path[-1].vocabulary, ["a", "b"])
        self.assertGreaterEqual(path[-1].score, path[0].score)

    def test_logit_search_returns_greedy_path(self):
        reference = {"a": 9, "b": 8, "c": 1}
        labels = ["a", "b", "c"]
        true_words = ["a", "a", "b", "b", "c"]
        logits = np.array([
            [3.0, 0.0, -1.0],
            [3.0, 0.0, -1.0],
            [0.0, 3.0, -1.0],
            [0.0, 3.0, -1.0],
            [3.0, 0.0, -1.0],
        ])

        path = optimize_vocabulary_from_logits(
            reference,
            labels,
            max_size=2,
            true_words=true_words,
            logits=logits,
            candidate_labels=labels,
        )

        self.assertEqual([step.size for step in path], [1, 2])
        self.assertEqual(path[-1].vocabulary, ["a", "b"])
        self.assertGreaterEqual(path[-1].score, path[0].score)

    def test_invalid_reference_rejected(self):
        with self.assertRaises(ValueError):
            scalar_ovmi({"a": -1}, ["a"], accuracy=1)

    def test_one_word_scalar_channel_requires_perfect_accuracy(self):
        with self.assertRaises(ValueError):
            scalar_ovmi({"a": 1}, ["a"], accuracy=0.9)

        self.assertTrue(math.isclose(scalar_ovmi({"a": 1}, ["a"], accuracy=1.0), 0.0))


if __name__ == "__main__":
    unittest.main()
