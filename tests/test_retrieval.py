import unittest

import numpy as np

from ovmi.retrieval import (
    assert_no_test_leakage,
    build_candidate_pools,
    common_retained_sentence_ids,
    paired_permutation_test_top1,
    score_sentence_retrieval,
)


class RetrievalTests(unittest.TestCase):
    def test_perfect_oracle_is_top1_one_and_shuffled_is_near_chance(self):
        n_sentences = 5_000
        n_candidates = 20
        rng = np.random.default_rng(7)
        sentence_ids = tuple(range(n_sentences))
        target_matrix = rng.normal(size=(n_sentences, 32))
        target_matrix /= np.linalg.norm(target_matrix, axis=1, keepdims=True)
        targets = {sentence_id: target_matrix[sentence_id] for sentence_id in sentence_ids}
        lengths = {sentence_id: 20 for sentence_id in sentence_ids}
        pools = build_candidate_pools(sentence_ids, lengths, n_candidates, seed=13)

        oracle = score_sentence_retrieval(targets, targets, pools)
        self.assertEqual(oracle.top1, 1.0)

        permutation = rng.permutation(n_sentences)
        shuffled = {
            sentence_id: target_matrix[int(permutation[sentence_id])]
            for sentence_id in sentence_ids
        }
        shuffled_scores = score_sentence_retrieval(shuffled, targets, pools)
        chance = 1.0 / n_candidates
        self.assertLess(abs(shuffled_scores.top1 - chance), 0.015)

    def test_candidate_pools_are_deterministic_and_paired(self):
        sentence_ids = tuple(range(30))
        lengths = {sentence_id: 10 for sentence_id in sentence_ids}
        first = build_candidate_pools(sentence_ids, lengths, 20, seed=42)
        second = build_candidate_pools(sentence_ids, lengths, 20, seed=42)

        for sentence_id in sentence_ids:
            np.testing.assert_array_equal(first[sentence_id], second[sentence_id])
            self.assertEqual(first[sentence_id][0], sentence_id)
            self.assertEqual(len(set(first[sentence_id].tolist())), 20)

    def test_tuple_sentence_ids_remain_atomic(self):
        sentence_ids = tuple(("subject", index) for index in range(20))
        lengths = {sentence_id: 10 for sentence_id in sentence_ids}
        pools = build_candidate_pools(sentence_ids, lengths, 20, seed=42)
        self.assertEqual(pools[sentence_ids[0]].shape, (20,))
        self.assertEqual(pools[sentence_ids[0]][0], sentence_ids[0])

    def test_candidate_pool_rejects_insufficient_unique_targets(self):
        with self.assertRaisesRegex(ValueError, "unique target sentences"):
            build_candidate_pools(tuple(range(19)), {i: 10 for i in range(19)}, 20)

    def test_length_matched_pool_stays_within_twenty_percent(self):
        sentence_ids = tuple(range(30))
        lengths = {sentence_id: 10 + sentence_id % 2 for sentence_id in sentence_ids}
        pools = build_candidate_pools(
            sentence_ids,
            lengths,
            20,
            seed=1,
            length_matched=True,
        )

        for sentence_id, pool in pools.items():
            true_length = lengths[sentence_id]
            for distractor in pool[1:]:
                self.assertLessEqual(abs(lengths[distractor] - true_length), true_length * 0.20)

    def test_common_subset_intersects_every_condition(self):
        retained = {"a": {1, 2, 3}, "b": {2, 3, 4}, "c": {3, 4}}
        self.assertEqual(common_retained_sentence_ids(retained), (3,))

    def test_test_derived_vocabulary_fails_loudly(self):
        assert_no_test_leakage("validation", "greedy")
        assert_no_test_leakage("external", "frequency")
        assert_no_test_leakage("random", "random")
        with self.assertRaisesRegex(AssertionError, "selection_split='test'"):
            assert_no_test_leakage("test", "leaky")

    def test_paired_statistics_report_observed_difference(self):
        result = paired_permutation_test_top1(
            [1, 1, 2, 2],
            [2, 1, 2, 2],
            n_permutations=100,
            n_bootstrap=100,
            seed=3,
        )
        self.assertEqual(result.mean_difference, 0.25)
        self.assertEqual(result.n_pairs, 4)
        self.assertGreaterEqual(result.p_value, 0.0)
        self.assertLessEqual(result.p_value, 1.0)


if __name__ == "__main__":
    unittest.main()
