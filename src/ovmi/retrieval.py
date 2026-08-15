"""Deterministic sentence-retrieval metrics for decoded brain-to-text output.

The functions in this module operate on already decoded sentence hypotheses.  They
do not know how a vocabulary was selected or how token predictions were decoded;
that separation keeps retrieval a parallel readout of the existing sentence-level
evaluation rather than a new decoding task.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Hashable, Iterable, Mapping, Sequence

import numpy as np


SentenceId = Hashable
ALLOWED_SELECTION_SPLITS = frozenset({"validation", "external", "random"})


@dataclass(frozen=True)
class RetrievalScores:
    """Aggregate and per-sentence retrieval results."""

    top1: float
    top10: float
    mrr: float
    n_sentences: int
    sentence_ids: tuple[SentenceId, ...]
    ranks: np.ndarray


@dataclass(frozen=True)
class PairedTestResult:
    """Paired top-1 comparison result."""

    mean_difference: float
    ci_low: float
    ci_high: float
    p_value: float
    n_pairs: int


def assert_no_test_leakage(selection_split: str, method: str = "vocabulary") -> None:
    """Fail if test-derived information selected a retrieval vocabulary."""

    if selection_split not in ALLOWED_SELECTION_SPLITS:
        allowed = ", ".join(sorted(ALLOWED_SELECTION_SPLITS))
        raise AssertionError(
            f"{method} has selection_split={selection_split!r}; retrieval only accepts "
            f"validation-, external-, or random-derived vocabularies ({allowed})."
        )


def common_retained_sentence_ids(
    retained_by_condition: Mapping[Hashable, Iterable[SentenceId]],
) -> tuple[SentenceId, ...]:
    """Return the deterministic intersection retained by every condition."""

    if not retained_by_condition:
        return ()
    iterator = iter(retained_by_condition.values())
    common = set(next(iterator))
    for retained in iterator:
        common.intersection_update(retained)
    return tuple(sorted(common, key=repr))


def pick_vocab_size_for_coverage(
    order: Sequence[int],
    token_counts: Sequence[float],
    total_tokens: int,
    target: float,
) -> tuple[int, float]:
    """Select the shortest prefix reaching target corpus coverage.

    This is the dataset-agnostic equivalent of the Figure 4 notebook matching
    utility and deliberately preserves its capped-prefix behavior.
    """

    order = np.asarray(order, dtype=np.int64)
    token_counts = np.asarray(token_counts, dtype=np.float64)
    if len(order) == 0 or total_tokens == 0:
        return 0, 0.0
    coverage = np.cumsum(token_counts[order]) / total_tokens
    index = int(np.searchsorted(coverage, target, side="left"))
    index = min(index, len(order) - 1)
    return index + 1, float(coverage[index])


def build_candidate_pools(
    sentence_ids: Sequence[SentenceId],
    token_lengths: Mapping[SentenceId, int],
    n_candidates: int,
    *,
    seed: int = 0,
    length_matched: bool = False,
    length_tolerance: float = 0.20,
) -> dict[SentenceId, np.ndarray]:
    """Build fixed, sentence-keyed pools with unique distractors.

    Candidate index zero is always the true target. Distractors are sampled once
    without replacement, using a stable seed derived from ``seed`` and sentence id.
    Length-matched pools use targets within +/- ``length_tolerance`` of the true
    target's whitespace-token length. The function fails instead of sampling with
    replacement because duplicates would make increasing N cease to be harder.
    """

    unique_ids = tuple(dict.fromkeys(sentence_ids))
    if len(unique_ids) != len(sentence_ids):
        raise ValueError("sentence_ids must be unique")
    if n_candidates < 2:
        raise ValueError("n_candidates must be at least 2")
    if len(unique_ids) < n_candidates:
        raise ValueError(
            f"N={n_candidates} requires at least {n_candidates} unique target sentences; "
            f"only {len(unique_ids)} are available. Sampling with replacement is "
            "intentionally forbidden because it would not define a harder retrieval task."
        )
    missing_lengths = set(unique_ids).difference(token_lengths)
    if missing_lengths:
        raise KeyError(f"Missing token lengths for sentence ids: {sorted(missing_lengths, key=repr)!r}")

    pools: dict[SentenceId, np.ndarray] = {}
    for true_id in unique_ids:
        distractors = [candidate for candidate in unique_ids if candidate != true_id]
        if length_matched:
            true_length = int(token_lengths[true_id])
            lower = true_length * (1.0 - length_tolerance)
            upper = true_length * (1.0 + length_tolerance)
            distractors = [
                candidate
                for candidate in distractors
                if lower <= int(token_lengths[candidate]) <= upper
            ]
            if len(distractors) < n_candidates - 1:
                raise ValueError(
                    f"N={n_candidates} length-matched retrieval for {true_id!r} needs "
                    f"{n_candidates - 1} unique distractors within +/-{length_tolerance:.0%}, "
                    f"but only {len(distractors)} are available."
                )
        rng = np.random.default_rng(_stable_sentence_seed(seed, true_id, n_candidates))
        chosen = rng.choice(len(distractors), size=n_candidates - 1, replace=False)
        pool_values = [true_id, *(distractors[int(index)] for index in chosen)]
        pool = np.empty(n_candidates, dtype=object)
        pool[:] = pool_values
        pools[true_id] = pool
    return pools


def score_sentence_retrieval(
    hypothesis_embeddings: Mapping[SentenceId, np.ndarray],
    target_embeddings: Mapping[SentenceId, np.ndarray],
    candidate_pools: Mapping[SentenceId, Sequence[SentenceId]],
    *,
    sentence_ids: Iterable[SentenceId] | None = None,
) -> RetrievalScores:
    """Rank each unrestricted target among its paired candidate pool."""

    selected = tuple(
        sorted(
            hypothesis_embeddings if sentence_ids is None else sentence_ids,
            key=repr,
        )
    )
    ranks = np.empty(len(selected), dtype=np.int64)
    for row, sentence_id in enumerate(selected):
        if sentence_id not in hypothesis_embeddings:
            raise KeyError(f"No hypothesis embedding for retained sentence {sentence_id!r}")
        pool = tuple(candidate_pools[sentence_id])
        if not pool or pool[0] != sentence_id:
            raise ValueError(f"Candidate pool for {sentence_id!r} must put the true target first")
        missing = [candidate for candidate in pool if candidate not in target_embeddings]
        if missing:
            raise KeyError(f"Missing target embeddings for candidates: {missing!r}")

        hypothesis = _normalize_vector(hypothesis_embeddings[sentence_id])
        targets = np.stack([_normalize_vector(target_embeddings[candidate]) for candidate in pool])
        similarities = targets @ hypothesis
        true_similarity = similarities[0]
        # Optimistic ties make the perfect-string oracle exactly one even if two
        # corpus targets happen to have identical text/embeddings.
        ranks[row] = 1 + int(np.count_nonzero(similarities[1:] > true_similarity))

    if not len(ranks):
        return RetrievalScores(np.nan, np.nan, np.nan, 0, selected, ranks)
    return RetrievalScores(
        top1=float(np.mean(ranks == 1)),
        top10=float(np.mean(ranks <= 10)),
        mrr=float(np.mean(1.0 / ranks)),
        n_sentences=len(ranks),
        sentence_ids=selected,
        ranks=ranks,
    )


def paired_permutation_test_top1(
    ranks_a: Sequence[int],
    ranks_b: Sequence[int],
    *,
    n_permutations: int = 10_000,
    n_bootstrap: int = 10_000,
    seed: int = 0,
) -> PairedTestResult:
    """Paired sign-flip test and paired bootstrap CI for top-1 accuracy."""

    ranks_a = np.asarray(ranks_a, dtype=np.int64)
    ranks_b = np.asarray(ranks_b, dtype=np.int64)
    if ranks_a.shape != ranks_b.shape or ranks_a.ndim != 1:
        raise ValueError("ranks_a and ranks_b must be aligned one-dimensional arrays")
    if len(ranks_a) == 0:
        raise ValueError("At least one paired sentence is required")
    differences = (ranks_a == 1).astype(np.float64) - (ranks_b == 1).astype(np.float64)
    return paired_permutation_test_differences(
        differences,
        n_permutations=n_permutations,
        n_bootstrap=n_bootstrap,
        seed=seed,
    )


def paired_permutation_test_differences(
    differences: Sequence[float],
    *,
    n_permutations: int = 10_000,
    n_bootstrap: int = 10_000,
    seed: int = 0,
) -> PairedTestResult:
    """Sign-flip test and bootstrap CI for paired sentence-level differences."""

    differences = np.asarray(differences, dtype=np.float64)
    if differences.ndim != 1 or len(differences) == 0:
        raise ValueError("At least one paired one-dimensional difference is required")
    observed = float(np.mean(differences))
    rng = np.random.default_rng(seed)

    extreme = 0
    remaining = n_permutations
    while remaining:
        batch = min(remaining, 2_000)
        signs = rng.choice(np.array([-1.0, 1.0]), size=(batch, len(differences)))
        permuted = np.mean(signs * differences, axis=1)
        extreme += int(np.count_nonzero(np.abs(permuted) >= abs(observed)))
        remaining -= batch
    p_value = (extreme + 1.0) / (n_permutations + 1.0)

    bootstrap_indices = rng.integers(0, len(differences), size=(n_bootstrap, len(differences)))
    bootstrap_means = np.mean(differences[bootstrap_indices], axis=1)
    ci_low, ci_high = np.quantile(bootstrap_means, [0.025, 0.975])
    return PairedTestResult(
        mean_difference=observed,
        ci_low=float(ci_low),
        ci_high=float(ci_high),
        p_value=float(p_value),
        n_pairs=len(differences),
    )


def _stable_sentence_seed(
    seed: int,
    sentence_id: SentenceId,
    n_candidates: int,
) -> int:
    # Do not include the matching variant: when its eligible universe is the same
    # (as in LibriBrain, where every held-out sentence has 50 tokens), the pools
    # should be identical rather than differing only because of a control label.
    payload = f"{seed}|{sentence_id!r}|{n_candidates}".encode("utf-8")
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    return int.from_bytes(digest, "little", signed=False)


def _normalize_vector(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float64)
    if vector.ndim != 1:
        raise ValueError("Sentence embeddings must be one-dimensional")
    norm = float(np.linalg.norm(vector))
    if norm <= 0:
        raise ValueError("Sentence embeddings must have non-zero norm")
    return vector / norm
