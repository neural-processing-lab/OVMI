"""Core OVMI computation and greedy vocabulary optimisation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Hashable, Iterable, Literal, Mapping, Sequence, Union

import numpy as np

from .references import default_reference

Method = Literal["scalar", "full"]
MatrixLike = np.ndarray
AccuracyLike = Union[float, Mapping[Hashable, float]]
ReferenceLike = Union[Mapping[Hashable, float], None]


@dataclass(frozen=True)
class OVMIResult:
    """Detailed OVMI computation result."""

    score: float
    coverage: float
    in_vocab_information: float
    output_entropy: float
    conditional_entropy: float
    vocabulary_size: int


@dataclass(frozen=True)
class VocabularySearchStep:
    """One step in a greedy OVMI vocabulary search path."""

    size: int
    vocabulary: list[Hashable]
    added: Hashable
    score: float
    details: OVMIResult


def ovmi(
    reference: Mapping[Hashable, float] | Iterable[Hashable] | None = None,
    vocabulary: Iterable[Hashable] | None = None,
    *,
    method: Method = "scalar",
    accuracy: AccuracyLike | None = None,
    confusion_matrix: MatrixLike | None = None,
    labels: Sequence[Hashable] | None = None,
    return_details: bool = False,
) -> float | OVMIResult:
    """Compute OVMI for a vocabulary.

    Parameters
    ----------
    reference:
        Mapping from words to non-negative frequencies or probabilities. If
        omitted, SUBTLEX-UK is downloaded/cached and used by default. For
        convenience, passing only one positional iterable is interpreted as the
        vocabulary with the default reference.
    vocabulary:
        Supported decoder vocabulary.
    method:
        ``"scalar"`` for the homogeneous scalar approximation, or ``"full"``
        for the empirical-confusion-matrix computation from Proposition 2.
    accuracy:
        Shared correct-decoding probability for ``method="scalar"``, or a
        mapping from words to per-word correct-decoding probabilities. If a
        mapping is supplied, the scalar channel uses the macro average over the
        selected vocabulary.
    confusion_matrix:
        Empirical confusion matrix as a NumPy array for ``method="full"``.
        Rows are intended words and columns are predicted words.
    labels:
        Row/column labels for ``confusion_matrix``. If omitted, the matrix must
        already be ordered like ``vocabulary``.
    return_details:
        Return an :class:`OVMIResult` instead of only the score.
    """

    resolved_reference, resolved_vocabulary = _resolve_reference_and_vocabulary(reference, vocabulary)

    if method == "scalar":
        if accuracy is None:
            raise ValueError("accuracy is required when method='scalar'.")
        return scalar_ovmi(
            resolved_reference,
            resolved_vocabulary,
            accuracy=accuracy,
            return_details=return_details,
        )
    if method == "full":
        if confusion_matrix is None:
            raise ValueError("confusion_matrix is required when method='full'.")
        return full_ovmi(
            resolved_reference,
            resolved_vocabulary,
            confusion_matrix=confusion_matrix,
            labels=labels,
            return_details=return_details,
        )
    raise ValueError("method must be either 'scalar' or 'full'.")


def scalar_ovmi(
    reference: Mapping[Hashable, float] | Iterable[Hashable] | None = None,
    vocabulary: Iterable[Hashable] | None = None,
    *,
    accuracy: AccuracyLike,
    return_details: bool = False,
) -> float | OVMIResult:
    """Compute OVMI under the homogeneous symmetric scalar channel.

    The channel assigns probability ``accuracy`` to the correct word and
    distributes the remaining probability uniformly over the other supported
    words.
    """

    reference, vocabulary = _resolve_reference_and_vocabulary(reference, vocabulary)
    vocab, weights = _conditional_weights(reference, vocabulary)
    vocab_size = len(vocab)
    cov = coverage(reference, vocab)

    if vocab_size == 0 or cov == 0:
        return _maybe_details(0.0, cov, 0.0, 0.0, 0.0, vocab_size, return_details)

    scalar_accuracy = _accuracy_for_vocabulary(accuracy, vocab)

    if vocab_size == 1:
        if scalar_accuracy != 1:
            raise ValueError("accuracy must be 1.0 for a one-word scalar channel.")
        return _maybe_details(0.0, cov, 0.0, 0.0, 0.0, vocab_size, return_details)

    output = _symmetric_output_distribution(weights, scalar_accuracy)
    output_entropy = entropy(output)
    conditional_entropy = _symmetric_row_entropy(vocab_size, scalar_accuracy)
    in_vocab_information = output_entropy - conditional_entropy
    score = cov * in_vocab_information

    return _maybe_details(
        score,
        cov,
        in_vocab_information,
        output_entropy,
        conditional_entropy,
        vocab_size,
        return_details,
    )


def full_ovmi(
    reference: Mapping[Hashable, float] | Iterable[Hashable] | None = None,
    vocabulary: Iterable[Hashable] | None = None,
    *,
    confusion_matrix: MatrixLike,
    labels: Sequence[Hashable] | None = None,
    return_details: bool = False,
) -> float | OVMIResult:
    """Compute OVMI from an empirical in-vocabulary confusion matrix.

    Rows are intended words, columns are predicted words. Matrix entries may be
    probabilities or counts; rows are normalised internally. ``confusion_matrix``
    must be a NumPy array.
    """

    reference, vocabulary = _resolve_reference_and_vocabulary(reference, vocabulary)
    vocab, weights = _conditional_weights(reference, vocabulary)
    vocab_size = len(vocab)
    cov = coverage(reference, vocab)

    if vocab_size == 0 or cov == 0:
        return _maybe_details(0.0, cov, 0.0, 0.0, 0.0, vocab_size, return_details)

    channel = _channel_for_vocabulary(confusion_matrix, vocab, labels)
    output = weights @ channel
    output_entropy = entropy(output)
    row_entropies = _row_entropies(channel)
    conditional_entropy = float(weights @ row_entropies)
    in_vocab_information = output_entropy - conditional_entropy
    score = cov * in_vocab_information

    return _maybe_details(
        score,
        cov,
        in_vocab_information,
        output_entropy,
        conditional_entropy,
        vocab_size,
        return_details,
    )


def optimize_vocabulary(
    reference: Mapping[Hashable, float] | Iterable[Hashable] | None = None,
    candidates: Iterable[Hashable] | None = None,
    *,
    size: int,
    method: Method = "scalar",
    accuracy: AccuracyLike | None = None,
    confusion_matrix: MatrixLike | None = None,
    labels: Sequence[Hashable] | None = None,
) -> list[Hashable]:
    """Greedily select a vocabulary that maximises OVMI.

    At each step, the candidate with the largest OVMI gain is added. Ties are
    resolved by the order of ``candidates``.
    """

    reference, candidates = _resolve_reference_and_vocabulary(reference, candidates, vocabulary_name="candidates")

    if size < 0:
        raise ValueError("size must be non-negative.")

    candidate_list = _unique(candidates)
    if size > len(candidate_list):
        raise ValueError("size cannot exceed the number of unique candidates.")

    selected: list[Hashable] = []
    remaining = list(candidate_list)

    for _ in range(size):
        best_index = 0
        best_score = float("-inf")

        for idx, word in enumerate(remaining):
            candidate_vocabulary = [*selected, word]
            if method == "scalar" and len(candidate_vocabulary) == 1:
                score = 0.0
            else:
                score = ovmi(
                    reference,
                    candidate_vocabulary,
                    method=method,
                    accuracy=accuracy,
                    confusion_matrix=confusion_matrix,
                    labels=labels,
                )
            if float(score) > best_score:
                best_index = idx
                best_score = float(score)

        selected.append(remaining.pop(best_index))

    return selected


def optimize_vocabulary_range(
    reference: Mapping[Hashable, float] | Iterable[Hashable] | None = None,
    candidates: Iterable[Hashable] | None = None,
    *,
    max_size: int | None = None,
    min_size: int = 1,
    method: Method = "scalar",
    accuracy: AccuracyLike | None = None,
    confusion_matrix: MatrixLike | None = None,
    labels: Sequence[Hashable] | None = None,
) -> list[VocabularySearchStep]:
    """Greedily optimise OVMI and return the path over vocabulary sizes."""

    reference, candidates = _resolve_reference_and_vocabulary(reference, candidates, vocabulary_name="candidates")
    candidate_list = _unique(candidates)
    max_size = _resolve_max_size(max_size, candidate_list)
    _validate_size_range(min_size, max_size, len(candidate_list))

    def scorer(vocabulary: list[Hashable]) -> OVMIResult:
        return _score_vocabulary(
            reference,
            vocabulary,
            method=method,
            accuracy=accuracy,
            confusion_matrix=confusion_matrix,
            labels=labels,
        )

    return _greedy_search_path(candidate_list, min_size=min_size, max_size=max_size, scorer=scorer)


def optimize_vocabulary_from_embeddings(
    reference: Mapping[Hashable, float] | Iterable[Hashable] | None = None,
    candidates: Iterable[Hashable] | None = None,
    *,
    max_size: int | None = None,
    min_size: int = 1,
    true_words: Sequence[Hashable],
    predicted_embeddings: np.ndarray,
    target_embeddings: np.ndarray,
    candidate_labels: Sequence[Hashable] | None = None,
) -> list[VocabularySearchStep]:
    """Greedily optimise OVMI using nearest-cosine embedding predictions."""

    reference, candidates = _resolve_reference_and_vocabulary(reference, candidates, vocabulary_name="candidates")
    candidate_list = _unique(candidates)
    candidate_labels = _resolve_candidate_labels(
        candidate_labels,
        candidate_list,
        target_embeddings,
        "target_embeddings",
        label_axis=0,
    )
    max_size = _resolve_max_size(max_size, candidate_list)
    _validate_size_range(min_size, max_size, len(candidate_list))

    def scorer(vocabulary: list[Hashable]) -> OVMIResult:
        confusion = confusion_matrix_from_embeddings(
            vocabulary,
            true_words=true_words,
            predicted_embeddings=predicted_embeddings,
            target_embeddings=target_embeddings,
            candidate_labels=candidate_labels,
        )
        return full_ovmi(reference, vocabulary, confusion_matrix=confusion, return_details=True)

    return _greedy_search_path(candidate_list, min_size=min_size, max_size=max_size, scorer=scorer)


def optimize_vocabulary_from_logits(
    reference: Mapping[Hashable, float] | Iterable[Hashable] | None = None,
    candidates: Iterable[Hashable] | None = None,
    *,
    max_size: int | None = None,
    min_size: int = 1,
    true_words: Sequence[Hashable],
    logits: np.ndarray,
    candidate_labels: Sequence[Hashable] | None = None,
) -> list[VocabularySearchStep]:
    """Greedily optimise OVMI using masked-softmax logit distributions."""

    reference, candidates = _resolve_reference_and_vocabulary(reference, candidates, vocabulary_name="candidates")
    candidate_list = _unique(candidates)
    candidate_labels = _resolve_candidate_labels(candidate_labels, candidate_list, logits, "logits", label_axis=1)
    max_size = _resolve_max_size(max_size, candidate_list)
    _validate_size_range(min_size, max_size, len(candidate_list))

    def scorer(vocabulary: list[Hashable]) -> OVMIResult:
        confusion = confusion_matrix_from_logits(
            vocabulary,
            true_words=true_words,
            logits=logits,
            candidate_labels=candidate_labels,
        )
        return full_ovmi(reference, vocabulary, confusion_matrix=confusion, return_details=True)

    return _greedy_search_path(candidate_list, min_size=min_size, max_size=max_size, scorer=scorer)


def confusion_matrix_from_embeddings(
    vocabulary: Iterable[Hashable],
    *,
    true_words: Sequence[Hashable],
    predicted_embeddings: np.ndarray,
    target_embeddings: np.ndarray,
    candidate_labels: Sequence[Hashable] | None = None,
) -> np.ndarray:
    """Build an empirical confusion matrix from nearest-cosine embeddings."""

    vocab = _unique(vocabulary)
    predicted = _as_2d_array("predicted_embeddings", predicted_embeddings)
    targets = _as_2d_array("target_embeddings", target_embeddings)
    labels = _resolve_candidate_labels(candidate_labels, vocab, targets, "target_embeddings", label_axis=0)

    if predicted.shape[0] != len(true_words):
        raise ValueError("true_words length must match predicted_embeddings rows.")
    if predicted.shape[1] != targets.shape[1]:
        raise ValueError("predicted_embeddings and target_embeddings must have the same width.")

    _validate_nonzero_rows("predicted_embeddings", predicted)
    _validate_nonzero_rows("target_embeddings", targets)

    positions = _positions(labels)
    missing = [word for word in vocab if word not in positions]
    if missing:
        raise ValueError(f"vocabulary words missing from candidate_labels: {missing!r}")

    selected_indices = [positions[word] for word in vocab]
    selected_targets = targets[selected_indices]
    similarities = _normalise_rows(predicted) @ _normalise_rows(selected_targets).T
    predicted_indices = np.argmax(similarities, axis=1)

    vocab_positions = _positions(vocab)
    confusion = np.zeros((len(vocab), len(vocab)), dtype=float)
    for true_word, predicted_index in zip(true_words, predicted_indices):
        row = vocab_positions.get(true_word)
        if row is not None:
            confusion[row, predicted_index] += 1.0
    return confusion


def confusion_matrix_from_logits(
    vocabulary: Iterable[Hashable],
    *,
    true_words: Sequence[Hashable],
    logits: np.ndarray,
    candidate_labels: Sequence[Hashable] | None = None,
) -> np.ndarray:
    """Build an expected confusion matrix from masked-softmax logits."""

    vocab = _unique(vocabulary)
    logit_array = _as_2d_logits(logits)
    labels = _resolve_candidate_labels(candidate_labels, vocab, logit_array, "logits", label_axis=1)

    if logit_array.shape[0] != len(true_words):
        raise ValueError("true_words length must match logits rows.")

    positions = _positions(labels)
    missing = [word for word in vocab if word not in positions]
    if missing:
        raise ValueError(f"vocabulary words missing from candidate_labels: {missing!r}")

    selected_indices = [positions[word] for word in vocab]
    selected_logits = logit_array[:, selected_indices]

    vocab_positions = _positions(vocab)
    selected_rows: list[int] = []
    selected_sample_logits: list[np.ndarray] = []
    for sample_logits, true_word in zip(selected_logits, true_words):
        row = vocab_positions.get(true_word)
        if row is None:
            continue
        selected_rows.append(row)
        selected_sample_logits.append(sample_logits)

    confusion = np.zeros((len(vocab), len(vocab)), dtype=float)
    if not selected_sample_logits:
        return confusion

    sample_logits = np.vstack(selected_sample_logits)
    probabilities = _softmax_rows(sample_logits)
    for row, probability in zip(selected_rows, probabilities):
        confusion[row] += probability
    return confusion


def coverage(
    reference: Mapping[Hashable, float] | Iterable[Hashable] | None = None,
    vocabulary: Iterable[Hashable] | None = None,
) -> float:
    """Return lexical coverage C(S) under the reference distribution."""

    reference, vocabulary = _resolve_reference_and_vocabulary(reference, vocabulary)
    probs = _normalised_reference(reference)
    vocab = set(vocabulary)
    return float(sum(probs.get(word, 0.0) for word in vocab))


def entropy(probabilities: np.ndarray) -> float:
    """Shannon entropy in bits."""

    if not isinstance(probabilities, np.ndarray):
        raise TypeError("probabilities must be a numpy.ndarray.")

    probs = probabilities.astype(float, copy=False)
    if probs.ndim != 1:
        raise ValueError("probabilities must be one-dimensional.")
    if np.any(~np.isfinite(probs)):
        raise ValueError("probabilities must be finite.")
    if np.any(probs < -1e-12):
        raise ValueError("probabilities must be non-negative.")

    probs = np.clip(probs, 0.0, None)
    total = probs.sum()
    if total <= 0:
        return 0.0
    probs = probs / total
    positive = probs[probs > 0]
    return float(-(positive * np.log2(positive)).sum())


def _conditional_weights(
    reference: ReferenceLike,
    vocabulary: Iterable[Hashable],
) -> tuple[list[Hashable], np.ndarray]:
    vocab = _unique(vocabulary)
    probs = _normalised_reference(reference)
    weights = np.array([probs.get(word, 0.0) for word in vocab], dtype=float)
    total = weights.sum()
    if total > 0:
        weights = weights / total
    return vocab, weights


def _normalised_reference(reference: ReferenceLike) -> dict[Hashable, float]:
    if reference is None:
        reference = default_reference()

    if not reference:
        raise ValueError("reference must contain at least one entry.")

    probs = {word: float(weight) for word, weight in reference.items()}
    values = np.array(list(probs.values()), dtype=float)

    if np.any(~np.isfinite(values)):
        raise ValueError("reference weights must be finite.")
    if np.any(values < 0):
        raise ValueError("reference weights must be non-negative.")

    total = float(values.sum())
    if total <= 0:
        raise ValueError("reference weights must have positive total mass.")

    return {word: weight / total for word, weight in probs.items()}


def _resolve_reference_and_vocabulary(
    reference: Mapping[Hashable, float] | Iterable[Hashable] | None,
    vocabulary: Iterable[Hashable] | None,
    *,
    vocabulary_name: str = "vocabulary",
) -> tuple[ReferenceLike, Iterable[Hashable]]:
    if vocabulary is not None:
        return _as_reference(reference), vocabulary

    if reference is None:
        raise ValueError(f"{vocabulary_name} is required.")
    if isinstance(reference, Mapping):
        raise ValueError(
            f"{vocabulary_name} is required. To use the default SUBTLEX-UK reference, "
            f"pass {vocabulary_name} as the first positional argument."
        )
    return None, reference


def _as_reference(reference: Mapping[Hashable, float] | Iterable[Hashable] | None) -> ReferenceLike:
    if reference is None:
        return None
    if not isinstance(reference, Mapping):
        raise TypeError("reference must be a mapping, None, or omitted.")
    return reference


def _score_vocabulary(
    reference: ReferenceLike,
    vocabulary: list[Hashable],
    *,
    method: Method,
    accuracy: AccuracyLike | None,
    confusion_matrix: MatrixLike | None,
    labels: Sequence[Hashable] | None,
) -> OVMIResult:
    if method == "scalar":
        if accuracy is None:
            raise ValueError("accuracy is required when method='scalar'.")
        if len(vocabulary) == 1:
            cov = coverage(reference, vocabulary)
            return OVMIResult(
                score=0.0,
                coverage=cov,
                in_vocab_information=0.0,
                output_entropy=0.0,
                conditional_entropy=0.0,
                vocabulary_size=1,
            )
        result = scalar_ovmi(reference, vocabulary, accuracy=accuracy, return_details=True)
    elif method == "full":
        if confusion_matrix is None:
            raise ValueError("confusion_matrix is required when method='full'.")
        result = full_ovmi(
            reference,
            vocabulary,
            confusion_matrix=confusion_matrix,
            labels=labels,
            return_details=True,
        )
    else:
        raise ValueError("method must be either 'scalar' or 'full'.")
    return result


def _greedy_search_path(
    candidates: list[Hashable],
    *,
    min_size: int,
    max_size: int,
    scorer,
) -> list[VocabularySearchStep]:
    selected: list[Hashable] = []
    remaining = list(candidates)
    path: list[VocabularySearchStep] = []

    for size in range(1, max_size + 1):
        best_index = 0
        best_details: OVMIResult | None = None

        for index, word in enumerate(remaining):
            details = scorer([*selected, word])
            if best_details is None or details.score > best_details.score:
                best_index = index
                best_details = details

        added = remaining.pop(best_index)
        selected.append(added)

        if size >= min_size:
            if best_details is None:
                raise RuntimeError("greedy search failed to score any candidate.")
            path.append(
                VocabularySearchStep(
                    size=size,
                    vocabulary=list(selected),
                    added=added,
                    score=best_details.score,
                    details=best_details,
                )
            )

    return path


def _resolve_max_size(max_size: int | None, candidates: Sequence[Hashable]) -> int:
    if max_size is None:
        return len(candidates)
    return max_size


def _validate_size_range(min_size: int, max_size: int, candidate_count: int) -> None:
    if min_size < 1:
        raise ValueError("min_size must be at least 1.")
    if max_size < min_size:
        raise ValueError("max_size must be greater than or equal to min_size.")
    if max_size > candidate_count:
        raise ValueError("max_size cannot exceed the number of unique candidates.")


def _resolve_candidate_labels(
    candidate_labels: Sequence[Hashable] | None,
    candidates: Sequence[Hashable],
    array: np.ndarray,
    array_name: str,
    *,
    label_axis: int,
) -> list[Hashable]:
    if not isinstance(array, np.ndarray):
        raise TypeError(f"{array_name} must be a numpy.ndarray.")
    if array.ndim != 2:
        raise ValueError(f"{array_name} must be two-dimensional.")

    labels = list(candidates if candidate_labels is None else candidate_labels)
    if len(labels) != array.shape[label_axis]:
        axis_name = "rows" if label_axis == 0 else "columns"
        raise ValueError(f"candidate_labels length must match {array_name} {axis_name}.")
    _positions(labels)
    return labels


def _as_2d_array(name: str, array: np.ndarray) -> np.ndarray:
    if not isinstance(array, np.ndarray):
        raise TypeError(f"{name} must be a numpy.ndarray.")
    values = array.astype(float, copy=False)
    if values.ndim != 2:
        raise ValueError(f"{name} must be two-dimensional.")
    if np.any(~np.isfinite(values)):
        raise ValueError(f"{name} entries must be finite.")
    return values


def _as_2d_logits(logits: np.ndarray) -> np.ndarray:
    if not isinstance(logits, np.ndarray):
        raise TypeError("logits must be a numpy.ndarray.")
    values = logits.astype(float, copy=False)
    if values.ndim != 2:
        raise ValueError("logits must be two-dimensional.")
    if np.any(np.isnan(values)) or np.any(values == np.inf):
        raise ValueError("logits must not contain NaN or positive infinity.")
    return values


def _validate_nonzero_rows(name: str, array: np.ndarray) -> None:
    if np.any(np.linalg.norm(array, axis=1) <= 0):
        raise ValueError(f"{name} rows must have non-zero norm.")


def _normalise_rows(array: np.ndarray) -> np.ndarray:
    return array / np.linalg.norm(array, axis=1, keepdims=True)


def _softmax_rows(logits: np.ndarray) -> np.ndarray:
    if np.any(np.all(np.isneginf(logits), axis=1)):
        raise ValueError("each selected logit row must contain at least one finite value.")
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(shifted)
    exp = np.where(np.isfinite(logits), exp, 0.0)
    return exp / exp.sum(axis=1, keepdims=True)


def _accuracy_for_vocabulary(accuracy: AccuracyLike, vocabulary: Sequence[Hashable]) -> float:
    if isinstance(accuracy, Mapping):
        missing = [word for word in vocabulary if word not in accuracy]
        if missing:
            raise ValueError(f"vocabulary words missing from accuracy: {missing!r}")
        values = np.array([float(accuracy[word]) for word in vocabulary], dtype=float)
        if np.any(~np.isfinite(values)):
            raise ValueError("accuracy values must be finite.")
        if np.any((values < 0) | (values > 1)):
            raise ValueError("accuracy values must lie in [0, 1].")
        return float(values.mean())

    value = float(accuracy)
    if not np.isfinite(value):
        raise ValueError("accuracy must be finite.")
    if value < 0 or value > 1:
        raise ValueError("accuracy must lie in [0, 1].")
    return value


def _channel_for_vocabulary(
    confusion_matrix: MatrixLike,
    vocabulary: Sequence[Hashable],
    labels: Sequence[Hashable] | None,
) -> np.ndarray:
    if not isinstance(confusion_matrix, np.ndarray):
        raise TypeError("confusion_matrix must be a numpy.ndarray.")

    matrix = confusion_matrix.astype(float, copy=False)

    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or matrix.shape[0] == 0:
        raise ValueError("confusion_matrix must be square.")
    if np.any(~np.isfinite(matrix)):
        raise ValueError("confusion_matrix entries must be finite.")
    if np.any(matrix < 0):
        raise ValueError("confusion_matrix entries must be non-negative.")

    if labels is None:
        if matrix.shape != (len(vocabulary), len(vocabulary)):
            raise ValueError(
                "without labels, confusion_matrix must have shape "
                "(len(vocabulary), len(vocabulary))."
            )
        submatrix = matrix
    else:
        if len(labels) != matrix.shape[0]:
            raise ValueError("labels length must match confusion_matrix dimensions.")
        positions = _positions(labels)
        missing = [word for word in vocabulary if word not in positions]
        if missing:
            raise ValueError(f"vocabulary words missing from labels: {missing!r}")
        indices = [positions[word] for word in vocabulary]
        submatrix = matrix[np.ix_(indices, indices)]

    row_sums = submatrix.sum(axis=1, keepdims=True)
    if np.any(row_sums <= 0):
        raise ValueError("each selected confusion_matrix row must have positive mass.")
    return submatrix / row_sums


def _symmetric_output_distribution(weights: np.ndarray, accuracy: float) -> np.ndarray:
    vocab_size = weights.shape[0]
    error_mass = (1.0 - accuracy) / (vocab_size - 1)
    return accuracy * weights + error_mass * (1.0 - weights)


def _row_entropies(matrix: np.ndarray) -> np.ndarray:
    if matrix.ndim != 2:
        raise ValueError("matrix must be two-dimensional.")
    clipped = np.clip(matrix, 0.0, None)
    row_sums = clipped.sum(axis=1, keepdims=True)
    normalised = np.divide(clipped, row_sums, out=np.zeros_like(clipped), where=row_sums > 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.where(normalised > 0, normalised * np.log2(normalised), 0.0)
    return -terms.sum(axis=1)


def _symmetric_row_entropy(vocab_size: int, accuracy: float) -> float:
    row = np.full(vocab_size, (1.0 - accuracy) / (vocab_size - 1), dtype=float)
    row[0] = accuracy
    return entropy(row)


def _maybe_details(
    score: float,
    cov: float,
    in_vocab_information: float,
    output_entropy: float,
    conditional_entropy: float,
    vocabulary_size: int,
    return_details: bool,
) -> float | OVMIResult:
    score = _clean_float(score)
    if not return_details:
        return score
    return OVMIResult(
        score=score,
        coverage=_clean_float(cov),
        in_vocab_information=_clean_float(in_vocab_information),
        output_entropy=_clean_float(output_entropy),
        conditional_entropy=_clean_float(conditional_entropy),
        vocabulary_size=vocabulary_size,
    )


def _unique(items: Iterable[Hashable]) -> list[Hashable]:
    seen: set[Hashable] = set()
    unique_items: list[Hashable] = []
    for item in items:
        if item not in seen:
            unique_items.append(item)
            seen.add(item)
    return unique_items


def _positions(labels: Sequence[Hashable]) -> dict[Hashable, int]:
    positions: dict[Hashable, int] = {}
    for index, label in enumerate(labels):
        if label in positions:
            raise ValueError(f"duplicate label in labels: {label!r}")
        positions[label] = index
    return positions


def _clean_float(value: float) -> float:
    if abs(value) < 1e-12:
        return 0.0
    return float(value)
