"""Core OVMI computation."""

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
        mapping from words to per-word correct-decoding probabilities.
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
    """Compute OVMI under a symmetric scalar channel.

    With a scalar ``accuracy``, the channel assigns that probability to the
    correct word and distributes the remaining probability uniformly over the
    other supported words. With an accuracy mapping, each intended word gets
    its own correct-decoding probability and distributes its own error mass
    uniformly over the other supported words.
    """

    reference, vocabulary = _resolve_reference_and_vocabulary(reference, vocabulary)
    vocab, weights = _conditional_weights(reference, vocabulary)
    vocab_size = len(vocab)
    cov = coverage(reference, vocab)

    if vocab_size == 0 or cov == 0:
        return _maybe_details(0.0, cov, 0.0, 0.0, 0.0, vocab_size, return_details)

    if vocab_size == 1:
        only_accuracy = _accuracy_values_for_vocabulary(accuracy, vocab)[0]
        if only_accuracy != 1:
            raise ValueError("accuracy must be 1.0 for a one-word scalar channel.")
        return _maybe_details(0.0, cov, 0.0, 0.0, 0.0, vocab_size, return_details)

    if isinstance(accuracy, Mapping):
        channel = _scalar_channel_for_vocabulary(accuracy, vocab)
        output = weights @ channel
        output_entropy = entropy(output)
        conditional_entropy = float(weights @ _row_entropies(channel))
    else:
        # The homogeneous symmetric channel has an exact O(V) form.  Avoiding
        # the otherwise dense V x V matrix is essential for realistic
        # open-vocabulary systems (for example, V=125,000).
        correct_probability = _accuracy_values_for_vocabulary(accuracy, vocab)[0]
        error_probability = (1.0 - correct_probability) / (vocab_size - 1)
        output = error_probability + weights * (correct_probability - error_probability)
        output_entropy = entropy(output)
        conditional_entropy = _symmetric_channel_entropy(
            correct_probability,
            error_probability,
            vocab_size,
        )
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


def _accuracy_values_for_vocabulary(accuracy: AccuracyLike, vocabulary: Sequence[Hashable]) -> np.ndarray:
    if isinstance(accuracy, Mapping):
        missing = [word for word in vocabulary if word not in accuracy]
        if missing:
            raise ValueError(f"vocabulary words missing from accuracy: {missing!r}")
        values = np.array([float(accuracy[word]) for word in vocabulary], dtype=float)
        if np.any(~np.isfinite(values)):
            raise ValueError("accuracy values must be finite.")
        if np.any((values < 0) | (values > 1)):
            raise ValueError("accuracy values must lie in [0, 1].")
        return values

    value = float(accuracy)
    if not np.isfinite(value):
        raise ValueError("accuracy must be finite.")
    if value < 0 or value > 1:
        raise ValueError("accuracy must lie in [0, 1].")
    return np.full(len(vocabulary), value, dtype=float)


def _scalar_channel_for_vocabulary(accuracy: AccuracyLike, vocabulary: Sequence[Hashable]) -> np.ndarray:
    values = _accuracy_values_for_vocabulary(accuracy, vocabulary)
    vocab_size = len(vocabulary)
    error_mass = (1.0 - values) / (vocab_size - 1)
    channel = np.repeat(error_mass[:, np.newaxis], vocab_size, axis=1)
    np.fill_diagonal(channel, values)
    return channel


def _symmetric_channel_entropy(
    correct_probability: float,
    error_probability: float,
    vocabulary_size: int,
) -> float:
    """Return H(Y|X) for a homogeneous V-way symmetric channel."""

    terms = []
    if correct_probability > 0:
        terms.append(-correct_probability * np.log2(correct_probability))
    if error_probability > 0:
        terms.append(
            -(vocabulary_size - 1) * error_probability * np.log2(error_probability)
        )
    return float(sum(terms))


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


def _row_entropies(matrix: np.ndarray) -> np.ndarray:
    if matrix.ndim != 2:
        raise ValueError("matrix must be two-dimensional.")
    clipped = np.clip(matrix, 0.0, None)
    row_sums = clipped.sum(axis=1, keepdims=True)
    normalised = np.divide(clipped, row_sums, out=np.zeros_like(clipped), where=row_sums > 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.where(normalised > 0, normalised * np.log2(normalised), 0.0)
    return -terms.sum(axis=1)


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
