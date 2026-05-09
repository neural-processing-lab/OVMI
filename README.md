# OVMI

`ovmi` computes open-vocabulary mutual information and greedily optimises vocabularies for OVMI.

```python
from ovmi import ovmi, optimize_vocabulary

reference = {
    "yes": 120,
    "no": 80,
    "pain": 30,
    "water": 20,
    "music": 10,
}

score = ovmi(reference, ["yes", "no", "water"], accuracy=0.47)

vocab = optimize_vocabulary(
    reference,
    candidates=reference.keys(),
    size=3,
    accuracy=0.47,
)
```

If no reference distribution is provided, `ovmi` downloads and caches the SUBTLEX-UK frequency norm from OSF, then uses its `Spelling` and `FreqCount` columns:

```python
score = ovmi(["yes", "no", "water"], accuracy=0.47)

vocab = optimize_vocabulary(
    candidates=["yes", "no", "pain", "water", "music"],
    size=3,
    accuracy=0.47,
)
```

You can also load it directly:

```python
from ovmi import load_subtlex_uk

reference = load_subtlex_uk()
```

The default method is the homogeneous scalar approximation:

```python
score = ovmi(reference, ["yes", "no", "water"], method="scalar", accuracy=0.47)
```

For optimisation, `accuracy` may also be a per-word mapping; the scalar approximation uses the macro average for each candidate vocabulary:

```python
accuracies = {"yes": 0.70, "no": 0.65, "pain": 0.50, "water": 0.55, "music": 0.20}
vocab = optimize_vocabulary(reference, reference.keys(), size=3, accuracy=accuracies)
```

For full OVMI from Proposition 2, pass a NumPy empirical confusion matrix whose rows are intended words and columns are predicted words:

```python
import numpy as np
from ovmi import full_ovmi

labels = ["yes", "no", "water"]
confusion = np.array([
    [18, 1, 1],
    [2, 15, 3],
    [1, 4, 10],
])

score = full_ovmi(reference, labels, confusion_matrix=confusion, labels=labels)
```

Vocabulary optimisation uses the same methods:

```python
vocab = optimize_vocabulary(reference, reference.keys(), size=3, method="scalar", accuracy=0.47)
vocab = optimize_vocabulary(reference, reference.keys(), size=3, method="full", confusion_matrix=confusion, labels=labels)
```

For a proper greedy search over vocabulary sizes, use the range API. It returns one step for each vocabulary size:

```python
from ovmi import optimize_vocabulary_range

path = optimize_vocabulary_range(
    reference,
    candidates=reference.keys(),
    max_size=3,
    accuracy=0.47,
)

best = max(path, key=lambda step: step.score)
```

When model outputs are available, OVMI can rebuild the induced decoder at each greedy step.

For embedding models, pass predicted embeddings for examples, target embeddings for candidate words, and the true word for each example. Each candidate vocabulary is scored by nearest cosine similarity within that vocabulary:

```python
import numpy as np
from ovmi import optimize_vocabulary_from_embeddings

words = ["yes", "no", "pain", "water"]
true_words = ["yes", "no", "pain", "water"]
predicted_embeddings = np.array([
    [0.9, 0.1],
    [0.1, 0.9],
    [-0.9, 0.0],
    [0.7, 0.3],
])
target_embeddings = np.array([
    [1.0, 0.0],
    [0.0, 1.0],
    [-1.0, 0.0],
    [0.8, 0.2],
])

path = optimize_vocabulary_from_embeddings(
    reference,
    candidates=words,
    max_size=4,
    true_words=true_words,
    predicted_embeddings=predicted_embeddings,
    target_embeddings=target_embeddings,
    candidate_labels=words,
)
```

For logit models, pass logits over candidate words. At each step, OVMI masks the logits to the candidate vocabulary and renormalises with softmax:

```python
from ovmi import optimize_vocabulary_from_logits

logits = np.array([
    [3.0, 0.1, -1.0, 2.0],
    [0.2, 3.0, -1.0, 0.0],
    [0.0, 0.1, 3.0, -1.0],
    [2.0, 0.0, -1.0, 2.8],
])

path = optimize_vocabulary_from_logits(
    reference,
    candidates=words,
    max_size=4,
    true_words=true_words,
    logits=logits,
    candidate_labels=words,
)
```
