# OVMI

`ovmi` computes open-vocabulary mutual information for a fixed vocabulary.

The default method is the homogeneous scalar approximation:

```python
from ovmi import ovmi

reference = {
    "yes": 120,
    "no": 80,
    "pain": 30,
    "water": 20,
    "music": 10,
}

score = ovmi(reference, ["yes", "no", "water"], accuracy=0.47)
```

If no reference distribution is provided, `ovmi` downloads and caches the
SUBTLEX-UK frequency norm from OSF, then uses its `Spelling` and `FreqCount`
columns:

```python
score = ovmi(["yes", "no", "water"], accuracy=0.47)
```

You can also load it directly:

```python
from ovmi import load_subtlex_uk

reference = load_subtlex_uk()
```

## Per-Word Accuracies

Pass an accuracy mapping when each intended word has its own correct-decoding
probability. Each row distributes its remaining error mass uniformly over the
other words in the selected vocabulary:

```python
accuracies = {
    "yes": 0.70,
    "no": 0.65,
    "water": 0.55,
}

score = ovmi(reference, ["yes", "no", "water"], accuracy=accuracies)
```

## Full Empirical Confusion Matrix

For full OVMI from an empirical confusion matrix, pass a NumPy array whose rows
are intended words and columns are predicted words. Matrix entries may be counts
or probabilities; rows are normalised internally.

```python
import numpy as np
from ovmi import ovmi, full_ovmi

labels = ["yes", "no", "water"]
confusion = np.array([
    [18, 1, 1],
    [2, 15, 3],
    [1, 4, 10],
])

score = ovmi(
    reference,
    ["yes", "no", "water"],
    method="full",
    confusion_matrix=confusion,
    labels=labels,
)

same_score = full_ovmi(reference, labels, confusion_matrix=confusion, labels=labels)
```

## Details

Set `return_details=True` to get the component terms alongside the OVMI score:

```python
details = ovmi(reference, ["yes", "no", "water"], accuracy=0.47, return_details=True)

print(details.score)
print(details.coverage)
print(details.in_vocab_information)
```
