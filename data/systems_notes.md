# System-coordinate notes

## Reference distribution and shared conventions

All coordinates use the SUBTLEX-UK `Spelling`/`FreqCount` distribution from
van Heuven et al. (2014). Words are stripped, Unicode-normalised with NFKC,
case-folded, and curly apostrophes are mapped to ASCII apostrophes; frequencies
that collide after normalisation are summed. The resulting reference has
160,020 types and entropy **H(p) = 9.77172891764 bits**. The source workbook is
the repository's `experiments/data/cache/SUBTLEX-UK.xlsx`. See
[van Heuven et al. (2014)](https://doi.org/10.1080/17470218.2013.850521).

`scripts/build_systems_csv.py` rebuilds every row in `systems.csv`. For each
fixed vocabulary it calls the repository OVMI implementation under the
homogeneous symmetric channel. `I_invocab_*` is the conditional mutual
information before multiplication by coverage; `OVMI_*` is exactly
`coverage * I_invocab_*`; `H_pS_bits` is the entropy ceiling computed from
each system's own restricted and renormalised vocabulary distribution. The
default figure plots both `P_neural` and
`P_system` where both exist; LM-assisted coordinates are identified in their
direct text labels without a marker overlay. Isolated classification and
continuous +LM measurements are labelled separately and deliberately left
unconnected because they are not matched LM ablations. The two Willett-50
markers share the same numerical coverage but receive a symmetric five-point
horizontal display dodge so both remain visible. `--neural-only` suppresses
the LM-assisted coordinates.

The plotted probabilities are not all the same experimental object. Isolated
classification accuracy is the closest match to the symmetric channel. A
WER-derived `1 - WER` mixes substitutions, deletions, insertions, alignment,
and language-model effects. When no error breakdown is available it is stored
as a **lower bound** on token success because insertions add WER without a
corresponding failed intended token. This qualification is retained in the
caption and provenance rather than encoded with arrows.

The figure's thin dashed curve is not a universal ceiling. It is a noiseless
frequency-selection frontier formed by sweeping `V`, choosing the `V` most
frequent SUBTLEX-UK words, and plotting `(C_V, H(p_{S_V}))`; the final point is
`(1, H(p))`. Bespoke vocabularies such as the Moses/Willett caregiving set may
lie above this curve at the same coverage because their renormalised
in-vocabulary distribution can have greater entropy. Sanity checks therefore
compare each information coordinate with its own row's `H_pS_bits`, never with
the frequency-selection frontier.

## MEG-MASC 2023, V=50

- Row: `meg_masc_2023_v50`. The dataset source is [Gwilliams et al.,
  *Scientific Data* 10, 862 (2023)](https://doi.org/10.1038/s41597-023-02752-5),
  with the data archived at [OSF](https://doi.org/10.17605/OSF.IO/AG3KJ).
- The supplied seed-level top-1 results are **P=0.093, 0.083, and 0.080** for
  the 50-way classifier. Their mean, **P=0.085333**, is plotted with mean plus
  or minus one SEM across the three training seeds. These are not results
  reported in the dataset paper, and the SEM does not include test-set sampling.
- The vocabulary is rebuilt from the four supplied pre-tokenized stories:
  `cable_spool_fort.txt`, `easy_money.txt`, `lw1.txt`, and
  `the_black_willow.txt`. The builder NFKC-normalises and case-folds whitespace
  tokens, excludes standalone punctuation, preserves corpus clitic forms such
  as `'s` and `n't`, ranks by pooled frequency, and resolves ties
  alphabetically. The exact ranked words and counts are emitted to
  `data/vocabularies/meg_masc_2023_v50.csv`; SHA-256 hashes of all four source
  texts are checked in `scripts/build_systems_csv.py`.
- The top-50 cutoff is `one` with 25 occurrences. The derived vocabulary gives
  `C=0.397030`, `H(p_S)=4.921387`, `I_invocab=0.083742`, and
  `OVMI=0.033248` bits under SUBTLEX-UK. Mapping the seed-SEM endpoints through
  OVMI gives `I_invocab=[0.075708, 0.092036]` bits.
- No external language model is used, so only `P_neural` is populated. The
  point is perceived speech recorded with non-invasive MEG.

## LibriBrain100 2025, d'Ascoli method on subject 0, V=50

- Row: `dascoli_libribrain100_s0_v50`.
- This is a local result obtained by training the method of [d'Ascoli et al.,
  Nature Communications 16, 10521
  (2025)](https://doi.org/10.1038/s41467-025-65499-0) on **LibriBrain100 subject
  0** and evaluating the Sherlock test set. It replaces the paper's
  device/task-unspecified discussion-level aggregate.
- The three seed-level top-1 balanced accuracies are **25.6%, 25.6%, and
  26.2%**, giving `P_neural=0.258`. Seed variation is reported as mean plus or
  minus one SEM (`0.256–0.260`) and propagated through the OVMI channel. It is
  not a confidence interval and excludes test-set sampling. No per-word results
  were archived for the requested nested bootstrap.
- The exact 50-word vocabulary is embedded in `scripts/build_systems_csv.py`:
  `is the a to it i not was we be he that have this they of there and are in
  but will so all my for she were any really at out our am its had him an very
  has do can time think good always new people as on`.
- Its direct SUBTLEX-UK coverage is `C=0.389913`, with
  `H(p_S)=4.981781`, `I_invocab=0.609016`, and `OVMI=0.237463` bits.

## Armeni et al. (2022), V=50

- Row: `armeni_2022_v50`.
- Dataset source: [Armeni et al., Scientific Data 9, 278
  (2022)](https://doi.org/10.1038/s41597-022-01382-7). The participants listened
  to *The Adventures of Sherlock Holmes* during non-invasive MEG recording; the
  paper identifies the supplied [plain-text story
  source](https://sherlock-holm.es/stories/plain-text/advs.txt).
- The vocabulary is rebuilt rather than manually transcribed. The builder
  downloads that complete plaintext, normalises it with NFKC, case-folds,
  preserves internal apostrophes, extracts alphabetic word tokens, counts them,
  and breaks frequency ties alphabetically. The resulting top 50 are:
  `the and i to of a in that it you he was his is my have as had with which at
  for but not me be we there from this said upon holmes so him her she very
  your been all no what on one then were by are an`. The cutoff token is `an`
  with 333 occurrences; the next token is `would` with 327.
- The three supplied seed-level top-1 balanced accuracies are **21.1%, 21.0%,
  and 20.2%**, giving `P_neural=0.207667`. Seed variation is reported as mean
  plus or minus one SEM (`0.204819–0.210515`) and propagated through OVMI. It is
  not a confidence interval and excludes test-set sampling. No per-word results
  were archived for the requested nested bootstrap.
- This vocabulary has direct SUBTLEX-UK coverage `C=0.407597`, with
  `H(p_S)=4.995528`, `I_invocab=0.430752`, and `OVMI=0.175573` bits.

## Brain2Qwerty v2 (Zhang, Levy et al., 2026)

- Row: `brain2qwerty_v2`.
- Source: the most recent version found, [*Accurate Decoding of Natural
  Sentences from Non-Invasive Brain Recordings*, v2
  (2026)](https://facebookresearch.github.io/brain2qwerty/assets/brain2qwerty_v2.pdf),
  abstract and Discussion. It reports mean WER 39% (best participant 22%) for
  delayed typing by healthy participants after language-model decoding.
- `P_system=0.61` records the mechanical lower bound `1-0.39` for provenance,
  but coverage and information are NaN and the row is not plotted. The task is
  typing rather than speech and the open sentence generator has no fixed word
  class vocabulary S, so forcing it onto this plane would manufacture a
  coordinate.

## Moses et al. (2021), 50 words

- Row: `moses_2021_v50`.
- Source: [Moses et al., NEJM 385, 217–227
  (2021)](https://doi.org/10.1056/NEJMoa2027540), Results/Word Detection and
  Classification, pp. 222–223, reports **47.1%** isolated-word accuracy over
  9,000 attempts; Results, p. 221, reports median sentence-decoding WER
  **25.6%** with the language model.
- The same sentence experiment reports **60.5% WER without language
  modelling**. The plotted neural point nevertheless uses the requested 47.1%
  isolated-word classification accuracy because it is the paper's direct
  fixed-class neural result and best matches the symmetric channel. Thus the
  two points are left unconnected; using the matched sentence WERs instead
  would give lower-bound probabilities 0.395 and 0.744.
- The published 50-word set is embedded in the data-building script. Its direct
  SUBTLEX coverage is `0.172671`. `P_neural=0.471`; `P_system=0.744` is the
  lower-bound conversion `1-WER`. The LM-assisted coordinate is
  (`C=0.172671`, `I_invocab=2.659020` bits), OVMI `0.459135` bits. The
  neural-only coordinate has `I_invocab=1.376335` and OVMI `0.237653` bits.
- A 95% Wilson interval on 9,000 isolated attempts is stored for `P_neural`
  (`0.460702–0.481323`). The LM-assisted row uses the paper's reported 95%
  interval for median WER across 15 ten-sentence blocks, mapped to
  `P_system=0.629–0.829`. Both are propagated through the OVMI channel.
- Stretch: the default point combines sentence-level LM/WER performance with
  the same 50-word vocabulary used for isolated classification; it is not a
  purely neural 50-way classifier coordinate.

## Willett et al. (2023), 50 words

- Row: `willett_2023_v50`.
- Source: [Willett et al., Nature 620, 1031–1036
  (2023)](https://doi.org/10.1038/s41586-023-06377-x), Fig. 1d and Results,
  p. 1032, reports **94%** 50-way isolated-word accuracy (20 trials per word),
  not the approximate 95.5% in the task prompt. Table 1, p. 1034, reports
  **9.1%** WER with the 50-word language model.
- The paper reuses the Moses 50-word vocabulary, giving direct SUBTLEX coverage
  `0.172671`. `P_neural=0.94`; `P_system=0.909` is `1-WER` and is a lower bound.
  LM-assisted `I_invocab=3.596815` bits and OVMI `0.621066` bits; neural-only
  `I_invocab=3.799108` and OVMI `0.655996` bits.
- The stored neural 95% Wilson interval (`0.923529–0.953104`) uses 1,000 trials
  and is propagated for the isolated row. The LM-assisted row uses the paper's
  95% percentile interval from 10,000 bootstrap resamples over 250 sentence
  trials (`P_system=0.888–0.928`).

## Willett et al. (2023), 125,000 words

- Row: `willett_2023_v125k`.
- Source: [Willett et al. (2023)](https://doi.org/10.1038/s41586-023-06377-x),
  Table 1, p. 1034, reports **23.8% WER** for the 125,000-word vocabulary.
- The exact decoder lexicon was not recovered as a machine-readable supplement.
  Coverage therefore uses the current upstream CMU Pronouncing Dictionary,
  the lexicon family named by the paper, after removal of pronunciation-variant
  suffixes. This proxy has 126,052 unique spellings; 66,987 intersect the
  normalised SUBTLEX types and account for `C=0.978779`. The nominal CSV V is
  kept at the reported 125,000. This is a high-coverage approximation, not an
  assertion that the two word lists are identical.
- No substitution/deletion/insertion breakdown suitable for token success was
  located, so `P_system=0.762=1-WER` is marked lower-bound and `P_neural` is
  NaN. This gives `I_invocab=7.190494` bits (0.746 of its own `H(p_S)`) and OVMI
  `7.037907` bits. The y-coordinate is below the entropy ceiling because 23.8%
  WER is substantial; the coverage itself is correctly near one.
- The paper reports pre-LM **phoneme** error rate (19.7% for vocal speech), not
  a pre-LM word success probability. It is therefore not placed on this
  word-level plane.
- The paper's 95% percentile interval from 10,000 bootstrap resamples over 400
  sentence trials is retained (`P_system=0.741–0.782`); words are not treated as
  independent binomial trials.

## Card et al. (2024), 125,000 words

- Row: `card_2024_v125k`.
- Source: [Card et al., NEJM 391, 609–618
  (2024)](https://doi.org/10.1056/NEJMoa2314132), Results/Online Decoding
  Performance, reports **2.5% WER** (95% CI 2.0–3.1%) over the final five Copy
  Task sessions with the 125,000-word vocabulary.
- The same explicitly labelled CMUdict proxy is used, hence `C=0.978779`.
  `P_system=0.975=1-WER` is a lower bound; `P_neural` is unavailable. The
  coordinate has `I_invocab=9.354130` bits (0.971 of its own `H(p_S)`) and OVMI
  `9.155629` bits.
- The published 95% WER interval is mapped directly to
  `P_system=0.969–0.980`. It was computed from 10,000 bootstrap resamples over
  individual sentence trials pooled across the final five evaluation sessions;
  the exact sentence count for that aggregate is not reported. Stretch: this is continuous, LM-assisted
  alignment performance rather than fixed-class isolated-word accuracy.
- Card likewise exposes raw phoneme-decoder performance, but no comparable
  pre-LM word probability suitable for `P_neural`; only the +LM point is shown.

## Missing values and exclusions

No NaN is filled silently. Brain2Qwerty lacks a comparable speech task and
fixed S; the 125k systems lack word-level pre-LM `P_neural`. The plotter prints
every excluded row and the reason when it runs.
