# Reference distributions for the main table

Run `python scripts/build_reference_distributions.py` from the repository root
to regenerate the five `word,weight` CSV files. The table generator normalises
these positive weights internally.

- `subtlex_uk.csv`: SUBTLEX-UK `Spelling`/`FreqCount`, with NFKC,
  case-folding, and typographic-apostrophe normalisation. Source: van Heuven et
  al. (2014), <https://doi.org/10.1080/17470218.2013.850521>; the source
  workbook is downloaded from <https://osf.io/d3jbg/download> by the OVMI
  reference loader.
- `switchboard_conversational.csv`: lexical token counts from the tagged
  transcript in the official 36-call NLTK Switchboard Corpus Sample. Common
  auxiliary contractions are expanded, possessive markers, punctuation, and
  partial-word tokens are excluded. Source archive and fixed SHA-256 digest are
  recorded in `scripts/build_reference_distributions.py`.
- `ucv_aac.csv`: the 36 Universal Core Vocabulary words used in the paper's
  exploratory notebook, weighted by their SUBTLEX-UK counts.
- `sherlock_libribrain_test.csv`: empirical counts of the 3,550 target-word
  tokens in the identical MEG-XL LibriBrain test split stored in each of the
  five local prediction runs (1,137 word types).
- `individual_target_moses.csv`: Moses et al.'s published 50-word caregiving
  vocabulary, weighted by SUBTLEX-UK counts. This is the fixed individual
  target reference in the main table.
- `bnc_spoken.csv`: the context-governed and demographic spoken subsets of the
  British National Corpus, summed across part-of-speech entries. Common
  auxiliary contractions are expanded; punctuation, multi-word entries, and
  partial-word tokens are excluded. The published Kilgarriff frequency lists,
  URLs, and fixed SHA-256 digests are recorded in the builder. This reference
  is used only as the second estimator of broad spoken English in the
  reference-dependence figure.

The five main-table entropies are 9.7717, 8.2683, 4.3046, 8.4438, and 4.2741
bits, respectively. The builder prints the BNC-spoken entropy used by the
robustness axis.
