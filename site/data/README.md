# Explorer data

`leaderboard.json` is the generated comparison data file, produced by:

```bash
python scripts/build_site_data.py
```

The file is generated from `data/systems.csv`, the reference CSVs under
`data/references/`, the Tang decoder vocabulary at
`data/vocabularies/tang_decoder_vocab.json`, and the scoring pipeline in
`scripts/make_main_table.py`.

## Schema

- `references`: reference label, description, entropy, and source.
- `systems`: one row per reported decoder operating point. A single paper can
  contribute separate isolated-decoder or language-model-assisted rows. Some
  study-level rows aggregate participant results with an explicit SEM.
- `systems[].decoder_method`: optional decoder provenance shown for rows using
  a named decoder from another study (for example MEG-XL or d’Ascoli et al.).
- `systems[].metric`: the reported metric and the scalar `p_correct` used by the
  historical symmetric-channel estimate. `p_is_lower_bound` marks WER-derived
  values, where `P = 1 - WER` is conservative.
- `systems[].references`: coverage, in-vocabulary information, OVMI, normalised
  OVMI, and uncertainty endpoints for every communication reference.
- `top_frequency_curves`: the optional noiseless frequency-selected curve. It
  currently exists only for SUBTLEX-UK and is not described as a universal
  frontier.

To add a study, update the canonical analysis data and vocabulary
reconstruction first, regenerate the paper tables, then regenerate this JSON.
