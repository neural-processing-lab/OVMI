# Confound-table source notes

This file is the cell-level audit trail for `data/confounds.csv`. The generator
requires one hidden `trace` tag for each displayed confound cell. Page numbers
below are printed page numbers where available; for arXiv manuscripts they are
PDF pages. “Not reported” means that the cited paper and supplement were
checked for the requested quantity but did not state it.

**Inference policy. No populated entry is inferred.** Values mechanically
constructed for the local Table 1 evaluation (rather than quoted from a paper)
are explicitly identified as local constructions. No duration is calculated
from trial counts, and no recording time is treated as calibration time. `N/A`
is only an applicability label for sentence leakage in an isolated-word result.

## Card et al. (2024), 125k +LM

Paper: Card et al., *An Accurate and Rapidly Calibrating Speech
Neuroprosthesis*, NEJM 391:609–618,
[doi:10.1056/NEJMoa2314132](https://doi.org/10.1056/NEJMoa2314132).

- **Recording modality:** four intracortical microelectrode arrays. Methods,
  “Surgical Implantation,” p. 611. <!-- trace: card_2024_v125k:system.recording_modality -->
- **Speech type:** attempted speech. Abstract, p. 609, and Methods, “Speech Task
  Designs,” p. 613. <!-- trace: card_2024_v125k:system.speech_type -->
- **Participants:** one participant with ALS enrolled in the BrainGate clinical
  study. Abstract and “Participants,” pp. 609–611. <!-- trace: card_2024_v125k:system.participants -->
- **Hours per participant:** Figure 2 gives 15.9, 17.6, 18.8, 80.9, and 94.5
  cumulative hours for the final five displayed evaluation sessions; the table
  reports their stated range rather than averaging them. Results, “Online
  Speech-Decoding Performance,” and Fig. 2, pp. 614–615.
  <!-- trace: card_2024_v125k:system.hours_per_participant -->
- **Task structure:** real-time, continuous sentence decoding in copy and
  conversation tasks. Methods, “Speech Task Designs,” p. 613.
  <!-- trace: card_2024_v125k:system.task_structure -->
- **Vocabulary provenance:** conversation output was limited to a 125,000-word
  dictionary intended to cover English, not a study-specific fixed list.
  Methods, “Speech Task Designs,” p. 613. <!-- trace: card_2024_v125k:system.vocabulary_provenance -->
- **Language model:** a 5-gram model followed by transformer rescoring; the
  displayed word-error result is after that pipeline. Fig. 1 and “Decoding
  Speech,” pp. 612–613; Supplement, §S3. <!-- trace: card_2024_v125k:system.language_model -->
- **Calibration:** 0.5 h of initial data, then 1.4 h additional data before the
  first 125k evaluation (1.9 h cumulative); later requested updates used 20
  sentences in about 7.5 min. Results, pp. 614–616, Fig. 2.
  <!-- trace: card_2024_v125k:system.calibration_burden -->
- **Split discipline:** training blocks and predetermined evaluation blocks are
  distinguished, but exact sentence overlap between those blocks and the LM
  corpus is not reported. Methods, “Evaluation,” p. 614; hence `—`.
  <!-- trace: card_2024_v125k:system.split_discipline -->

## Willett et al. (2023)

Paper: Willett et al., *A High-Performance Speech Neuroprosthesis*, Nature
620:1031–1036,
[doi:10.1038/s41586-023-06377-x](https://doi.org/10.1038/s41586-023-06377-x).

### Shared acquisition cells

- **Recording modality:** four intracortical microelectrode arrays. Abstract and
  opening Results, pp. 1031–1032.
  <!-- trace: willett_2023_v125k:system.recording_modality -->
  <!-- trace: willett_2023_v50:neural.recording_modality -->
  <!-- trace: willett_2023_v50:system.recording_modality -->
- **Speech type:** attempted speech, with the main Table 1 values using vocalized
  attempted speech. “Decoding attempted speech,” pp. 1032–1034.
  <!-- trace: willett_2023_v125k:system.speech_type -->
  <!-- trace: willett_2023_v50:neural.speech_type -->
  <!-- trace: willett_2023_v50:system.speech_type -->
- **Participants:** one BrainGate2 participant with bulbar-onset ALS who could
  not speak intelligibly. p. 1032 and Reporting Summary p. 16.
  <!-- trace: willett_2023_v125k:system.participants -->
  <!-- trace: willett_2023_v50:neural.participants -->
  <!-- trace: willett_2023_v50:system.participants -->

### Willett 125k +LM

- **Hours per participant:** Card et al. (2024), Discussion p. 617, explicitly
  summarizes this predecessor as 16.8 h over 15 days to reach the quoted 23.8%
  WER. This is a stated cross-paper value, not a conversion from 10,850
  sentences. <!-- trace: willett_2023_v125k:system.hours_per_participant -->
- **Task structure:** real-time whole-sentence decoding. “Decoding attempted
  speech,” pp. 1032–1033. <!-- trace: willett_2023_v125k:system.task_structure -->
- **Vocabulary provenance:** the large model uses a 125,000-word vocabulary for
  general English. pp. 1032–1033. <!-- trace: willett_2023_v125k:system.vocabulary_provenance -->
- **Language model:** custom 125,000-word trigram in Kaldi; the reported WER is
  for the combined RNN+LM pipeline. Fig. 2 caption and Table 1, pp. 1033–1034.
  <!-- trace: willett_2023_v125k:system.language_model -->
- **Calibration:** data collection and RNN training lasted 140 min per
  evaluation day on average, including breaks. p. 1032.
  <!-- trace: willett_2023_v125k:system.calibration_burden -->
- **Split discipline:** RNN evaluation sentences were held out and never
  duplicated in neural training. The paper does not report exact overlap with
  the external LM corpus, which remains `—` within the cell. p. 1032.
  <!-- trace: willett_2023_v125k:system.split_discipline -->

### Willett isolated 50

- **Hours per participant:** the instructed-delay experiment reports 20 trials
  per word but not recording hours. Fig. 1 caption, p. 1032; hence `—`.
  <!-- trace: willett_2023_v50:neural.hours_per_participant -->
- **Task structure:** cross-validated 50-way naive-Bayes classification of
  single prompted words. p. 1032 and Extended Data Fig. 3, p. 9.
  <!-- trace: willett_2023_v50:neural.task_structure -->
- **Vocabulary provenance:** the 50-word set is reused from Moses et al.; that
  source documents participant and caregiving criteria (Moses Supplement §S3,
  pp. 8–9). Willett pp. 1032–1033.
  <!-- trace: willett_2023_v50:neural.vocabulary_provenance -->
- **Language model:** none in the isolated naive-Bayes classification result.
  p. 1032. <!-- trace: willett_2023_v50:neural.language_model -->
- **Calibration:** time to usable performance is not reported for this isolated
  classifier; hence `—`. p. 1032 and Extended Data Fig. 3.
  <!-- trace: willett_2023_v50:neural.calibration_burden -->
- **Split discipline:** sentence leakage is not applicable to an isolated-word
  result; this is an applicability label, not an inferred experimental fact.
  p. 1032. <!-- trace: willett_2023_v50:neural.split_discipline -->

### Willett 50 +LM

- **Hours per participant:** the sentence decoder shares the accumulated RNN
  training data summarized by Card et al. (2024), Discussion p. 617: 16.8 h
  over 15 days. <!-- trace: willett_2023_v50:system.hours_per_participant -->
- **Task structure:** real-time continuous sentence decoding. pp. 1032–1033.
  <!-- trace: willett_2023_v50:system.task_structure -->
- **Vocabulary provenance:** the paper explicitly reuses the Moses word set and
  test sentences. p. 1033. <!-- trace: willett_2023_v50:system.vocabulary_provenance -->
- **Language model:** the WER includes a 50-word LM, but the main paper does not
  state its n-gram/neural type; the cell therefore retains `type —` rather than
  borrowing the 125k model's trigram label. pp. 1032–1034.
  <!-- trace: willett_2023_v50:system.language_model -->
- **Calibration:** 140 min per evaluation day on average for data collection and
  RNN training, including breaks. p. 1032.
  <!-- trace: willett_2023_v50:system.calibration_burden -->
- **Split discipline:** neural evaluation sentences were held out and never
  duplicated in RNN training; exact overlap with the small LM corpus is not
  reported. pp. 1032–1033. <!-- trace: willett_2023_v50:system.split_discipline -->

## Moses et al. (2021)

Paper: Moses et al., *Neuroprosthesis for Decoding Speech in a Paralyzed
Person with Anarthria*, NEJM 385:217–227,
[doi:10.1056/NEJMoa2027540](https://doi.org/10.1056/NEJMoa2027540), and its
Supplementary Appendix.

### Shared acquisition cells

- **Recording modality:** subdural high-density ECoG over speech sensorimotor
  cortex. Main Methods, pp. 219–220.
  <!-- trace: moses_2021_v50:system.recording_modality -->
  <!-- trace: moses_2021_v50:neural.recording_modality -->
- **Speech type:** attempted speech. Main Abstract and Methods, pp. 217–220.
  <!-- trace: moses_2021_v50:system.speech_type -->
  <!-- trace: moses_2021_v50:neural.speech_type -->
- **Participants:** one clinical participant with anarthria and spastic
  quadriparesis after brain-stem stroke. Main Abstract and “Participant,”
  pp. 217–220. <!-- trace: moses_2021_v50:system.participants -->
  <!-- trace: moses_2021_v50:neural.participants -->
- **Hours per participant:** 22 h 30 min of isolated-word data across 48
  sessions. Supplement §S6, p. 16.
  <!-- trace: moses_2021_v50:system.hours_per_participant -->
  <!-- trace: moses_2021_v50:neural.hours_per_participant -->
- **Vocabulary provenance:** the participant helped select the list; criteria
  included sentence utility and basic caregiving needs. Supplement §S3,
  pp. 8–9. <!-- trace: moses_2021_v50:system.vocabulary_provenance -->
  <!-- trace: moses_2021_v50:neural.vocabulary_provenance -->
- **Calibration:** recording burden is reported, but time from setup to usable
  performance is not; hence `—`. Main pp. 220–223; Supplement §§S3,S6.
  <!-- trace: moses_2021_v50:system.calibration_burden -->
  <!-- trace: moses_2021_v50:neural.calibration_burden -->

### Moses 50 +LM

- **Task structure:** sentences were decoded in real time from separately
  detected word attempts, rather than uninterrupted phoneme streaming.
  Supplement §S3, pp. 9–10.
  <!-- trace: moses_2021_v50:system.task_structure -->
- **Language model:** fifth-order interpolated Kneser–Ney n-gram; the reported
  sentence WER includes Viterbi/LM rescoring. Supplement §S10, pp. 31–35;
  main Fig. 2 and Results, pp. 221–223.
  <!-- trace: moses_2021_v50:system.language_model -->
- **Split discipline:** the 50 evaluation sentences were sampled from the
  Mechanical Turk corpus (§S5, pp. 14–15), and that same corpus trained the LM
  (§S10, p. 31). Exact evaluation sentences can therefore occur in LM training:
  this is stated provenance, not an inferred overlap rate.
  <!-- trace: moses_2021_v50:system.split_discipline -->

### Moses isolated 50

- **Task structure:** cued isolated-word attempts and 50-way classification.
  Supplement §S3, pp. 8–9. <!-- trace: moses_2021_v50:neural.task_structure -->
- **Language model:** none in the reported isolated-word classification.
  Main “Word Detection and Classification,” pp. 222–223.
  <!-- trace: moses_2021_v50:neural.language_model -->
- **Split discipline:** sentence leakage is not applicable; the relevant
  trial/block discipline was disjoint 10-fold cross-validation with no
  train/test overlap in an assessment. Supplement §S6, p. 16.
  <!-- trace: moses_2021_v50:neural.split_discipline -->

## LibriBrain100 2025 local row

Sources: Özdogan et al., *LibriBrain: Over 50 Hours of Within-Subject MEG to
Improve Speech Decoding Methods at Scale*, NeurIPS 2025 Datasets and
Benchmarks, §§3 and B, pp. 3–4 and 17–18; d'Ascoli et al., *Towards Decoding
Individual Words from Non-invasive Brain Recordings*, Nature Communications
16:10521 (2025), Methods “Splitting,” pp. 8–9; and the local evaluation record
in `data/systems_notes.md`.

- **Recording modality:** MEG, LibriBrain §3.1, p. 3.
  <!-- trace: dascoli_libribrain100_s0_v50:neural.recording_modality -->
- **Speech type:** audiobook listening (perceived speech), §3.1, p. 3.
  <!-- trace: dascoli_libribrain100_s0_v50:neural.speech_type -->
- **Participants:** one healthy volunteer, §3.1, p. 3.
  <!-- trace: dascoli_libribrain100_s0_v50:neural.participants -->
- **Hours per participant:** Table 1 reports 52.32 h; Appendix Table 6 reports
  53:02:41 including held-out material. The CSV uses the paper's comparison
  table value, 52.32 h. pp. 2 and 17.
  <!-- trace: dascoli_libribrain100_s0_v50:neural.hours_per_participant -->
- **Task structure:** contextual, word-locked classification follows the
  d'Ascoli method; d'Ascoli “Methods/Training” and “Splitting,” pp. 7–9. The
  displayed balanced-accuracy value itself is a local result documented in
  `data/systems_notes.md`, not a d'Ascoli paper result.
  <!-- trace: dascoli_libribrain100_s0_v50:neural.task_structure -->
- **Vocabulary provenance:** the exact 50-word Sherlock evaluation vocabulary
  is the documented local Table 1 construction in `data/systems_notes.md`; it
  is frequency-selected from the evaluation corpus rather than claimed as a
  paper quotation. <!-- trace: dascoli_libribrain100_s0_v50:neural.vocabulary_provenance -->
- **Language model:** nearest-neighbour word-embedding classification without
  sentence-generation rescoring; d'Ascoli “Evaluation metrics” and
  “Implementation details,” pp. 7–8.
  <!-- trace: dascoli_libribrain100_s0_v50:neural.language_model -->
- **Calibration:** no time to usable participant-specific performance is
  reported; hence `—`. LibriBrain baseline §4 and d'Ascoli “Training.”
  <!-- trace: dascoli_libribrain100_s0_v50:neural.calibration_burden -->
- **Split discipline:** whole validation/test sessions and story chapters were
  acquired separately and held out to control leakage. LibriBrain Appendix B,
  pp. 17–18. <!-- trace: dascoli_libribrain100_s0_v50:neural.split_discipline -->

## Tang 2023 continuous decoder

Paper: Tang et al., *Semantic reconstruction of continuous language from
non-invasive brain recordings*, *Nature Neuroscience* 26, 858--866 (2023),
[doi:10.1038/s41593-023-01304-9](https://doi.org/10.1038/s41593-023-01304-9).

- **Recording modality:** BOLD fMRI. Figure 1a and Methods.
  <!-- trace: tang_2023_v6867:system.recording_modality -->
- **Speech type:** perceived narrative speech. Figure 1a--c.
  <!-- trace: tang_2023_v6867:system.speech_type -->
- **Participants:** the main decoding analyses use three healthy subjects.
  Methods, “Subjects.” <!-- trace: tang_2023_v6867:system.participants -->
- **Hours per participant:** each subject listened to 16 h of narrative stories
  for training. Figure 1a. <!-- trace: tang_2023_v6867:system.hours_per_participant -->
- **Task structure:** continuous language reconstruction from novel fMRI
  recordings. Abstract and Figure 1b--c.
  <!-- trace: tang_2023_v6867:system.task_structure -->
- **Vocabulary provenance:** 6,867 unique words occurring at least twice in the
  encoding-model training dataset. Methods, “Language model.”
  <!-- trace: tang_2023_v6867:system.vocabulary_provenance -->
- **Language model:** an autoregressive language model proposes candidate
  continuations and the encoding model scores them during beam search. Figure
  1b and Methods, “Language decoder.”
  <!-- trace: tang_2023_v6867:system.language_model -->
- **Calibration:** time to usable performance is not reported; hence `—`.
  <!-- trace: tang_2023_v6867:system.calibration_burden -->
- **Split discipline:** perceived-speech test stories were not used for model
  training. Figure 1c. <!-- trace: tang_2023_v6867:system.split_discipline -->

## Armeni 2022 local row

Paper: Armeni et al., *A 10-hour Within-participant Magnetoencephalography
Narrative Dataset to Test Models of Language Comprehension*, Scientific Data
9:278 (2022), [doi:10.1038/s41597-022-01382-7](https://doi.org/10.1038/s41597-022-01382-7),
plus the local evaluation record in `data/systems_notes.md`.

- **Recording modality:** MEG. Methods “MEG data acquisition,” p. 5.
  <!-- trace: armeni_2022_v50:neural.recording_modality -->
- **Speech type:** attentive audiobook listening. “Task and experimental
  design,” pp. 3–4. <!-- trace: armeni_2022_v50:neural.speech_type -->
- **Participants:** three participants reporting no neurological,
  developmental, or language deficits. Methods “Participants,” p. 2.
  <!-- trace: armeni_2022_v50:neural.participants -->
- **Hours per participant:** ten hour-long sessions per participant. Abstract
  p. 1 and Methods pp. 2–3. <!-- trace: armeni_2022_v50:neural.hours_per_participant -->
- **Task structure:** the paper supplies continuous narrative recordings; the
  displayed 50-way isolated-word classification is a local evaluation recorded
  in `data/systems_notes.md`, not a result in the dataset paper.
  <!-- trace: armeni_2022_v50:neural.task_structure -->
- **Vocabulary provenance:** the local evaluation takes the top 50 normalized
  word tokens from the complete Sherlock plaintext identified under “Stimulus
  materials,” p. 2; exact construction is in `data/systems_notes.md`.
  <!-- trace: armeni_2022_v50:neural.vocabulary_provenance -->
- **Language model:** no external LM rescoring is used in the local
  classification result; `data/systems_notes.md`, “Armeni et al. (2022), V=50.”
  <!-- trace: armeni_2022_v50:neural.language_model -->
- **Calibration:** time to usable performance is not reported; hence `—`.
  Dataset paper pp. 1–8 and local record.
  <!-- trace: armeni_2022_v50:neural.calibration_burden -->
- **Split discipline:** the dataset paper defines sessions and runs but not the
  local classifier's train/test sentence split; hence `—`. “Task and
  experimental design,” pp. 3–4.
  <!-- trace: armeni_2022_v50:neural.split_discipline -->

## MEG-MASC 2023 local V=50 result

Dataset paper: Gwilliams et al., *Introducing MEG-MASC: a high-quality
magneto-encephalography dataset for evaluating natural speech processing*,
*Scientific Data* 10, 862 (2023),
[doi:10.1038/s41597-023-02752-5](https://doi.org/10.1038/s41597-023-02752-5),
plus the local evaluation record in `data/systems_notes.md`.

- **Recording modality:** whole-head MEG; Methods, “MEG acquisition and
  preprocessing.” <!-- trace: meg_masc_2023_v50:neural.recording_modality -->
- **Speech type:** perceived natural speech while listening to fictional
  stories; abstract and “Stimuli.”
  <!-- trace: meg_masc_2023_v50:neural.speech_type -->
- **Participants:** 27 healthy English-speaking adults; abstract and
  “Participants.” <!-- trace: meg_masc_2023_v50:neural.participants -->
- **Hours per participant:** the dataset contains approximately two hours of
  story listening per participant; abstract.
  <!-- trace: meg_masc_2023_v50:neural.hours_per_participant -->
- **Task structure:** the paper supplies naturalistic story-listening data; the
  displayed 50-way word classification is the supplied local evaluation, not a
  result from the dataset paper.
  <!-- trace: meg_masc_2023_v50:neural.task_structure -->
- **Vocabulary provenance:** the local evaluation uses the top 50 pooled
  frequency tokens from the four supplied pre-tokenized story texts; exact
  construction and the emitted vocabulary are documented in
  `data/systems_notes.md`.
  <!-- trace: meg_masc_2023_v50:neural.vocabulary_provenance -->
- **Language model:** no external LM rescoring is used in the supplied local
  top-1 result. <!-- trace: meg_masc_2023_v50:neural.language_model -->
- **Calibration:** time to usable classifier performance was not supplied;
  hence `—`. <!-- trace: meg_masc_2023_v50:neural.calibration_burden -->
- **Split discipline:** train/test sentence or story separation for the local
  classifier was not supplied; hence `—`.
  <!-- trace: meg_masc_2023_v50:neural.split_discipline -->

## Sparsity interpretation

Run `python3 scripts/make_confound_table.py` for current counts. Missingness is
counted only from literal em-dash cells, not from `N/A` cells. The most sparse
column should be quoted in the reporting-checklist text as direct evidence of
what the literature does not currently report consistently.
