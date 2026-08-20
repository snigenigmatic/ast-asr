# SPIRE-SIES cross-corpus generalization protocol

Status: **locked for preparation; evaluation not yet authorized.**

Classification: cross-corpus generalization evidence on a **real speaker-disjoint
split**. Unlike every Svarah result in this repository, the grouping unit here is an
actual corpus-provided speaker, not a demographic profile cluster.

## First-principles decision

Every Svarah result so far carries `publication_valid: false` for one reason: the
authoritative 117 Svarah speaker identities are not obtainable, so folds use 115
demographic *profile clusters*. A profile cluster is not a speaker, so no
speaker-disjoint fairness claim is defensible from Svarah alone.

SPIRE-SIES removes that specific blocker. It ships **real speaker IDs** and a
**ready-made speaker-disjoint split** (1126 train / 198 val, zero overlap). It
therefore supports a legitimate speaker-level generalization analysis without
depending on an external metadata release.

This protocol does **not** rescue or reinterpret H6. H6 failed its safety gate and
stays failed. This is an independent axis: *does anything we learned on Svarah
transfer to a different Indian-English corpus, measured on real speakers?*

## Frozen contract

| Setting | Value |
| --- | --- |
| Corpus | `VectorSigma389/spire-sies` (private), 17 per-language Parquet shards |
| Split source | repository `splits.json` **and** the per-row `split` column; both must agree |
| Evaluation speakers | the 198 `val` speakers only |
| Corpus role | **evaluation only — never trained on** |
| Audio | `audio_bytes` decoded, mono, 48 kHz source resampled to 16 kHz |
| Duration filter | 1.0 s to 30.0 s inclusive |
| Reference text | corpus `reference` verbatim; no CTC/uppercase rewriting |
| WER normalizer | `ast_asr.metrics.normalize_for_wer` (repo standard, for comparability) |
| Edit counts | `ast_asr.metrics.word_edit_counts` (concatenated substitutions/deletions/insertions) |
| Family taxonomy | `ast_asr.taxonomy.SVARAH_LANGUAGE_FAMILIES`, cross-checked against the corpus `language_family` column; disagreement or unknown language fails closed |
| Families present | Dravidian and Indo-Aryan only (**no Sino-Tibetan**) |
| Condition | clean speech only in this protocol; noise conditions are a separate registered extension |
| Decoding | greedy, FP32, `ast_asr.inference.greedy_transcribe` |
| Cluster unit | **speaker** (real), 198 clusters |

## Known constraints (state these, do not hide them)

1. **Two families only.** The cross-corpus family contrast is 2-way; Svarah's
   primary worst-group endpoint is 3-way. The two are therefore *not* the same
   endpoint and must never be compared as if they were.
2. **No age labels.** Age is `Unknown` corpus-wide; no age axis exists here.
   Gender is present and usable.
3. **Clean read speech.** This is image-caption read speech, so it is a clean
   source. Any robustness statement requires added synthetic noise and is out of
   scope for this protocol.
4. **Long per-language tail.** Some languages have very few utterances (e.g.
   Sindhi 13). Analysis pools to **family**; per-language numbers are descriptive
   only and carry no hypothesis test.

## Registered endpoints

For each evaluated checkpoint, report over the 198-speaker val split:

- overall WER (clean);
- WER per language family (2 cells) and the **worst-family WER**;
- WER per gender;
- worst-20%-speaker WER.

## Registered comparison and uncertainty

Checkpoints are compared **pairwise on identical utterances**. For each of 10,000
resamples, draw 198 speaker IDs with replacement once and apply those identical
multiplicities to every arm. Recompute per-arm pooled cell WERs, per-arm worst
family, and the within-pair delta. Report percentile intervals descriptively.

Because these are real speakers, the interval is a genuine speaker-clustered
interval. It is still a single-corpus observation and does not become a general
fairness claim.

## Forbidden

- Training on SPIRE, or using it to select any hyperparameter.
- Comparing the SPIRE 2-family worst-group number to the Svarah 3-family
  worst-group number as if they were the same endpoint.
- Presenting per-language cells as tested groups.
- Reusing `spire_loader.normalize_transcript` (uppercase CTC normalizer from the
  abandoned wav2vec2 pipeline) for scoring.
- Downloading or materializing any part of this corpus onto a local machine. All
  preparation and evaluation runs on Modal into a Modal volume.

## Implementation note (why this lives in `scripts/`)

The H7 recovery authorization freezes the SHA-256 manifests of `src/` and
`configs/` and permits no source change beyond launcher coordinates. To avoid
invalidating that preflight, this workstream adds **no files to `src/` or
`configs/`**. Pure logic lives in `scripts/spire_crosscorpus.py` and imports
`ast_asr` read-only. After H7 r1 concludes, this may be refactored into `src/`.
