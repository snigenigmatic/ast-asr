# Draft request for authoritative Svarah speaker metadata

**Status:** draft only. Do not send from this repository without the team's
review. It requests only the missing release metadata required to reproduce
speaker-disjoint evaluation.

## Email draft

**To:** Svarah dataset contact listed in the official dataset documentation  
**Subject:** Request for original Svarah utterance-to-speaker metadata

Dear Svarah maintainers,

We are reproducing and extending Indian-English ASR experiments using the
public Svarah release. We found that the currently accessible Hugging Face
Parquet revisions contain the demographic fields but not the `speaker_id`
column described in the project documentation.

Could you please share, or point us to an official source for, the original
`meta_speaker_stats.csv` and the corresponding utterance-to-speaker mapping for
the released 6,656 utterances? An archive containing the documented
`svarah_manifest.json` and metadata file would also be sufficient, provided the
mapping can be matched to the public utterances.

We will use the material only to construct speaker-disjoint folds for research
evaluation. Please let us know any license, citation, access, or redistribution
restrictions that apply. We will record the source revision and checksum and
will not redistribute files that are not authorized for redistribution.

Thank you for maintaining the dataset and for any guidance you can provide.

Best regards,  
[Name]  
[Institution / project contact]

## Receipt and provenance checklist

Do not open the authoritative-data gate until every applicable item is saved in
an immutable receipt record next to the received files.

| Check | Required receipt |
| --- | --- |
| Official origin | Maintainer reply preserved, or official source URL/repository/release location recorded; include sender/date or retrieval date. |
| Scope confirmation | Written confirmation that the mapping is for the Svarah release being used, or a documented relation between the supplied archive and release. |
| File identity | SHA-256 for each received metadata, manifest, or archive file; preserve original filenames. |
| Versioning | Release name, tag, commit/revision, or dated maintainer statement; record retrieval date in ISO-8601 form. |
| Row alignment | Demonstrate a one-to-one alignment to all 6,656 released utterances using utterance IDs or the documented manifest alignment; retain the validation report. |
| Speaker cardinality | `speaker_id` is non-empty for every mapped row and has exactly 117 distinct IDs; save the program output. |
| Structural validation | Required metadata columns, duplicate/empty-ID checks, transcript/duration cross-checks, and `ast-asr prepare-data` output are retained. |
| Fold validation | Five fold manifests prove no speaker overlap and exactly one out-of-fold test appearance per speaker. |
| Restrictions | Preserve license/citation/access terms and any redistribution prohibition from the reply or official source. Store restricted files only in approved private storage. |
| Publication boundary | Mark the data gate passed only after all previous checks pass; do not mix new results with 115 profile-cluster development artifacts. |

## Minimal receipt record fields

```text
received_at:
official_source_url_or_reply_reference:
contact_or_maintainer:
release_or_revision:
files:
  - original_name:
    local_storage_reference:
    sha256:
restrictions:
alignment:
  expected_utterances: 6656
  matched_utterances:
  method:
speaker_ids:
  non_empty: true
  distinct_count: 117
validation_artifacts:
  - authoritative_metadata_audit.json
  - dataset_manifest.json
  - fold_validation.json
```

The supporting audit is [speaker-identity-audit.md](speaker-identity-audit.md).
