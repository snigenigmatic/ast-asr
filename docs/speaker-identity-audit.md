# Svarah speaker-identity audit

**Audit date:** 2026-08-11  
**Decision:** **blocked for publication-valid speaker-disjoint experiments.**

## Question and acceptance test

The FR-CISPO protocol requires one authoritative identifier per real Svarah
speaker. A source is acceptable only when all of the following are true:

1. it is tied to an official Svarah source revision or a documented release;
2. it has a non-empty `speaker_id` field aligned to the 6,656 utterances;
3. it contains exactly 117 distinct IDs; and
4. `prepare-data` can cross-check transcripts/durations, create five immutable
   folds, and prove no overlap plus one out-of-fold test appearance per ID.

Demographic-profile signatures, filename components, row order, and an
accidental total of 117 groups are not evidence of a speaker identity.

## Evidence chain

### 1. Official GitHub repository documents the missing field

The official repository is `https://github.com/AI4Bharat/Svarah` at
`2c644dc4f67da2aae63a0bd9ceb2c6f8679ff05c` (2025-07-03). Its README says
that the original `meta_speaker_stats.csv` has 11 columns, including
`speaker_id` as a unique speaker identifier, and describes a release tree
containing `audio/`, `svarah_manifest.json`, and `meta_speaker_stats.csv`.

The complete reachable Git history has seven commits:

```text
2c644dc  2025-07-03  Update README.md
074065e  2023-05-27  Update README.md
82404f0  2023-05-27  Update README.md
cdeafc9  2023-05-26  Updated README.md
8d2840e  2023-05-26  updated README.md
663471d  2023-05-26  added REAME
48fbff0  2023-05-26  initial commit
```

`git ls-tree -r` for every one of these commits contains only the README and
evaluation scripts. No audio, manifest, or metadata CSV was ever committed to
that repository history.

### 2. Every accessible official Hugging Face data revision removes it

The pinned dataset revision is
`ai4bharat/Svarah@ebbf7777fe771490696a3f7b007097606fa8c924`. The Hugging Face
API lists eight reachable commits. Their complete file inventories contain only
`.gitattributes`, `README.md`, and Parquet data shards (three shards after
`e241ad2`, one prior shard at `a5dd7f7`); none contains a CSV, manifest, or
speaker-metadata file.

Reading Parquet footers remotely (without treating paths as identities) shows
the same 11 fields at every data-bearing revision:

```text
audio_filepath, duration, text, gender, age-group, primary_language,
native_place_state, native_place_district, highest_qualification,
job_category, occupation_domain
```

`speaker_id` is absent at all seven data-bearing revisions:

| Revision | Parquet shards | First-shard rows | `speaker_id` field |
| --- | ---: | ---: | --- |
| `ebbf777` | 3 | 2,219 | no |
| `cb155bc` | 3 | 2,219 | no |
| `6b8d523` | 3 | 2,219 | no |
| `3893d83` | 3 | 2,219 | no |
| `189ee7e` | 3 | 2,219 | no |
| `e241ad2` | 3 | 2,219 | no |
| `a5dd7f7` | 1 | 6,656 | no |

The initial commit `db9ab85` contains no Parquet data. The pinned README does
still state that Svarah has 6,656 utterances and 117 speakers; this is a
dataset-level claim, not a recoverable mapping from utterance to speaker.

### 3. Existing development fallback is correctly non-authoritative

The supplied development CSV is pinned in prior run evidence by SHA-256
`e2daa48863581eb41befd1826b7b14cd80e05f3a1bab9b72d08e7814248f1f94` and was
used only to derive 115 demographic-profile clusters. It is not an
authoritative speaker mapping. The profile-cluster pipeline must remain marked
`publication_valid: false` and must not be combined with speaker-level OOF
results.

## Commands run

```powershell
uv run python -c "from huggingface_hub import HfApi; ..."
```

This returned eight official Hugging Face commits, all of whose file inventories
were inspected with `HfApi.list_repo_files(..., revision=<commit>)`.

```powershell
C:\Kaustubh\ast-asr\.venv\Scripts\python.exe -
```

Using `HfFileSystem` plus `pyarrow.parquet.ParquetFile`, this read the remote
Parquet footer and Arrow schema for each data-bearing official revision. It
returned `False` for `"speaker_id" in schema.names` for all seven revisions.

```powershell
git clone --filter=blob:none --no-checkout https://github.com/AI4Bharat/Svarah.git ...
git -C <clone> log --all --oneline
git -C <clone> ls-tree -r --name-only <each-commit>
```

This established that the GitHub repository documents the original release
format but does not host the required files in reachable history.

## Implemented guard

`ast_asr.data.audit_authoritative_speaker_metadata` now records a candidate
file's SHA-256, required columns, row count, empty IDs, and distinct ID count.
It returns `structurally_valid_for_speaker_folds: true` only for a candidate
with the required identity fields and exactly 117 non-empty unique IDs. It does
not claim that the file is official; its recorded source provenance and
`prepare-data` alignment validation remain mandatory.

`inspect_svarah_storage` now embeds this audit per CSV candidate and no longer
sets `speaker_fold_ready` merely because a file is named
`meta_speaker_stats.csv`.

When the authoritative path does pass, `prepare-data` now records SHA-256
hashes of both `svarah_manifest.json` and `meta_speaker_stats.csv` inside the
immutable dataset manifest. This binds later fold and prediction artifacts to
the exact files that passed the identity check.

The local guard is covered by:

```powershell
uv run pytest -q tests/test_data_manifest.py
# 4 passed
uv run python -m compileall -q src scripts/modal_fr_cispo.py
git diff --check
```

## Exact blocker and next action

No accessible official artifact establishes the required utterance-to-117
speaker mapping. Therefore do **not** run publication-valid SFT, policy, fold,
or clustered-bootstrap experiments from the public Parquet release.

The narrow next action is to obtain the original Svarah archive (or an official
utterance-to-speaker mapping) from the maintainers, with its release URL or
revision, file SHA-256, and confirmation that it matches the 6,656 released
utterances. The official dataset README names Tahir Javed
(`tahir@cse.iitm.ac.in`) as its contact. Once received, save the immutable
provenance record, run `ast-asr prepare-data`, and require the acceptance test
above before opening the speaker-level experiment gate.
