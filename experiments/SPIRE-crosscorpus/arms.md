# SPIRE cross-corpus evaluation arms — verified coordinates

Verified by listing the `ast-asr-fr-cispo-runs` volume on 2026-08-20, not by
reading result documents. Use these literal values; a wrong name fails closed in
`evaluate_spire` rather than silently evaluating the wrong adapter.

| Arm label | `--checkpoint-run-name` | `--checkpoint-output-name` | `--checkpoint-name` |
| --- | --- | --- | --- |
| `zero-shot` | *(omit all three)* | *(omit)* | *(omit)* |
| `sft-epoch1` | `profile-dev-full-sft-20260810` | `profile-sft-development` | `checkpoint-epoch-1` |
| `h5-beta0-s2026` | `profile-h5-refkl-beta0-r1-s2026-20260812` | `h5-beta0-fr-cispo` | `checkpoint-last-safe` |
| `h5-beta004-s2026` | `profile-h5-refkl-beta004-s2026-20260812` | `h5-beta004-fr-cispo` | `checkpoint-last-safe` |
| `h6-beta0-s2027` | `profile-h6-refkl-beta0-s2027-20260812` | `h6-beta0-fr-cispo` | `checkpoint-last-safe` |
| `h6-beta004-s2027` | `profile-h6-refkl-beta004-s2027-20260812` | `h6-beta004-fr-cispo` | `checkpoint-last-safe` |

## Traps that would otherwise cost a run

1. **The H5 beta-zero control lives in the `-r1-` run.**
   `profile-h5-refkl-beta0-s2026-20260812` contains only `resolved-policy-configs`
   and **no adapter output** — that launch failed and was recovered. The audited
   control checkpoint is under `profile-h5-refkl-beta0-r1-s2026-20260812`. The
   beta-`0.04` treatment has no `-r1-` variant.
2. **Policy runs expose `checkpoint-last-safe`, not `checkpoint-final`.** These
   were bounded 40-cycle runs, so the trainer classified them
   `exploratory_bounded` and named the checkpoint accordingly.
3. **SFT exposes five epoch checkpoints.** Epoch 1 is the selected one per
   `research-state.yaml` (`selected_epoch: 1`, validation macro-family WER
   `0.1766945`). Do not silently evaluate a later epoch.
4. **Seed 2028 has no treatment arm.** Its beta-zero control tripped the KL gate
   at cycle 27, so by the locked stop rule the treatment was never launched. Any
   seed-2028 comparison is impossible, not merely missing.

## Provenance cross-check

`spire_eval_entry.py` records `adapter_checkpoint_revision` via
`directory_content_hash`. Confirm it equals the revision already published in
`experiments/H6-replication/result-20260812.md`:

| Arm | Expected checkpoint revision |
| --- | --- |
| `h5-beta0-s2026` | `eb519c5b60dba9573c4c56808ff9526005b5cd0e1bebae79851a1aea13e902ea` |
| `h5-beta004-s2026` | `9ac3b236270dd9a9bbce12f4b7988aac3a8e5f5942ec4b7ce7ef3e2da41cd41b` |
| `h6-beta0-s2027` | `33b389492935d45ad2f773a6bd82fae2cf188b6517fed057b15fbde3c61b814c` |
| `h6-beta004-s2027` | `f53aa3acdad257fbe569a10a0d3e5de19d7e625b4d73c05315489833357382cc` |
| SFT source (epoch 1) | `d204df40dfcd694733a171998ad5d97fdb43eecbc5dc19846d98bce012cd4c1e` |

A mismatch means the volume contents drifted from the recorded H5/H6 evidence and
the evaluation must stop rather than be reported.

## Registered pairings

Cross-corpus deltas are only meaningful within a matched pair:

- `h5-beta0-s2026` vs `h5-beta004-s2026` — the seed where the in-domain worst
  group improved 2.22 pp.
- `h6-beta0-s2027` vs `h6-beta004-s2027` — the seed where it worsened 0.40 pp.

`zero-shot`, `sft-epoch1`, and the two beta-zero controls also form a descriptive
ladder (base -> supervised -> RL) but that ladder is not a matched pair, so it
carries no bootstrap test.

The interesting question this can answer: **does the beta penalty's in-domain
sign flip between seeds reproduce out-of-domain on real speakers?** If the
cross-corpus deltas disagree in sign across the two seeds as well, the
non-replication is a property of the method rather than of the Svarah folds.
