# H1 checkpoint evaluation launcher failure — 2026-08-11

**Classification:** exploratory engineering attempt; `publication_valid: false`.

## Outcome

**Failed before evaluation.** The one registered Modal invocation built its
images and started an L4 container, then raised `ModuleNotFoundError: No module
named 'ast_asr'` from `scripts/modal_fr_cispo.py:518`, before the historical
checkpoint was loaded or any audio was decoded. No evaluation output directory
or prediction artifact was created. This is a launcher/import failure, not a
WER, fairness, robustness, model-loading, or policy-training result.

The protocol prohibits a retry under the same run name. No training, new
checkpoint, alternate command, or second evaluation was launched.

## Registered invocation identity

| Field | Value |
| --- | --- |
| Submitting Git revision | `294ad99bd277874936797e27c213b16c1e87f02a` |
| Modal app | `ap-Yvpz9qb4tuKUqRR1FNg0bi` |
| Modal function call | `fc-01KZQXAQHCXEMTYHWV81ZGP8TJ` |
| L4 container | `ta-01KZQXAR2NCHT86SJGJPSC5A2R` |
| Container start | 2026-08-11 13:29:30 IST |
| Failure / app stop | 2026-08-11 13:29:32 / 13:29:34 IST |
| Fixed run name | `profile-h1-checkpoint-eval-20260811` |
| Requested arm | `fr-cispo` |
| Historical checkpoint coordinates | `profile-h1-klstop-s2026-20260811/h1-klstop-fr-cispo/checkpoint-final` |

```powershell
$env:PYTHONUTF8='1'
$env:PYTHONIOENCODING='utf-8'
uvx modal run scripts/modal_fr_cispo.py::run_profile_evaluation `
  --run-name profile-h1-checkpoint-eval-20260811 `
  --arm fr-cispo `
  --checkpoint-run-name profile-h1-klstop-s2026-20260811 `
  --checkpoint-output-name h1-klstop-fr-cispo `
  --checkpoint-name checkpoint-final
```

## Input and source hashes

| Artifact | SHA-256 / revision |
| --- | --- |
| H1 adapter content hash (required historical input) | `4654e4116cb1a9ebce142e3ccdc11ddd36ec3bb323c9a0ff879b6b0b3c987fe8` |
| H1 starting SFT adapter | `d204df40dfcd694733a171998ad5d97fdb43eecbc5dc19846d98bce012cd4c1e` |
| Historical volume config (required) | `94c5312c70353b1bc597ffe95c1a1f4c32166a03187e0636ddae9f2df9e6317c` |
| H1 result record | `4183ae08b71c319e5db2152fb993faaa9183b2e9d4324087c56c51b94947cf65` |
| Locked evaluation protocol | `46c24f2e2349a3399c08a073ad0b7ed9ed701f03d7675664129d55fc42d80c8a` |
| Submitted Modal launcher | `0c2863fcd239034aa0dcbbba9677fb86942461213c2d5663066265c61b86857b` |
| Submitted evaluator-label helper | `fa081fc79185f86a98a58a538cc4dceb0c77c4e09e2bfa808d181e9dc7ad6d02` |
| Local JSON config (not a replacement for the required volume config) | `83a9286761a430a29f10349e8e2c822254c46cd634c7ef4019946526a78895a5` |

The output-volume inspection returned **no such directory** for
`profile-h1-checkpoint-eval-20260811`. The historical input directory was
separately inspected and still contained `checkpoint-final`, `run.json`,
`resolved_config.json`, rollout diagnostics, and movement artifacts. Thus the
failure did not overwrite or remove the historical checkpoint.

## Metrics

No `resolved_config.json`, `run.json`, `metrics.json`, `predictions.jsonl`, or
`edit_counts.json` was produced for this invocation; consequently no output
artifact hash, FP32 WER, condition-level metric, or solo/batched prediction
result exists. The expected 5,772 predictions were not generated.

Any future attempt requires a new protocol and a committed launcher import-path
repair. It must use a new run name and retain `publication_valid: false` until
authoritative Svarah speaker IDs are available.
