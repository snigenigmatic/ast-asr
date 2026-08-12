# H5 beta-zero launcher recovery protocol

Status: **locked recovery; one retry authorized after commit**

Publication validity: **false**

## Failure boundary

The preregistered beta-zero invocation
`profile-h5-refkl-beta0-s2026-20260812` failed before `train-policy`, model
loading, CUDA rollout, or cycle zero. The Modal wrapper attempted to import
`torch` from its launcher interpreter while constructing environment
provenance. That interpreter does not contain the project environment, which
is available only through `uv run --frozen`. No H5 loss, ratio, KL, checkpoint,
prediction, or WER observation exists under the failed name. It is a launcher
failure and cannot enter the matched comparison.

## Sole repair

Collect the same runtime fields by executing a JSON-only Python probe through
the already pinned project command:

```text
uv run --frozen python -c <runtime probe>
```

No model, objective, seed, data, checkpoint, optimizer, safety threshold,
cycle count, or evaluation setting may change. The probe must remain
read-only and its returned values must still be written into the immutable
launcher artifact.

Local gates before the retry:

1. full tests, Ruff, compileall, and `git diff --check` pass;
2. a regression test or isolated launcher smoke proves the wrapper itself no
   longer imports `torch` directly;
3. the repair and this recovery protocol are committed;
4. the failed run remains untouched and the recovery run directory is absent.

The local isolated launcher smoke executed the repaired probe through the
project environment and returned Python, Torch, CUDA-availability, lockfile,
launcher, and source hashes without importing Torch in the wrapper
interpreter. The full local suite passed 66 tests; Ruff, compileall, and diff
checks passed before this protocol was committed.

## One authorized recovery invocation

The only allowed beta-zero retry uses a new run name and otherwise preserves
the original command byte-for-byte:

```powershell
uvx modal run scripts/modal_fr_cispo.py::run_profile_fr_cispo_smoke `
  --run-name profile-h5-refkl-beta0-r1-s2026-20260812 `
  --seed 2026 `
  --sft-run-name profile-dev-full-sft-20260810 `
  --sft-output-name profile-sft-development `
  --output-name h5-beta0-fr-cispo `
  --learning-rate 0.00001 `
  --reference-kl-beta 0.0 `
  --rollout-cycles 40 `
  --probe-examples 32 `
  --maximum-new-tokens 225
```

There is no second automatic retry. A pre-cycle failure is preserved as new
engineering evidence. A model/safety failure is an H5 control result and stops
the paired experiment pending root review.

If the recovery control completes safely, its fixed evaluation checkpoint is:

```text
/artifacts/profile-h5-refkl-beta0-r1-s2026-20260812/
  h5-beta0-fr-cispo/checkpoint-last-safe
```
