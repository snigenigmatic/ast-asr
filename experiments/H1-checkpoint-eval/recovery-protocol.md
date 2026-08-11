# H1 checkpoint evaluation launcher recovery protocol

**Status:** preregistered, exploratory infrastructure retry. This document permits exactly one new Modal invocation after a committed launcher repair; it does not authorize training, checkpoint replacement, objective changes, or a second retry.

## Failure being repaired

The registered invocation `profile-h1-checkpoint-eval-20260811` failed before checkpoint loading or audio decoding. Modal imported `scripts/modal_fr_cispo.py` with an interpreter outside the project `uv run` environment; the direct `ast_asr.modal_evaluation` import at line 518 raised `ModuleNotFoundError`. The immutable failure record is `failure-launcher-20260811.md` and must not be edited.

**No metrics were observed and no output directory existed for the failed invocation.** In particular, it produced no `resolved_config.json`, `run.json`, `metrics.json`, `predictions.jsonl`, or `edit_counts.json`, and generated zero of the expected 5,772 predictions. This recovery is a fresh infrastructure attempt, not a continuation or reinterpretation of that failed run.

## Frozen change and unchanged semantics

The only allowed code change is evaluator bootstrap placement:

1. `scripts/modal_fr_cispo.py::run_profile_evaluation` must not import `ast_asr` directly in Modal's wrapper interpreter.
2. It must invoke `uv run --frozen python -m ast_asr.modal_evaluation resolve-checkpoint ...` before `ast-asr evaluate-fold`.
3. The package resolver remains the sole implementation of allowed arm labels and checkpoint-coordinate validation. Thus `zero-shot` alone resolves to `base`; `sft` and every policy arm, including `fr-cispo`, require exactly three single-component coordinates and preserve their input label.

No other evaluator, data, model, checkpoint, decoding, output naming, or training semantics may change. The evaluation command remains FP32 through `evaluate-fold` and retains the existing solo/batched prediction checks.

## Fixed inputs

| Item | Required value |
| --- | --- |
| Retry run name | `profile-h1-checkpoint-eval-r1-20260811` |
| Requested arm | `fr-cispo` |
| Historical policy run | `profile-h1-klstop-s2026-20260811` |
| Historical checkpoint output | `h1-klstop-fr-cispo` |
| Historical checkpoint directory | `checkpoint-final` (legacy bounded-smoke name; not a 300-cycle final model) |
| H1 adapter content hash | `4654e4116cb1a9ebce142e3ccdc11ddd36ec3bb323c9a0ff879b6b0b3c987fe8` |
| H1 starting SFT adapter hash | `d204df40dfcd694733a171998ad5d97fdb43eecbc5dc19846d98bce012cd4c1e` |
| Required volume config hash | `94c5312c70353b1bc597ffe95c1a1f4c32166a03187e0636ddae9f2df9e6317c` |
| Base model | `openai/whisper-tiny@169d4a4341b33bc18d8881c4b69c2e104e1cc0af` |
| Dataset | `ai4bharat/Svarah@ebbf7777fe771490696a3f7b007097606fa8c924` |
| Profile metadata CSV | SHA-256 `e2daa48863581eb41befd1826b7b14cd80e05f3a1bab9b72d08e7814248f1f94` |
| MUSAN archive | MD5 `0c472d4fc0c5141eca47ad1ffeb2a7df` |

## Preconditions and stop rules

Before submission, record the committed revision containing this protocol and the launcher repair. Verify all of the following:

1. The historical checkpoint exists at the exact three-component volume path and its directory content hash equals the fixed H1 adapter hash.
2. The volume-side config hash equals the frozen hash above.
3. The new retry output directory is absent. The failed run name remains permanently reserved and must not be reused.
4. Targeted tests prove the package CLI resolves both `zero-shot` and `fr-cispo`, and static regression coverage proves the Modal wrapper has no direct `ast_asr.modal_evaluation` import.

Stop and write a new immutable failure record if package resolution, input identity, adapter loading, mask/invariance checks, prediction count, or metric finiteness fails. Do not submit another command, retrain, rename the old adapter, modify the objective, or substitute a checkpoint.

## Exactly one permitted command

```powershell
$env:PYTHONUTF8='1'
$env:PYTHONIOENCODING='utf-8'
uvx modal run scripts/modal_fr_cispo.py::run_profile_evaluation `
  --run-name profile-h1-checkpoint-eval-r1-20260811 `
  --arm fr-cispo `
  --checkpoint-run-name profile-h1-klstop-s2026-20260811 `
  --checkpoint-output-name h1-klstop-fr-cispo `
  --checkpoint-name checkpoint-final
```

The expected output path is `/artifacts/profile-h1-checkpoint-eval-r1-20260811/evaluation-fr-cispo`. The only allowed outcomes are: (a) one complete exploratory evaluation record, or (b) one immutable failure record explaining why the evaluation stopped.

## Interpretation boundary

This run evaluates the existing 20-cycle bounded H1 adapter using 115 demographic-profile clusters. It must be marked `publication_valid: false`. No speaker-disjoint claim, confidence interval, development-gate decision, learning-rate selection, five-fold launch, or paper-result claim is allowed.
