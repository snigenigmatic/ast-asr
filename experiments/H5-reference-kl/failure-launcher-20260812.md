# H5 beta-zero launcher failure — 2026-08-12

**Status: launcher failure; not an H5 experiment result.** This is an
immutable evidence record of one authorized invocation of the preregistered
beta-zero control. It did not execute `train-policy`, did not begin cycle 0,
and does not update the H5 safety, movement, KL, or WER evidence.

## Invocation identity

| Field | Value |
| --- | --- |
| Source revision | `4f447e9e7b82c23a344bde228be0580ebdb4b9a7` |
| Modal app | `ap-X6vBlRrVEngIW2eEdzqU0m` ([run](https://modal.com/apps/snigenigmatic/main/ap-X6vBlRrVEngIW2eEdzqU0m)) |
| Function | `run_profile_fr_cispo_smoke` / `fu-fxQMVwJP38m8ipgrlaJNIf` |
| Function call | `fc-01KZTKXM6C4N65HCPCPQTG099K` |
| Function container | `ta-01KZTKXPPTK9QNBG2VY5YKB19R` |
| Failure timestamp | `2026-08-12 14:42:57+05:30` |
| Requested arm | beta-zero FR-CISPO, `reference_kl_beta=0.0` |
| Requested output | `profile-h5-refkl-beta0-s2026-20260812/h5-beta0-fr-cispo` |

The launch built image IDs `im-NCeBLHinrJb9NUJyq8Wf5Z`,
`im-5j4knT7ZYecYpBojawk08p`, `im-SXwZTa8cmmrZHI4jphsVyb`, and
`im-XrA2MxKQ5S75wXeiWECE2d`. The runtime container printed CUDA 12.8 before
the exception, but its launcher interpreter did not have `torch` available.

## Exact stop

Modal's timestamped application log records this stack at the failure time:

```text
File "/root/modal_fr_cispo.py", line 692, in run_profile_fr_cispo_smoke
    "modal_runtime": _modal_runtime_identity(),
File "/root/modal_fr_cispo.py", line 127, in _modal_runtime_identity
    import torch
ModuleNotFoundError: No module named 'torch'
```

The exception happened while constructing the launcher provenance object,
before `_write_once(...policy-launches...)`, before `subprocess.run(command)`,
and before the wrapper's failure-commit handler. Therefore no policy container
work, source-SFT loading, rollout generation, optimizer step, safety gate, or
cycle diagnostic occurred. This is an image/interpreter boundary defect in the
launcher, not a fail-closed H5 policy safety stop.

## Artifact inspection

Immediately after termination, `modal volume ls ast-asr-fr-cispo-runs
profile-h5-refkl-beta0-s2026-20260812` showed only
`resolved-policy-configs/h5-beta0-fr-cispo.json`. That immutable partial
configuration records `policy.reference_kl_beta: 0.0`.

The following artifacts are absent:

- `policy-launches/h5-beta0-fr-cispo.json`;
- `h5-beta0-fr-cispo/failure.json`;
- `h5-beta0-fr-cispo/run.json` and `movement.json`;
- `h5-beta0-fr-cispo/rollouts/` and `diagnostics/`;
- `h5-beta0-fr-cispo/checkpoint-last-safe`.

No beta-0.04 treatment or evaluation was launched. Do not compare this stop
with H1 or use it to support or falsify the reference-KL hypothesis. A future
launch requires a new reviewed recovery protocol after the launcher image
boundary is repaired; this record does not authorize a retry.
