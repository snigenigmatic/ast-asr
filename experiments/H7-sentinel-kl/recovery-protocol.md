# H7 sentinel-KL recovery protocol — r1

**Status:** **LOCKED — NOT APPROVED FOR RUN.** This document registers a
possible recovery boundary only. It does not authorize Modal, a code change, an
image build, a model load, a measurement, or any action against the original
H7 app or reserved output.

## Failure boundary

The original H7 authorization is consumed by its launcher failure, recorded in
`failure-launcher-20260812.md`. No task, container, image ID, function
execution, model load, scoring forward pass, or output write was observed;
transient image preparation remains unresolved. It is non-result engineering
evidence, not a K3, KL, fairness, robustness, or controller observation. The
original app and its absent reserved run root are immutable evidence and must
not be reused, resumed, explicitly stopped, or otherwise mutated by this
recovery protocol. A natural backend state transition may occur independently.

## Registered r1 coordinates

The only possible recovery uses a new, never-reused namespace:

```text
profile-h7-fixed-policy-sentinel-kl-r1-s2028-20260812
profile-h7-fixed-policy-sentinel-kl-r1
/artifacts/profile-h7-fixed-policy-sentinel-kl-r1-s2028-20260812
/artifacts/profile-h7-fixed-policy-sentinel-kl-r1-s2028-20260812/h7-fixed-policy-sentinel-kl-r1
```

The old app, old profile, old output, and old run name are prohibited. There is
no implicit retry and no reuse of the original `ast-asr-h7-sentinel-kl` app.
The new app name is exactly `ast-asr-h7-sentinel-kl-r1`.

## Sole permitted repair

After this protocol is separately reviewed and committed, the recovery source
may make exactly these launcher-level changes:

1. set the Windows launcher environment variable
   `PYTHONIOENCODING=utf-8` before the one local Modal CLI invocation; and
2. make coordinate-only updates to the Modal app name and H7 profile/output
   constants across the wrapper, `h7_runner`, `h7_modal`, and their tests, so
   the r1 coordinates and app name above are used consistently.

No other source, image, Python, dependency, model, dataset, processor, input
lock, bank, scorer, command argument, dtype, threshold, tolerance, seed,
corruption, checkpoint, reference, or arithmetic change is permitted. In
particular, r1 uses the same frozen policy/reference/config/prepared/fold
identities, 28 locked banks, sequence/token scoring math, Phase A gates,
FP16-CUDA model path, FP32 emitted arithmetic, and fail-closed evidence rules
as the locked H7 protocol and implementation-bound authorization.

## Required release and preflight gates

Before any r1 invocation, all of the following must be true:

1. the failure record and this recovery protocol have been reviewed and
   committed with the narrowly scoped launcher repair in a new reviewed code
   commit and immutable tag;
2. the exact r1 executable-source commit, source-tree hashes, input-lock hash,
   base-image digest, dependency lock, and command are recorded in a new r1
   implementation-bound authorization;
3. focused and full tests, Ruff, compileall, and `git diff --check` pass;
4. a local, no-Modal regression test demonstrates that the launcher command is
   supplied `PYTHONIOENCODING=utf-8` and that only r1 paths are constructed;
   before any invocation, this exact no-remote interpreter probe must also
   exit zero:

   ```powershell
   $env:PYTHONIOENCODING='utf-8'
   uvx --from modal==1.5.3 python -c "import locale, sys; assert sys.stdout.encoding.lower() == 'utf-8'; assert sys.stderr.encoding.lower() == 'utf-8'; print(sys.stdout.encoding, sys.stderr.encoding, locale.getpreferredencoding(False))"
   ```

   This tests the actual uvx Python stdout/stderr streams, not merely the
   PowerShell-reported encoding. Its expected stream output is `utf-8 utf-8`;
   the preferred encoding may remain `cp1252`;
5. the original failure evidence remains unchanged, the original H7 app is not
   selected, and the new r1 run root is absent as proved by a read-only volume
   listing; and
6. the new Modal app/function declaration has `retries=0`, non-creating
   volumes, a nonempty runtime `MODAL_IMAGE_ID` requirement, and the same
   failure-writer/volume-commit behavior.

Failure of any gate means r1 is not authorized. No broad worktree change,
alternate command, or stale app is an acceptable substitute.

## Conditional one-attempt command

Only a subsequent r1 implementation-bound authorization may permit this
single invocation, from Windows PowerShell:

```powershell
$env:PYTHONIOENCODING='utf-8'
uvx modal run scripts/modal_h7_sentinel.py
```

The command is allowed exactly once after the preceding gates pass. The
function must retain `retries=0`. A launcher/image/container/input/identity/
model/Phase-A failure consumes r1 and writes non-evaluable failure evidence;
success writes the same immutable measurement evidence under the r1 output.
Neither outcome authorizes a second r1 attempt, controller/beta adjustment,
training, rollout generation, decoding, WER selection, or any other recovery.
