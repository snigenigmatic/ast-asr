# H7 sentinel-KL r1 implementation-bound authorization — 2026-08-12

**Status:** authorized for exactly one r1 fixed-policy diagnostic attempt only
after every preflight below passes. This authorization implements the locked
`recovery-protocol.md`; it does not amend the H7 protocol, authorize training,
or permit reuse of the consumed r0 attempt.

## Immutable release and evidence identity

The r1 executable baseline is
`c5a7baf6d7b0aba13950c449fbdea01a22be1e32`
(`fix: isolate H7 r1 recovery coordinates`). Its r0 failure/recovery evidence
is committed at `080a19c07627389a5ca6ec11ff246305ad490d58`. The final reviewed
authorization commit must be tagged `h7-r1-authorized-20260812`; that tag must
resolve to `HEAD` and is the exact `H7_SOURCE_COMMIT` recorded at launch.

| Artifact | SHA-256 / commit |
| --- | --- |
| H7 r1 executable baseline | `c5a7baf6d7b0aba13950c449fbdea01a22be1e32` |
| r0 failure/recovery evidence commit | `080a19c07627389a5ca6ec11ff246305ad490d58` |
| Locked H7 protocol | `e1e40ff87bad8aea699d2912617cae7b83f9372c` |
| H7 input lock | `27f4c033eaba568f74f500ab560cb7b0bd2945b8c4bd8814cb37fb9cb8d41187` |
| Modal r1 wrapper | `98669c299eb84a52e1bdd5a8a40ad8473d030d2a9ae5eb25add80043f47c1ad6` |
| Full `src/` manifest | `5400a60c74ff2b195e0c1b33db95036b9f4459961dbb26431d070467b58aefcd` |
| Full `configs/` manifest | `23c9785cdd8feebb23ddab3d56d7c3be3fe9dee12b8f17e2cf49e482670c2665` |
| `uv.lock` | `f6ed29f46ad81e91637368cc91bf7b30134a08ab6dc0efdb8e590f374275312a` |
| `pyproject.toml` | `6852fe9145c7a6df3d5c64d3f8ad2e675a8935f6e57347c49997dd3cfff0e934` |
| r0 failure record (Markdown) | `22f28707df99edb24d6d47735d62ee6ac31c78cf02f28f0ff12c91a0fe8d15b5` |
| r0 failure record (JSON) | `84334a9583222ade0cd1f51fb0fc19a18c08e68d3a010c77ffc0329989faf12f` |
| Locked r1 recovery protocol | `523cc18620300c5cfb6de6930529e977b3f4b9bccb6cba0885b545915cc707ae` |

The frozen model/data/checkpoint/configuration identities, 28 locked replay
banks, candidate tokens/masks, H6 manifest, acoustic replay, sequence/K3 math,
FP16 CUDA scoring and FP32 emitted arithmetic remain exactly those registered
by the locked H7 protocol and input lock. The r1 coordinate change is not a
new experiment or new input selection.

## One permitted r1 target

The only app and coordinates are:

```text
app: ast-asr-h7-sentinel-kl-r1
run root: /artifacts/profile-h7-fixed-policy-sentinel-kl-r1-s2028-20260812
output: /artifacts/profile-h7-fixed-policy-sentinel-kl-r1-s2028-20260812/h7-fixed-policy-sentinel-kl-r1
```

The old app `ast-asr-h7-sentinel-kl` and old r0 root
`/artifacts/profile-h7-fixed-policy-sentinel-kl-s2028-20260812` are preserved
failure evidence. They must not be selected, reused, explicitly stopped, or
mutated. A natural backend state transition is not an r1 action.

The one and only permitted command is:

```powershell
$env:PYTHONIOENCODING='utf-8'
uvx modal run scripts/modal_h7_sentinel.py
```

The r1 function uses an L4 with `retries=0`; no second command, name reuse,
implicit retry, or recovery is authorized.

## Required preflight

Run all commands from the repository root before the one invocation. The two
commit commands must print the same full commit; the three worktree commands
must produce no output; every printed hash must match the table above.

```powershell
git rev-parse HEAD
git rev-parse 'h7-r1-authorized-20260812^{commit}'
git diff --exit-code c5a7baf6d7b0aba13950c449fbdea01a22be1e32 -- scripts/modal_h7_sentinel.py src configs uv.lock pyproject.toml experiments/H7-sentinel-kl/input-lock.json
git status --porcelain -- scripts/modal_h7_sentinel.py src configs uv.lock pyproject.toml experiments/H7-sentinel-kl/input-lock.json
git status --porcelain
(Get-FileHash -Algorithm SHA256 scripts/modal_h7_sentinel.py).Hash.ToLower()
(Get-FileHash -Algorithm SHA256 experiments/H7-sentinel-kl/input-lock.json).Hash.ToLower()
(Get-FileHash -Algorithm SHA256 experiments/H7-sentinel-kl/failure-launcher-20260812.md).Hash.ToLower()
(Get-FileHash -Algorithm SHA256 experiments/H7-sentinel-kl/failure-launcher-20260812.json).Hash.ToLower()
(Get-FileHash -Algorithm SHA256 experiments/H7-sentinel-kl/recovery-protocol.md).Hash.ToLower()
uv run --frozen python -c "from pathlib import Path; from ast_asr.modeling import directory_content_hash; print(directory_content_hash(Path('src'))); print(directory_content_hash(Path('configs')))"
```

The local Modal launcher stream must be fixed before `uvx modal` begins; an
in-process H7 `main()` cannot repair Rich output that occurs during Modal CLI
initialization. Run and require a zero exit from this no-remote probe:

```powershell
$env:PYTHONIOENCODING='utf-8'
uvx --from modal==1.5.3 python -c "import locale, sys; assert sys.stdout.encoding.lower() == 'utf-8'; assert sys.stderr.encoding.lower() == 'utf-8'; print(sys.stdout.encoding, sys.stderr.encoding, locale.getpreferredencoding(False))"
```

Its first two fields must be `utf-8 utf-8`; the preferred encoding may remain
`cp1252`. Also require a read-only volume listing to fail with the missing-path
diagnostic for the new root; a successful listing forbids the invocation:

```powershell
uvx modal volume ls ast-asr-fr-cispo-runs profile-h7-fixed-policy-sentinel-kl-r1-s2028-20260812
```

Before launch, the 102-test suite, the exact H7 executable-scope Ruff command
below, compileall, and `git diff --check` must pass and be recorded in the r1
launch evidence:

```powershell
uvx ruff check src/ast_asr scripts/modal_h7_sentinel.py tests/test_h7_core.py tests/test_h7_runner.py
```

Repository-wide Ruff currently reports 65 historical findings confined to
quarantined legacy code outside this executable scope; those findings are not
part of H7 and are not silently relabelled as passing. Failure to install or
run the exact scoped command is a failed preflight, not permission to skip it.

## Measurement and terminal boundary

The pinned Linux/amd64 CUDA image remains
`nvidia/cuda@sha256:09d8951b943dee03cf8fc841b6ea1f201ad33f82f76567171394853c0f494054`.
Both models remain frozen/evaluation-mode FP16 on CUDA. Before any forward
pass, r1 must validate all replay/audio/feature locks. Phase A scores cycle 027
twice and must reproduce saved-old K3/token `0.10983546078205109` within
`1e-7`, direct-policy saved-log-probabilities within `1e-5`, and repeated
production-shape log probabilities within `1e-7`; only then may it score the
remaining 27 banks once.

Every post-claim exception must write and commit immutable r1 failure evidence;
success must commit its immutable source/final manifests, rows, summaries, and
terminal decision. Missing identity, image, input, Phase A, determinism, or
scoring evidence is terminal `measurement_failed` / `non_evaluable`. All r1
artifacts retain `publication_valid: false` and `profile_cluster_count: 115`.
No outcome authorizes optimizer steps, training, decoding, WER selection,
controller/beta changes, another r1 attempt, or a further recovery.
