# H7 implementation-bound authorization — 2026-08-12

**Status:** authorized for exactly one fixed-policy diagnostic measurement
attempt, conditional on every preflight below passing. This authorization does
not amend the locked [protocol](protocol.md); if a preflight or runtime gate
fails, the attempt ends as non-evaluable evidence.

## Authorized scope and immutable identity

The locked input source baseline is Git commit
`b36b935f8fb942b98255902c10d60c0d270b6bb5` (`research: freeze H7 replay
input lock`). The executable implementation baseline is Git commit
`e4053c632577d22e6d86ccf5848a2686780452f8` (`fix: pin H7 image inputs`).
The committed authorization must be tagged `h7-authorized-20260812`; that tag
must resolve to `HEAD`, and the resolved commit is the exact launch source
commit recorded as `H7_SOURCE_COMMIT`. The separate implementation commits are:

| Component | Immutable commit |
| --- | --- |
| Locked protocol | `e1e40ff87bad8aea699d2912617cae7b83f9372c` |
| H7 scoring core | `b637efd4913a572c972a348ac871b9b2613841cc` |
| H7 runner/wrapper | `7e69b553a92df5388a97df0927354e8f71a49fe7` |
| Offline input-lock builder | `7eb055e9b4b796dafd6c0b9a887fae48cfd007b7` |
| Frozen input-lock source baseline | `b36b935f8fb942b98255902c10d60c0d270b6bb5` |
| Executable source baseline | `e4053c632577d22e6d86ccf5848a2686780452f8` |

Only the executable scope below may be sent to Modal, and it must byte-match
the executable implementation baseline:

```text
scripts/modal_h7_sentinel.py
src/
configs/
pyproject.toml
uv.lock
experiments/H7-sentinel-kl/input-lock.json
```

The frozen SHA-256 identities are:

| Input | SHA-256 |
| --- | --- |
| H7 input lock | `27f4c033eaba568f74f500ab560cb7b0bd2945b8c4bd8814cb37fb9cb8d41187` |
| H6 56-file manifest | `92c5b7c7fb3457ca41da462fd8475d8a29694541f9f7ae2137fe989f10118c7e` |
| Resolved beta-zero configuration | `3673abefc4322f4951ee067c8b6ed2c2fef93008b3f85c2cf66afd5abd406ae5` |
| Prepared manifest | `65bdd8cf87f5db0f815e742739be815d2306ddd2b9977ee5687774feb1a18b56` |
| Fold-0 manifest | `22e9ab64006fe8a33bac37f5f2b98887df6aed061e158252778c29c6d928a1f0` |
| Policy checkpoint directory | `a95530fd914b7fea9f3008a5c6451f3fedef2281443fce6b9dc0df5ba6a8d400` |
| SFT reference directory | `d204df40dfcd694733a171998ad5d97fdb43eecbc5dc19846d98bce012cd4c1e` |
| Cycle-027 trainable-parameter revision | `5292f4896a06ce2d1c7abf9dd589af01fb5b702e5bbf6fff8f4ea2fcb66c8ea9` |
| `uv.lock` | `f6ed29f46ad81e91637368cc91bf7b30134a08ab6dc0efdb8e590f374275312a` |
| `pyproject.toml` | `6852fe9145c7a6df3d5c64d3f8ad2e675a8935f6e57347c49997dd3cfff0e934` |
| Modal wrapper | `64a226cb19842309643408f2a948cbd53145ebf62c0b76d8db2efaab7efad333` |
| Full `src/` package manifest | `f5172faba0f22c726a72692fe36e6e641f53b77029a8e2e1b39c75ac726d0914` |
| Full `configs/` package manifest | `23c9785cdd8feebb23ddab3d56d7c3be3fe9dee12b8f17e2cf49e482670c2665` |

The model and dataset are fixed to `openai/whisper-tiny` revision
`169d4a4341b33bc18d8881c4b69c2e104e1cc0af` and
`ai4bharat/Svarah` revision `ebbf7777fe771490696a3f7b007097606fa8c924`.

## One permitted attempt

There is one reserved run root and one output leaf:

```text
/artifacts/profile-h7-fixed-policy-sentinel-kl-s2028-20260812
/artifacts/profile-h7-fixed-policy-sentinel-kl-s2028-20260812/h7-fixed-policy-sentinel-kl
```

The only permitted invocation is:

```powershell
uvx modal run scripts/modal_h7_sentinel.py
```

It invokes the module-global `run_h7_sentinel` function in the
`ast-asr-h7-sentinel-kl` app. The function is one L4 job, `retries=0`,
CPU `4`, memory `16384` MiB, and timeout `3600` seconds. No second invocation,
reserved-name reuse, silent recovery, or retry is authorized.

The job uses these pre-existing, non-creating mounts:

| Modal volume | Mount |
| --- | --- |
| `ast-asr-cache` | `/cache` |
| `ast-asr-data` | `/data` |
| `ast-asr-fr-cispo-runs` | `/artifacts` |

The frozen remote inputs are:

```text
/data/fr_cispo_profile/raw/Svarah
/data/fr_cispo_profile/prepared/dataset_manifest.json
/data/fr_cispo_profile/prepared/folds/fold-0.json
/artifacts/profile-h6-refkl-beta0-s2028-20260812/resolved-policy-configs/h6-beta0-fr-cispo.json
/artifacts/profile-h6-refkl-beta0-s2028-20260812/h6-beta0-fr-cispo/rollouts
/artifacts/profile-h6-refkl-beta0-s2028-20260812/h6-beta0-fr-cispo/checkpoint-last-safe
/artifacts/profile-dev-full-sft-20260810/profile-sft-development/checkpoint-epoch-1
```

## Preflight

Before invoking Modal, run these local read-only guards from the repository
root. The first two commands must print the same commit, the next three must
produce no output, and the printed hashes must exactly equal the corresponding
rows above. The authorization commit is outside the executable image scope;
the immutable tag binds it as the exact launch commit without self-reference.

```powershell
git rev-parse HEAD
git rev-parse 'h7-authorized-20260812^{commit}'
git diff --exit-code e4053c632577d22e6d86ccf5848a2686780452f8 -- scripts/modal_h7_sentinel.py src configs uv.lock pyproject.toml experiments/H7-sentinel-kl/input-lock.json
git status --porcelain -- scripts/modal_h7_sentinel.py src configs uv.lock pyproject.toml experiments/H7-sentinel-kl/input-lock.json
git status --porcelain
(Get-FileHash -Algorithm SHA256 scripts/modal_h7_sentinel.py).Hash.ToLower()
uv run --frozen python -c "from pathlib import Path; from ast_asr.modeling import directory_content_hash; print(directory_content_hash(Path('src'))); print(directory_content_hash(Path('configs')))"
```

Before the invocation, perform this read-only volume check. It must fail with
the Modal CLI's missing-path diagnostic; a successful listing means the
reserved run root already exists and the invocation is forbidden. This is
independent of, and deliberately repeated by, the remote wrapper's own
pre-claim check.

```powershell
uvx modal volume ls ast-asr-fr-cispo-runs profile-h7-fixed-policy-sentinel-kl-s2028-20260812
```

The H7 local guard performs the same scoped cleanliness check. Before it claims
the job, the remote wrapper must observe that the reserved **run root**
`/artifacts/profile-h7-fixed-policy-sentinel-kl-s2028-20260812` is absent.
If it exists, it must stop before writing, loading a model, or starting a paid
measurement. This absence check is an input condition, not permission to delete
or overwrite a prior run root.

## Runtime image and source provenance

The image declaration is frozen to the Linux/amd64 manifest
`nvidia/cuda@sha256:09d8951b943dee03cf8fc841b6ea1f201ad33f82f76567171394853c0f494054`
with Python `3.12`; it installs `git`, `ffmpeg`, `libsndfile1`, and
`uv>=0.8,<0.9`, then runs the two frozen `uv sync` commands encoded in the
wrapper against the hashed `pyproject.toml`, `uv.lock`, source package,
configuration, and input lock. Registry evidence is the Docker Registry v2
manifest-list digest for tag `nvidia/cuda:12.8.0-devel-ubuntu22.04`,
`sha256:54f18e2a8e1b3d03f77b9a6dc905533da46ac93a5513f10e8ba8e560db9fa5ab`,
and its selected Linux/amd64 schema-v2 manifest digest above.

This declaration alone is not an unambiguous image identity. At runtime,
`MODAL_IMAGE_ID` must be nonempty. The prelaunch authorization and executable
baseline bind the base-image digest plus the aggregate wrapper, `src/`,
`configs/`, `pyproject.toml`, and `uv.lock` hashes above. The runtime source
manifest records the Modal image ID, clean launch commit, exact command,
Python/Torch/CUDA runtime information, and its per-file source/input manifest.
Together these artifacts map the Modal build identity to the reviewed recipe;
the runtime file does not duplicate every prelaunch aggregate hash. Missing
image identity is terminal and non-evaluable; it is not a reason to relaunch.

## Permitted measurement and Phase A gates

This is a fixed-policy rescore only. The runner must prevalidate and reconstruct
all 28 locked 6-by-4 banks, including the 84 clean/noisy pair records,
deterministic white-noise replay, source-audio hashes, feature hashes, and
attention-mask hashes before the first model forward. It then freezes both
policy and reference in evaluation mode on CUDA FP16; selected log probabilities
and K3 arithmetic are emitted in FP32.

Phase A is cycle 027, scored twice in the original 6-by-4 shape. It may proceed
to the remaining 27 banks exactly once only when all of the following pass:

- policy and reference token IDs and valid-token masks exactly match the saved
  candidates;
- saved old-policy plus rescored reference reproduces K3/token
  `0.10983546078205109` within `1e-7`;
- direct frozen-policy rescoring matches saved valid old log probabilities to
  maximum absolute `1e-5` and gives the same global K3 within `1e-7`;
- the repeated Phase-A score is deterministic within maximum absolute log
  probability `1e-7`; and
- values are finite, valid denominators exist, and no selected `abs(d)` exceeds
  `20`.

No optimizer, gradient/update, rollout generation, decoding, WER computation,
checkpoint selection, controller/beta adjustment, training, or new sampling is
authorized.

## Terminal evidence and interpretation boundary

On success, the immutable output contains source and final manifests,
cycle-027 reproduction evidence, lossless token/candidate/utterance rows,
family-condition/bank/length summaries, and `terminal_decision.json`. It must
commit the output volume once.

After the run root has been claimed, every exception must produce immutable
`failure.json` with the actual phase, exception, source/image/config/input-lock
provenance, command, and expected identities. The Modal wrapper preserves a
runner-written failure and commits it; it writes a launcher-boundary failure
only if none exists. Either failure is terminal,
`measurement_failed`/`non_evaluable`, and consumes the one attempt.

All outputs must retain `publication_valid: false` and
`profile_cluster_count: 115`. They are diagnostic evidence only: no fairness,
robustness, efficacy, publication, or adaptive-controller conclusion follows.
Any recovery requires a separately reviewed, locked, and committed recovery
protocol with a new run and output name.
