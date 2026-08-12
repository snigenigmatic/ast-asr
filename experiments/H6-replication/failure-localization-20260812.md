# H6 sampled-K3 safety-stop localization — 2026-08-12

**Scope:** offline analysis of read-only copies of the immutable H5/H6 Modal
cycle diagnostics and rollouts. This does not amend the locked H6 protocol,
gate, or result, and it does not authorize another run. The downloaded
`h6_s2028_beta0/diagnostics/cycle-027.json` reproduces the recorded SHA-256
`5c267d055bcfd80a041b97bbb9e28dbb18aec1bbbdefa1bde55bb8ab5baeba45`.

Run the analysis with:

```powershell
uv run --frozen python experiments/H6-replication/analyze_kl_failure.py `
  --artifacts-root <read-only-modal-download-root> `
  --output <local-output.json>
```

The checked artifact root has five children: `h5_s2026_beta0`,
`h5_s2026_beta004`, `h6_s2027_beta0`, `h6_s2027_beta004`, and
`h6_s2028_beta0`. Each contains matching `diagnostics/cycle-*.json` and
`rollouts/cycle-*.json` files.

The script emits a deterministic input manifest containing every relative
path, byte size, and SHA-256, bound to its immutable Modal run/output path. The
376 checked input files produce:

| Local arm label | Immutable Modal volume path | Diagnostics / rollouts | Manifest SHA-256 |
| --- | --- | ---: | --- |
| `h5_s2026_beta0` | `/artifacts/profile-h5-refkl-beta0-r1-s2026-20260812/h5-beta0-fr-cispo` | 40 / 40 | `1b8f648fac4d4dc68a0aba80698e494a2ee0572f1042a58518c5f54060cd9448` |
| `h5_s2026_beta004` | `/artifacts/profile-h5-refkl-beta004-s2026-20260812/h5-beta004-fr-cispo` | 40 / 40 | `d97e6ef670fc638d963863ac21d17dec3592205d30e4737a23c136e03cb8b945` |
| `h6_s2027_beta0` | `/artifacts/profile-h6-refkl-beta0-s2027-20260812/h6-beta0-fr-cispo` | 40 / 40 | `b337a824c90c1bb90348cce4c63bce07902047dbf0803543332cd113204b0598` |
| `h6_s2027_beta004` | `/artifacts/profile-h6-refkl-beta004-s2027-20260812/h6-beta004-fr-cispo` | 40 / 40 | `abe6fc941cd15e9287d4dc220ec5b1e35cd86ba4c2b97cb3986a38dae11b1b2b` |
| `h6_s2028_beta0` | `/artifacts/profile-h6-refkl-beta0-s2028-20260812/h6-beta0-fr-cispo` | 28 / 28 | `92c5b7c7fb3457ca41da462fd8475d8a29694541f9f7ae2137fe989f10118c7e` |

## What the stored telemetry can and cannot answer

Each cycle stores the pre- and post-update sampled fixed-SFT K3, five trajectory
points (update zero plus four post-optimizer records) for
loss/K3/ratio/gradient, the six family ×
`{clean, white_train}` risks and dual probabilities, and six utterances with
four candidates each, including candidate WERs and token masks. This supports
batch-level localization by cycle, group weight, risk, candidate length, and
candidate WER.

It does **not** store per-token, candidate, utterance, or group K3
contributions; current-versus-reference token log probabilities; a fixed
sentinel candidate bank rescored at every cycle; or optimizer/gradient
contributions partitioned by group. Therefore no saved artifact can identify
which group or token caused K3, and the correlations below are descriptive,
not causal tests.

## Immediate failure boundary

The seed-2028 beta-zero run tripped at cycle 27 at post-update K3
`0.11095129698514938`. Crucially, the same new frozen rollout already measured
`0.10983546078205109` at update zero; the four inner updates added only
`0.0011158362030982971`. Thus the *immediate* crossing is not attributable to
the cycle-27 optimizer updates, their ratio movement, or that cycle's newly
computed dual weights.

The previous cycle ended at `0.07646089792251587`, but it used a different
fresh rollout. The apparent adjacent-cycle increase is `+0.034490399062633514`
and cannot be interpreted as a same-candidate-bank model jump. This distinction
matters: cycle 23 measured `0.09991230070590973` on 134 valid candidate tokens,
then cycle 24 measured `0.03693665564060211` on 486 tokens. The largest
adjacent decrease over the failed run was `-0.06297564506530762`.

The final cycle's ratio p99 was only `1.0187872648239136`, well below the 2.0
cap. This was a sampled fixed-reference-K3 stop, not a rollout/current-ratio
instability.

## Batch composition signal

At cycle 27, the 24 candidates contained 188 valid tokens (mean `7.8333`),
below the failed run's median cycle total of 256 tokens and mean cycle total of
309.96. Across the 28 seed-2028 cycles, post-cycle K3 versus mean candidate
length had descriptive Pearson `-0.497` and linear-cycle-detrended Pearson
`-0.685`. The same detrended direction appears in the two completed beta-zero
controls and the observed prefix of the failed seed-2028 beta-zero control:

| Arm | Cycles | K3 maximum | Detrended correlation: K3 vs mean valid candidate tokens |
| --- | ---: | ---: | ---: |
| H5 seed 2026 beta 0 | 40 | 0.02491910 | -0.425 |
| H6 seed 2027 beta 0 | 40 | 0.07627828 | -0.688 |
| H6 seed 2028 beta 0 | 28 | 0.11095130 | -0.685 |

This is a repeated negative descriptive **candidate-length association** for
the sampled estimator, not proof that short utterances cause divergence. The
missing per-token K3 terms prevent separating estimator variance from a
genuine short-utterance policy effect.

## Group-weight and risk checks

At the failing cycle, the dual probabilities ranged from `0.127145` to
`0.211164`; Sino-Tibetan × white was the maximum at `0.211164` (only 1.27× the
uniform `1/6`). The associated per-utterance loss weights ranged from
`0.762870` to `1.266982`. The dual therefore did not collapse onto a group.

Raw K3/probability correlation is high because both quantities trend with cycle
number. After linear cycle detrending in the failed beta-zero run, correlation
with the maximum group probability is `0.006` and with Sino-Tibetan × white
probability is `0.052`; correlation with maximum observed group risk is
`0.092`. These recorded linear batch summaries give no evidence that escalating
dual concentration or maximum observed risk tracked K3 after detrending. They
cannot exclude nonlinear, lagged, or unobserved per-group relationships.

## Defensible conclusion

The record supports a narrow conclusion: the H6 stop was a fail-closed
candidate-conditioned sampled-K3 threshold crossing that was already present
before the final cycle's optimizer updates. It is associated with shorter
candidate batches, occurred with a low rollout/current ratio p99, and shows no
corresponding concentration in the recorded dual weights. Recorded linear risk
summaries do not identify its cause. It does **not** establish a causal failure
mechanism or weaken the locked `failed_safety` decision.

Any future, separately preregistered safety study would need a fixed held-out
sentinel candidate bank and persisted per-token/per-utterance K3 contributions
before it could distinguish policy drift from candidate-bank measurement
variation.
