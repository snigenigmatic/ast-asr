# Architecture

`src/ast_asr` exposes a small set of deep modules:

- `data` and `folds` turn official source evidence into immutable speaker-
  disjoint manifests.
- `modeling`, `sft`, and `inference` own the Whisper/PEFT adapter seam and
  checkpoint round trips.
- `rollouts`, `whisper_policy`, `objectives`, and `optimization` own the frozen-
  rollout/live-policy seam. Objective axes vary through `ObjectiveSpec`; the
  trainer never branches on a heuristic reward.
- `group_weights` owns EMA risk and exponentiated-gradient dual state.
- `evaluation`, `metrics`, and `analysis` own prediction evidence, edit counts,
  out-of-fold validation, and speaker-clustered uncertainty.
- `gates` is the only module that decides whether a learning rate or five-fold
  run is allowed.

The legacy `ast-asr/` directory is deliberately outside the package. Deleting
it cannot alter any FR-CISPO import or command.
