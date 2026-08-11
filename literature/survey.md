# FR-CISPO verified literature survey

This library supports a narrow claim: **FR-CISPO is an experimental integration of
sequence-level post-training, group-aware optimization, and paired noise exposure
for a tiny Indian-English ASR model.** It does not establish that the integration
works. In particular, none of the cited language-model RL papers is evidence that
the same update is stable or beneficial for Whisper-tiny ASR.

## How this ledger was verified

Each entry has (1) a primary paper or official publisher record, (2) an independent
bibliographic/publisher record where available, and (3) BibTeX fetched from the
listed DOI resolver, ACL Anthology export, OpenSLR citation record, or arXiv BibTeX
endpoint. Claims below were checked against the paper abstract/full paper or the
official publisher record—not inferred from citation metadata. Semantic Scholar
returned rate-limit responses during this audit, so it is not used as evidence.

`Primary + official repository` is used only for author-published preprints that
have no separate proceedings record in Crossref. Those rows are clearly marked.

| Key | Area | Verification records | Decision for FR-CISPO |
|---|---|---|---|
| [radford2022whisper](papers/radford2022whisper.md) | Whisper | arXiv + OpenAI paper PDF | Use `whisper-tiny` as the fixed base and report additive-noise results separately. |
| [javed2023svarah](papers/javed2023svarah.md) | Indian-English benchmark | arXiv + INTERSPEECH DOI | Use only after authoritative speaker IDs are available; 117 speakers is a hard data gate. |
| [prabhavalkar2018mwer](papers/prabhavalkar2018mwer.md) | MWER ASR | arXiv + IEEE DOI | Keep WER-aligned, candidate-level rewards; compare N-best/candidate sampling choices explicitly. |
| [schulman2017ppo](papers/schulman2017ppo.md) | off-policy clipping | arXiv + official OpenAI Spinning Up record | Treat clipping/KL as safeguards, not proof of ASR improvement. |
| [shao2024grpo](papers/shao2024grpo.md) | GRPO | arXiv + DeepSeek official repository | Use group-relative candidates only with live rollout regeneration and logged policy drift. |
| [zheng2025gspo](papers/zheng2025gspo.md) | sequence ratios | arXiv + official Qwen technical note | Test sequence ratios as a hypothesis because the reward is sequence-level; do not import LLM results as ASR evidence. |
| [minimax2025m1](papers/minimax2025m1.md) | CISPO | arXiv + MiniMax official repository | Copy only the clipped, stop-gradient importance-weight idea; retain an ASR-specific ablation. |
| [shivakumar2025grpoasr](papers/shivakumar2025grpoasr.md) | GRPO for ASR | arXiv + IEEE ASRU DOI | Most directly relevant RL-ASR precedent; reproduce its live-rollout discipline, not its task/model claims. |
| [sagawa2020gdro](papers/sagawa2020gdro.md) | worst-group optimization | arXiv + ICLR/OpenReview publication record | Use worst-group WER as endpoint and regularize/early-stop; do not claim dual weights alone solve generalization. |
| [liu2022fairasr](papers/liu2022fairasr.md) | ASR fairness measurement | arXiv + IEEE DOI | Use speaker-clustered analysis and avoid unadjusted demographic claims. |
| [koenecke2020racial](papers/koenecke2020racial.md) | ASR disparity audit | PNAS DOI + PubMed Central | Treat group WER gaps as an auditing outcome, not demographic parity. |
| [vanzee2024group](papers/vanzee2024group.md) | multilingual Whisper fairness | ACL Anthology + Crossref DOI | Report worst-group/intersectional results instead of relying on aggregate WER. |
| [swain2024mitigating](papers/swain2024mitigating.md) | fairness mitigation | ACL Anthology + Crossref DOI | Do not assume LoRA or Group-DRO narrows gaps; include utility and disparity together. |
| [elghazaly2025fairness](papers/elghazaly2025fairness.md) | fairness under shift | ACL Anthology + Crossref DOI | Preserve clean and unseen-noise evaluation; reject an arm that only improves its seen condition. |
| [snyder2015musan](papers/snyder2015musan.md) | noise robustness | arXiv + OpenSLR official dataset record | Keep MUSAN babble unseen during training; train with white noise only if that protocol remains locked. |
| [hu2022lora](papers/hu2022lora.md) | efficient adaptation | arXiv + ICLR/OpenReview paper | LoRA is a resource-control choice, not a fairness intervention; rank/module choices need validation. |

## Training implications, ranked by evidence proximity

1. **Most direct:** MWER and the GRPO-for-ASR paper justify testing WER-aligned,
   live-rollout post-training. Neither validates FR-CISPO's fairness weighting.
2. **Method transfer only:** GSPO and CISPO motivate the sequence-ratio and
   clipped-weight ablations, but originate in language-model reasoning rather than
   encoder-decoder ASR.
3. **Evaluation guardrails:** Svarah, Koenecke, van Zee, Liu, Swain, and
   ElGhazaly support speaker-aware, worst-group, cross-condition reporting. They
   do not authorize a claim of demographic fairness from a non-significant test.
4. **Robustness guardrail:** Whisper and MUSAN support condition-specific noise
   evaluation. A gain on seen white noise alone is insufficient; unseen MUSAN
   babble must remain a held-out robustness check.

## Explicit non-transfers

- Do not copy MiniMax-M1 compute scale, its large-model benchmarks, or claim its
  CISPO results transfer to ASR.
- Do not call family-weighted WER "demographic parity" or "equal opportunity".
- Do not use the current 115 profile clusters as if they were Svarah's documented
  117 speaker identities.
- Do not select a policy arm from a single seed or a single seen-noise metric.

The per-paper notes record exact evidence, what the source measured, and which
part—if any—is safe to carry into the next protocol.
