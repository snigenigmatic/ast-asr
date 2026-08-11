# Group Sequence Policy Optimization

- **Authors/year:** Chujie Zheng, Shixuan Liu, Mingze Li, Xiong-Hui Chen, Bowen Yu, Chang Gao, Kai Dang, Yuqiong Liu, Rui Men, An Yang, Jingren Zhou, Junyang Lin (2025).
- **Stable record:** [arXiv:2507.18071](https://arxiv.org/abs/2507.18071); [official Qwen technical note](https://qwenlm.github.io/blog/gspo/).
- **Verification/BibTeX:** arXiv plus the Qwen team technical record; BibTeX fetched from `https://arxiv.org/bibtex/2507.18071` (the Qwen note publishes the same citation).
- **Verified claim:** GSPO moves importance-ratio calculation, clipping, reward assignment, and optimization to the sequence level. It uses sequence likelihood with length normalization, rather than independent token ratios, to match a sequence-level reward.
- **Method/results and reported metrics/datasets:** LLM/RLVR work; reports improved stability/efficiency/performance relative to GRPO, notably for MoE training. It provides no Whisper or ASR evidence.
- **Relevance:** directly motivates the sequence-ratio ablation because the FR-CISPO reward is utterance-WER based.
- **Copy / do not copy:** copy the unit-consistency hypothesis and length-normalized sequence-ratio audit. Do **not** claim GSPO validates FR-CISPO or omit token-ratio and no-RL controls.
