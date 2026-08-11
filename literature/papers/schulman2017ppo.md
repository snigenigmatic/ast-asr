# Proximal Policy Optimization Algorithms

- **Authors/year:** John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, Oleg Klimov (2017).
- **Stable record:** [arXiv:1707.06347](https://arxiv.org/abs/1707.06347); [OpenAI Spinning Up PPO documentation](https://spinningup.openai.com/en/latest/algorithms/ppo.html).
- **Verification/BibTeX:** primary arXiv paper plus official OpenAI algorithm record; BibTeX fetched from `https://arxiv.org/bibtex/1707.06347`.
- **Verified claim:** PPO introduces a clipped surrogate objective intended to constrain harmful policy changes while retaining first-order optimization.
- **Method/results and reported metrics/datasets:** generic RL method; the source reports control-task experiments, not ASR or WER.
- **Relevance:** supports logging ratios/KL and applying a predeclared trust-region gate when reusing rollout data.
- **Copy / do not copy:** copy the discipline of measuring update size and stopping unsafe updates. Do **not** cite PPO as evidence that any particular clipping threshold, KL estimator, or policy-learning rate is valid for Whisper ASR.
