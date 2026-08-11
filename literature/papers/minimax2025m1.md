# MiniMax-M1: Scaling Test-Time Compute Efficiently with Lightning Attention

- **Authors/year:** MiniMax (2025; full author list is retained in the arXiv record).
- **Stable record:** [arXiv:2506.13585](https://arxiv.org/abs/2506.13585); [official MiniMax repository](https://github.com/MiniMax-AI/MiniMax-M1).
- **Verification/BibTeX:** arXiv plus author-maintained model/research repository; BibTeX fetched from the official repository citation block and cross-checked with `https://arxiv.org/bibtex/2506.13585`.
- **Verified claim:** the authors introduce CISPO, described as clipping importance-sampling weights rather than token updates. Their public description frames it as part of a large-scale RL system for a hybrid-attention MoE reasoning model.
- **Method/results and reported metrics/datasets:** LLM reasoning/report-scale RL; the source reports its own benchmark comparisons and infrastructure scale, not speech recognition or fairness outcomes.
- **Relevance:** source for the narrow clipped, stop-gradient importance-weight primitive.
- **Copy / do not copy:** copy only the explicit ablation target—upper-capped importance weights outside the advantage centering. Do **not** import MiniMax compute claims, architecture, reward tasks, or a claim that CISPO has been validated for ASR.
