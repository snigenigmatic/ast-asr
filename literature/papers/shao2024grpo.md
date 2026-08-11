# DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models

- **Authors/year:** Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, Y. K. Li, Y. Wu, Daya Guo (2024).
- **Stable record:** [arXiv:2402.03300](https://arxiv.org/abs/2402.03300); [official DeepSeek repository](https://github.com/deepseek-ai/DeepSeek-Math).
- **Verification/BibTeX:** arXiv plus author-maintained release repository; BibTeX fetched from `https://arxiv.org/bibtex/2402.03300`.
- **Verified claim:** the work introduces Group Relative Policy Optimization (GRPO) as a PPO variant intended to reduce PPO memory use. The official release states that its RL model was trained using the proposed GRPO method.
- **Method/results and reported metrics/datasets:** math-reasoning LLM RL, with group-relative candidate rewards; the paper reports MATH results rather than speech recognition results.
- **Relevance:** motivates a *baseline arm* with multiple candidates and live rollouts.
- **Copy / do not copy:** copy only a well-instrumented group-relative baseline. Do **not** transfer benchmark gains, reward design, or LLM sequence lengths to ASR.
