# Robust Speech Recognition via Large-Scale Weak Supervision

- **Authors/year:** Alec Radford, Jong Wook Kim, Tao Xu, Greg Brockman, Christine McLeavey, Ilya Sutskever (2022).
- **Stable record:** [arXiv:2212.04356](https://arxiv.org/abs/2212.04356); [official OpenAI PDF](https://cdn.openai.com/papers/whisper.pdf).
- **Verification/BibTeX:** primary arXiv record plus author-hosted OpenAI paper; BibTeX fetched from `https://arxiv.org/bibtex/2212.04356`.
- **Verified claim:** Whisper was trained on 680,000 hours of multilingual, multitask weak supervision and is evaluated zero-shot. Its paper separately evaluates additive white and pub noise, so noise robustness must be reported by condition rather than assumed from clean WER.
- **Method/results and reported metrics/datasets:** foundation-model speech recognition, not fairness optimization; reports benchmark and noise-condition WER comparisons against LibriSpeech-trained systems.
- **Relevance:** fixes the base model and motivates clean/white-noise/unseen-babble evaluation.
- **Copy / do not copy:** copy a fixed base revision and condition-specific WER tables. Do **not** claim Whisper's pretraining robustness transfers to Indian English or that its benchmark gains establish FR-CISPO.
