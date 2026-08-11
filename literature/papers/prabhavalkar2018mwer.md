# Minimum Word Error Rate Training for Attention-Based Sequence-to-Sequence Models

- **Authors/year:** Rohit Prabhavalkar, Tara N. Sainath, Yonghui Wu, Patrick Nguyen, Zhifeng Chen, Chung-Cheng Chiu, Anjuli Kannan (2018; arXiv 2017).
- **Stable record:** [arXiv:1712.01818](https://arxiv.org/abs/1712.01818); [DOI:10.1109/ICASSP.2018.8461809](https://doi.org/10.1109/ICASSP.2018.8461809).
- **Verification/BibTeX:** arXiv plus IEEE proceedings record; BibTeX fetched from DOI content negotiation.
- **Verified claim:** the paper optimizes an approximation to expected word errors for attention seq2seq ASR. It compares sampled candidates with N-best hypotheses and reports the N-best approximation as more effective; its best reported improvement is up to 8.2% relative versus the baseline.
- **Method/results and reported metrics/datasets:** candidate WERs are used to form an expected-risk surrogate, aligning the training signal with the WER evaluation metric on a mobile voice-search task.
- **Relevance:** strongest classical justification for a WER-aligned candidate reward in Whisper post-training.
- **Copy / do not copy:** copy candidate-level, WER-aligned learning and explicit candidate construction. Do **not** copy its result magnitude to a tiny multilingual encoder-decoder model, and do not replace speaker-disjoint evaluation with utterance-only evaluation.
