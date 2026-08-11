# Group Relative Policy Optimization for Speech Recognition

- **Authors/year:** Prashanth Gurunath Shivakumar, Yile Gu, Ankur Gandhe, Ivan Bulyko (2025).
- **Stable record:** [arXiv:2509.01939](https://arxiv.org/abs/2509.01939); [DOI:10.1109/ASRU65441.2025.11434657](https://doi.org/10.1109/ASRU65441.2025.11434657).
- **Verification/BibTeX:** arXiv plus IEEE ASRU proceedings record; BibTeX fetched from DOI content negotiation.
- **Verified claim:** applies GRPO to ASR using rule-based rewards and reports up to 18.4% relative WER improvement, reduced hallucinations, improved out-of-domain robustness, and domain-adaptation results.
- **Method/results and reported metrics/datasets:** the closest direct precedent for GRPO-style post-training in speech recognition; reports WER, hallucination, OOD robustness, and domain adaptation, but its model/data/reward differ from FR-CISPO.
- **Relevance:** establishes that live RL post-training can be meaningfully tested against ASR WER.
- **Copy / do not copy:** copy live-rollout evaluation, transparent reward specification, and OOD reporting. Do **not** reuse the reported improvement as a target or claim that a rule reward transfers to Indian-English Whisper-tiny.
