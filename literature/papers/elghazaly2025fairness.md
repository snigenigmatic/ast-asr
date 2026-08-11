# Fairness in Automatic Speech Recognition Isn’t a One-Size-Fits-All

- **Authors/year:** Hend ElGhazaly, Bahman Mirheidari, Heidi Christensen, Nafise Sadat Moosavi (2025).
- **Stable record:** [ACL Anthology](https://aclanthology.org/2025.findings-emnlp.1044/); [DOI:10.18653/v1/2025.findings-emnlp.1044](https://doi.org/10.18653/v1/2025.findings-emnlp.1044).
- **Verification/BibTeX:** official ACL record plus Crossref DOI record; BibTeX fetched from DOI content negotiation.
- **Verified claim:** fine-tunes Whisper on Fair-Speech using basic fine-tuning, demographic rebalancing, gender-swapped augmentation, and contrastive learning, then evaluates in-domain and on three OOD sets. Its key result is that the best in-domain fairness method can be the worst OOD outcome; demographic balancing/generalization behavior differs by method.
- **Method/results and reported metrics/datasets:** fairness-as-generalization evaluation over Fair-Speech, LibriSpeech, EdAcc, and CognoSpeak; not Indian-English or RL post-training.
- **Relevance:** supports seen-white-noise and unseen-MUSAN separation, and a requirement that fairness improvements survive a held-out condition.
- **Copy / do not copy:** copy multi-domain utility/fairness evaluation. Do **not** report a training-condition-only gain as robust fairness.
