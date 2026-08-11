# MUSAN: A Music, Speech, and Noise Corpus

- **Authors/year:** David Snyder, Guoguo Chen, Daniel Povey (2015).
- **Stable record:** [arXiv:1510.08484](https://arxiv.org/abs/1510.08484); [OpenSLR SLR17](https://openslr.org/17/).
- **Verification/BibTeX:** arXiv plus the official dataset release/citation record; BibTeX fetched from `https://arxiv.org/bibtex/1510.08484` and cross-checked against the OpenSLR-provided citation.
- **Verified claim:** MUSAN is a CC-licensed corpus of music, speech, and noise. The paper describes varied technical/nontechnical noises and speech in 12 languages, and demonstrates VAD/music-speech use cases.
- **Method/results and reported metrics/datasets:** data-resource paper with VAD/music-speech demonstrations; it does not establish a particular WER gain for Whisper or Indian English.
- **Relevance:** provides an external, unlabeled corruption source for an **unseen** babble test condition.
- **Copy / do not copy:** copy reproducibly pinned noise audio and SNR-controlled mixing. Do **not** transcribe MUSAN, leak MUSAN babble into training if it is the unseen test condition, or call white-noise success general robustness.
