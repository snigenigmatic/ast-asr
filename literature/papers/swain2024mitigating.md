# On Mitigating Performance Disparities in Multilingual Speech Recognition

- **Authors/year:** Monorama Swain, Anna Katrine Van Zee, Anders Søgaard (2024).
- **Stable record:** [ACL Anthology](https://aclanthology.org/2024.emnlp-main.323/); [DOI:10.18653/v1/2024.emnlp-main.323](https://doi.org/10.18653/v1/2024.emnlp-main.323).
- **Verification/BibTeX:** official ACL record plus Crossref DOI record; BibTeX fetched from DOI content negotiation.
- **Verified claim:** compares ERM, LoRA, Group-DRO, spectral decoupling, and adapter fusion for multilingual ASR gender disparities. The paper reports the best performance/parity trade-off for adapter fusion; Group-DRO and spectral decoupling reduced performance with only slightly better parity, while LoRA slightly increased disparities.
- **Method/results and reported metrics/datasets:** Whisper/multilingual fairness-mitigation study across languages, model sizes, and gender, reporting performance and disparity trade-offs.
- **Relevance:** a direct warning against treating LoRA or group weighting as inherently fair.
- **Copy / do not copy:** copy joint utility/parity reporting and ablation mindset. Do **not** add adapter fusion to FR-CISPO without a separate protocol—it would materially change the method.
