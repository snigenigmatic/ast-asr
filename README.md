# Capstone — Fair & Robust Audio Transformers
## PW25_BJD_05 | PES University

### Project structure
```
capstone/
├── src/
│   ├── data_loader.py       # Svarah loader + accent→family mapping
│   ├── asr_inference.py     # Whisper / Wav2Vec2 / HuBERT inference
│   ├── noise_augment.py     # White / pink / babble noise at target SNR
│   ├── fairness_metrics.py  # ΔDP, ΔEO, Δg_noise, Poisson test
│   └── pipeline.py          # Orchestrator + CLI
├── outputs/                 # CSVs written here
└── cache/                   # HuggingFace dataset cache
```

---

### Setup (on your CUDA machine)
```bash
uv sync
```

---

### Running the pipeline

#### Quick smoke test (50 utterances, Whisper-tiny)
```bash
uv run src/pipeline.py --model whisper-tiny --max-samples 50
```

#### Full Svarah evaluation
```bash
uv run src/pipeline.py --model whisper-small
```

#### Compare multiple models (bash loop)
```bash
for MODEL in whisper-tiny whisper-small wav2vec2-base hubert-large; do
    uv run src/pipeline.py --model $MODEL --output outputs/results_${MODEL}.csv
done
```

#### Disaggregate by gender instead of language family
```bash
uv run src/pipeline.py --model whisper-small --group gender
```

---

### Outputs
Each run produces two files in `outputs/`:
- `results_{model}_clean.csv` — per-utterance: uid, accent, language_family, gender, reference, hypothesis
- `summary_{model}.csv` — ΔDP, ΔEO, max_noise_gap, Poisson p-value, per-group WERs

---

### Fairness metrics (from our position paper)

| Metric | Formula | Threshold |
|--------|---------|-----------|
| ΔDP | max\_{i,j} \|WER(gᵢ) − WER(gⱼ)\| | > 5 pp = flagged |
| ΔEO | max\_{i,j} \|TPR(gᵢ) − TPR(gⱼ)\| | — |
| Δg\_noise | WER(gᵢ, 0dB) − WER(gᵢ, clean) | per group |
| Poisson | Drop-in-deviance χ² test | p < 0.05 = systematic |

---

### Next steps (implementation roadmap)
1. **Baseline sweep** — run all 4 models on full Svarah, fill Table 3 gaps
2. **Noise robustness** — add SPIRE-SIES noise clips when access confirmed
3. **LoRA fine-tuning** — `peft` adapter on Wav2Vec2, re-evaluate ΔDP
4. **Adversarial de-biasing** — gradient reversal branch (Ganin et al. 2016)
5. **AccentDB integration** — secondary evaluation dataset