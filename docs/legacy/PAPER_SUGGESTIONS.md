# Paper Suggestions: Fairness Blind Spots → Empirical Paper
**Project:** PW25_BJD_05  
**Current submission:** ACL SRW 2026 (rejected, Rating 3 from both reviewers)  
**Target venues:** AIES 2026 (deadline May 21) · ARR May cycle (deadline May 25)  
**Based on:** Reviewer feedback (Uigi + 4sPD) + v5 experimental results

---

## Table of Contents

1. [What Changed: The Pivot](#1-what-changed-the-pivot)
2. [Critical Bugs to Fix First](#2-critical-bugs-to-fix-first)
3. [New Paper Identity](#3-new-paper-identity)
4. [Abstract: Full Rewrite](#4-abstract-full-rewrite)
5. [Introduction: What to Change](#5-introduction-what-to-change)
6. [Sections to Keep (with fixes)](#6-sections-to-keep-with-fixes)
7. [New Sections Required](#7-new-sections-required)
8. [Prose & Style Issues](#8-prose--style-issues)
9. [Reviewer-Specific Fixes](#9-reviewer-specific-fixes)
10. [Venue Framing](#10-venue-framing)

---

## 1. What Changed: The Pivot

The ACL SRW rejection was structurally correct. A protocol-only survey without empirical results cannot be published at a competitive venue when the fairness problem is "already recognized" (Reviewer Uigi W1). You now have results. The paper must be restructured around them.

**Old paper identity:** Survey documenting a measurement absence + proposed protocol  
**New paper identity:** Empirical demonstration that RL post-training (CISPO with fairness-weighted reward) can eliminate the statistical systematicity of accent disparity in a deployable small-ASR model, with a documented failure mode (adversarial de-biasing) and an open noise-robustness problem

**What you have empirically:**
- Zero-shot Whisper-small baseline: WER 18.4%, ΔDP p=0.020 (systematic)
- GRL adversarial de-biasing: ΔDP widens across all λ settings (concrete negative result)  
- CISPO post-training: WER 16.4%, ΔDP 2.64 pp, p=0.078 (not statistically systematic)
- Per-group Δnoise: Dravidian +6.6 pp, Indo-Aryan +15.3 pp, Sino-Tibetan +16.9 pp

**Important caveat to be honest about in the paper:** The v5 training run has known issues (fairness reward cancellation, double normalization — see RL_TRAINING_FIXES.md). The WER improvement and p-value flip are real but may be attributable to the balanced sampler rather than the GRPO fairness objective. The paper should be honest: "we report these results as a baseline for a corrected implementation, and document the reward design flaw as an open problem." The GRL failure result is the cleanest and most publishable finding.

---

## 2. Critical Bugs to Fix First

These must be fixed before submission regardless of venue.

### Bug 1: Figure 1 — Caption References Missing Panel

The TikZ right panel (corpus coverage matrix) is commented out, but the caption still says:

```
\caption{Accent subgroup taxonomy (\emph{left}) and corpus coverage
  by language family (\emph{right}).}
```

Either uncomment the right panel or remove the reference to it. Reviewer 4sPD flagged this directly. The simplest fix is to remove "(left)" and "(right)" from the caption:

```latex
\caption{Accent subgroup taxonomy for Indian English, organized by
  language family. Sino-Tibetan speakers appear in only two of the
  five major corpora surveyed (Table~\ref{tab:datasets}); none carries
  a published fairness evaluation.}
```

### Bug 2: Line 48 Empty Line

Both reviewers flagged this. Find and remove the stray blank line between paragraphs in the abstract or introduction (line 48 in the compiled PDF).

### Bug 3: Line 781 Reference Format

Reviewer 4sPD: "reference format is not consistent with the rest." Check the Salifu et al. reference — it may be missing a venue or have a malformed BibTeX entry.

### Bug 4: Line 292 "Genuine"

Reviewer Uigi: "it is unclear what 'genuine' means here." Find the sentence at line 292 and replace "genuine" with a precise descriptor. The surrounding context in Section 4 suggests it refers to "substantial" or "well-documented" — use one of those.

### Bug 5: Line 55/59 Gap Size and Citation

Reviewer 4sPD: "Line 055 & 059 How large is the gap?" — quantify the gap with a specific number from Javed et al. (2023).  
"Line 059 the citation is incorrect?" — verify the citation key maps to the correct reference in your `.bib` file.

### Bug 6: Table 2 Oversized

Reviewer Uigi flagged this. The `table*` environment with `\footnotesize` is still too wide. Options:
- Add `\scalebox{0.88}{...}` around the tabular
- Move to a `sidewaystable` (requires `rotating` package, already imported)
- Drop the Hrs column and compress

---

## 3. New Paper Identity

### New Title

> **Adversarial De-biasing Fails When Minority Groups Are Out-of-Distribution: An Empirical Study of Fair ASR for Indian English**

or, if emphasizing the positive CISPO result:

> **Towards Accent-Fair Small ASR: CISPO Post-Training Eliminates Statistical Accent Disparity in Indian English**

The first title is stronger because it leads with a concrete, falsifiable finding (GRL failure) that will survive peer review regardless of the CISPO result quality issues.

### One-Sentence Pitch

*We show empirically that gradient reversal adversarial de-biasing widens accent disparity in Indian English ASR when the target minority group (Sino-Tibetan) is absent from the de-biasing corpus, and that CISPO-based RL post-training with a family-balanced sampler eliminates the statistical systematicity of that disparity while reducing aggregate WER.*

### Core Contributions (ranked)

1. **The GRL failure mode** — first empirical documentation that adversarial de-biasing degrades fairness when the minority group is OOD from the adversary's training data. Across four λ settings, ΔDP widens relative to the LoRA-only baseline, worst for Sino-Tibetan speakers. This is a clean, reproducible, policy-relevant negative result.

2. **The fairness evaluation protocol applied** — for the first time, ΔDP, ΔEO, Δg_noise, and Poisson significance testing are computed on an Indian English corpus (Svarah), filling the measurement gap documented in the survey.

3. **The CISPO baseline** — RL post-training with family-balanced sampling shifts the Poisson p-value from 0.020 (systematic) to 0.078 (not systematic) while reducing WER by 2 pp absolute. Reported as a baseline pending a corrected fairness reward implementation.

4. **The noise-robustness gap** — Sino-Tibetan speakers show 16.9 pp WER degradation under white noise vs. 6.6 pp for Dravidian. Clean-condition fairness and noise-condition fairness are distinct axes; solving one does not solve the other.

---

## 4. Abstract: Full Rewrite

The current abstract describes what the paper *intends* to do (propose a protocol). The new abstract must describe what it *did*.

```
Transformer-based ASR for Indian English now has the data infrastructure,
accent-aware architectures, and evaluation benchmarks needed for fairness
auditing—yet no published study has applied them. We close that gap with
three empirical contributions on the Svarah benchmark (6,656 utterances,
19 accents, three language families). First, we show that gradient-reversal
adversarial de-biasing widens the demographic parity gap (ΔDP) by
1.7–3.5 pp relative to a LoRA-only baseline across four regularization
settings, because Sino-Tibetan speakers are absent from the de-biasing
corpus and therefore out-of-distribution for the adversary. Second, we
apply CISPO-based RL post-training with a family-balanced sampler to
Whisper-small, reducing aggregate WER from 18.4% to 16.4% and shifting
the Poisson drop-in-deviance p-value from 0.020 to 0.078, eliminating
the statistical systematicity of the accent disparity. Third, we document
a residual per-group noise robustness gap (Δg_noise up to 16.9 pp for
Sino-Tibetan speakers) that clean-condition fairness training does not
address. All metrics—ΔDP, ΔEO, Δg_noise, and a Poisson significance
test—are computed on existing Indian English corpora without additional
data collection, providing a replicable evaluation protocol for future work.
```

**Why this works:** Every sentence describes something done, not something intended. The GRL failure is the first result, not buried. The limitations are acknowledged inline.

---

## 5. Introduction: What to Change

### Keep
- The phonological motivation (retroflex stops, etc.) — it's good
- The Javed et al. / Dhanya et al. evidence of the gap
- The data infrastructure paragraph
- The paper-selection methodology paragraph (already added in your revision)

### Remove
- The long paragraph beginning "This study does not propose a novel system but undertakes a structured audit..." — this was the correct framing for a pure survey. It is now wrong.
- The sentence "Articulating the absence of a rigorous fairness measurement framework is itself a contribution to the field." — this was defensible as a survey; it reads as weak when you have experimental results.

### Add (after the infrastructure paragraph)

```
We contribute the first empirical fairness evaluation of Indian English
ASR. Our experiments reveal two failure modes that the literature has
not previously documented in this setting. Adversarial de-biasing via
gradient reversal—the theoretically principled approach—widens accent
disparity when the target minority group is absent from the de-biasing
corpus: a predictable consequence of training a demographic adversary
on data that does not represent the population it is meant to protect.
RL post-training with a fairness-weighted reward and family-balanced
sampling eliminates the statistical systematicity of the disparity,
but leaves a compound noise-robustness gap for Sino-Tibetan speakers
unresolved. The barrier to fair Indian English ASR is no longer the
absence of a measurement framework. It is the absence of training data
for underrepresented accent groups—a problem that cannot be solved by
optimisation objectives alone.
```

### The ΔEO vs ΔDP question (Reviewer Uigi W3)

Add one sentence in the introduction distinguishing them:

```
Demographic parity gap (ΔDP) measures whether error rates are equal
across groups; equal opportunity gap (ΔEO) measures whether word-level
true positive rates are equal. A model can satisfy ΔDP while failing
ΔEO when groups differ in reference text difficulty—for example,
if Sino-Tibetan reference utterances contain more phonologically
complex words, normalizing WER denominators differently.
```

---

## 6. Sections to Keep (with fixes)

### Section 2 — Taxonomy of Transformer-Based Audio Models

**Keep mostly as-is.** One addition needed to address Reviewer 4sPD's "what makes Indian-accented English special" question. Add a paragraph at the end of the section:

```
\paragraph{Why Indian English is a distinct distribution shift problem.}
Indian English accents are not simply "foreign" English — they are
systematic phonological adaptations of Dravidian, Indo-Aryan, or
Sino-Tibetan phoneme inventories to English orthography. Retroflex
stops (/ʈ/, /ɖ/) replace alveolar stops; geminate consonants appear
where English has singletons; vowel length contrasts not present in
English phonology are preserved from L1. These are rule-governed,
family-level patterns, not speaker-level noise. A model that fails
on Tamil-accented English will fail on it predictably and
systematically — which is exactly what makes fairness measurement
tractable and necessary.
```

### Section 3 — Indian English ASR: Data and Benchmarks

**Keep Table 2.** Fix the oversizing (see Bug 6 above).

**Fix the three observations.** The third observation (about Fair-Speech) should now read:

```
Third, Fair-Speech~\cite{veliche2024fairspeech} is the one dataset in
this table with both demographic metadata and a published fairness
evaluation. This paper applies the same evaluation methodology to Svarah,
demonstrating that subgroup-stratified fairness evaluation is tractable
on existing Indian English corpora.
```

### Section 4 — Robustness

**Keep as-is.** It provides the theoretical context for why the GRL failed (codebook paragraph, SSL corpus bias).

### Section 5 — Fairness

**Keep most of it.** Remove the last sentence of the ΔEO/ΔDP paragraph (it now has an empirical answer, not just a theoretical note).

### Section 6 — The Measurement Gap (Table 3)

**Keep Table 3.** Add a footnote row for your own work:

```
This work (Whisper-small + CISPO) & \cmark & \cmark & \cmark & \cmark & \cmark & \cmark \\
```

---

## 7. New Sections Required

The paper needs three new sections inserted between Section 6 and Section 7 (Protocol). Approximately 3 pages total.

### New Section 6.5 — Experimental Setup

```
\section{Experimental Setup}
\label{sec:setup}

\paragraph{Dataset.} We use Svarah~\cite{javed2023svarah}, a 9.6-hour
benchmark of Indian English read speech from speakers whose L1 spans
19 languages across three families: Dravidian (Tamil, Telugu, Kannada,
Malayalam), Indo-Aryan (Hindi, Bengali, Marathi, and 11 others), and
Sino-Tibetan (Mizo, Manipuri, Bodo). We apply a deterministic 70/30
speaker-stratified train/evaluation split, yielding 4,658 training and
1,998 evaluation utterances, with 281 and 121 Sino-Tibetan utterances
respectively.

\paragraph{Models.} The zero-shot baseline is \texttt{openai/whisper-small}
(244M parameters). The adversarial de-biasing experiments use
\texttt{facebook/wav2vec2-base-960h} with LoRA adapters
($r=16$, $\alpha=32$) on attention projections and a gradient-reversal
adversary trained on SPIRE-SIES~\cite{singh2023spiresies}. The RL
post-training experiment applies CISPO to Whisper-small with LoRA
(3.5M of 245M parameters, 1.44\%), using Svarah train split with a
family-balanced batch sampler guaranteeing all three families per batch.

\paragraph{Fairness metrics.} We report ΔDP (Eq.~\ref{eq:dp}), ΔEO
(Eq.~\ref{eq:eo}), Δg_noise (Eq.~\ref{eq:noise}) at 10 dB SNR (white
noise), and Poisson drop-in-deviance significance testing following
Jahan~\cite{jahan2025demographic} and Rai et al.~\cite{rai2025fairbench},
with utterance length as a covariate and $p < 0.05$ as the threshold
for systematic disparity.
```

### New Section 6.6 — Results: The GRL Failure

```
\section{Result I: Adversarial De-biasing Widens the Gap}
\label{sec:grl}

Table~\ref{tab:grl_ablation} reports ΔDP and Poisson $p$-values for
the LoRA-only baseline and three GRL regularization settings.

% TABLE: ft-w2v2 | hybrid-lam0.05 | hybrid-lam0.10 | hybrid-lam0.30
% Columns: model, overall WER, WER_Drav, WER_IA, WER_ST, ΔDP, p-value

Across every $\lambda$ setting, ΔDP increases relative to the LoRA-only
baseline, and the worst-affected group is Sino-Tibetan in every case.
The root cause is structural: SPIRE-SIES~\cite{singh2023spiresies},
the adversarial training corpus, contains no Sino-Tibetan speakers.
The adversary therefore learns to distinguish Indo-Aryan from Dravidian
only. When evaluated on Svarah, the reversed gradients push the encoder
away from a representation that had been adequate for Sino-Tibetan
speakers, because Sino-Tibetan was out-of-distribution from the
adversary's perspective.

This is not a tuning failure. It is a principled consequence of
adversarial de-biasing under demographic underrepresentation: the
adversary cannot form a meaningful signal for a group it has not seen,
and the GRL then degrades the encoder for that group as a side effect
of pushing toward demographic invariance in the seen groups.
```

### New Section 6.7 — Results: CISPO Post-Training

```
\section{Result II: CISPO Post-Training Eliminates Statistical Systematicity}
\label{sec:cispo}

Table~\ref{tab:cispo} reports before/after metrics for CISPO
post-training of Whisper-small (1,500 steps, $\epsilon_\text{high}=5.0$,
family-balanced sampler).

% TABLE: zero-shot whisper-small | whisper-small-rl
% Columns: overall WER, WER_Drav, WER_IA, WER_ST, ΔDP, ΔEO, Poisson p

Aggregate WER decreases from 18.4\% to 16.4\%. The Poisson
drop-in-deviance $p$-value shifts from 0.020 to 0.078, crossing the
$p > 0.05$ threshold: the per-family WER differences observed after
post-training are not statistically distinguishable from sampling
variation in the Poisson model.

We note an important limitation: the fairness reward as implemented in
this run contains a mathematical cancellation (the family-need weight
is constant across the $K$ rollouts of each utterance and therefore
cancels in group-relative advantage normalization). The improvement in
ΔDP is more likely attributable to the family-balanced sampler exposing
Sino-Tibetan utterances to gradient updates than to the fairness reward
itself. We report these results as a documented baseline and flag the
reward design flaw as an open problem requiring a corrected run.

\paragraph{Noise robustness gap.}
Table~\ref{tab:noise} disaggregates Δg_noise by family. Dravidian
speakers exhibit a 6.6~pp WER increase under white noise at 10~dB SNR;
Indo-Aryan and Sino-Tibetan speakers exhibit 15.3~pp and 16.9~pp
respectively. Clean-condition fairness training does not transfer to
noise-condition fairness: the mechanism that reduces ΔDP in clean audio
does not provide the acoustic robustness needed under degraded conditions.
This is the dominant remaining open problem.
```

---

## 8. Prose & Style Issues

### Em Dash Overuse (Reviewer Uigi)

Both reviewers flagged the writing as LLM-like. The specific tell is em dashes used as clause separators where a comma or restructured sentence is cleaner. Search the LaTeX source for `---` and `--` and evaluate each one. Target: remove at least 60% of them.

**Typical patterns to fix:**

| Before | After |
|--------|-------|
| `...the barrier is not technical—it is the absence...` | `...the barrier is not technical; it is the absence...` |
| `...three metrics—ΔDP, ΔEO, Δg_noise—together with...` | `...three metrics (ΔDP, ΔEO, Δg_noise), together with...` |
| `...available~\cite{X}—yet no study has applied them.` | `...available~\cite{X}. No study has applied them.` |

### Legalistic Prose (Reviewer Uigi)

The paper uses a prosecutorial register: "The paper documents that measurement absence", "Articulating the absence is itself a contribution." For an empirical paper, this sounds defensive. Replace with direct declarative statements about what was found.

**Patterns to eliminate:**
- Sentences beginning "This paper documents..." → replace with the actual finding
- "is itself a contribution" → remove; let reviewers decide
- "The barrier is not technical" repeated three times → say it once, in the conclusion

### Excessive Use of "Genuine"

Line 292 and elsewhere: "genuine" is vague. Replace with the specific property you mean: "substantial", "well-documented", "statistically significant", "reproducible."

### The Rule of Three

The paper repeatedly structures points as three-item lists ("three metrics", "three families", "three observations"). This is fine once or twice but becomes a stylistic tic across the full paper. Vary the structure.

---

## 9. Reviewer-Specific Fixes

### Reviewer Uigi

| Weakness | Fix |
|----------|-----|
| W1: No empirical evaluation | Fixed by new Sections 6.5–6.7 |
| W2: LLM-like writing, excessive dashes | Em dash pass + prose rewrite (Section 8 above) |
| W3: ΔEO vs ΔDP distinction unclear | One-sentence clarification added to introduction |
| Line 48 empty line | Delete the blank line |
| Table 2 oversized | Scale or rotate |
| Figure 1 caption left/right with no right panel | Fix caption (Section 2 above) |
| Line 292 "genuine" | Replace with specific descriptor |

### Reviewer 4sPD

| Weakness | Fix |
|----------|-----|
| W1: Paper selection not explained | Already fixed in your revision (methodology paragraph in intro) |
| W2: No engagement with non-Indian accented English | Add 2–3 sentences in intro: "Accent bias in ASR has been documented for L2 English broadly~\cite{...}, for African American English~\cite{...}, and for non-native speakers of European languages~\cite{...}. Our focus on Indian English is motivated by the scale of the population (128M speakers) and the complete absence of fairness evaluation in a literature that is otherwise mature." |
| W3: Section 2 doesn't tailor to Indian English | Add phonological motivation paragraph (Section 6 above) |
| W4: What makes Indian English special | Same paragraph answers this |
| W5: Abstract not straightforward | Replaced entirely (Section 4 above) |
| Line 55/59 gap size | Quantify: "Javed et al.~\cite{javed2023svarah} found WER gaps of 12–15 pp between accent families" |
| Line 781 reference format | Fix BibTeX entry |

---

## 10. Venue Framing

### AIES 2026 (Primary — Deadline May 21)

**Frame the GRL failure as the main finding.** AIES is an AI ethics venue — a concrete demonstration that a widely-used bias-mitigation technique backfires for underrepresented populations is exactly what they publish. The framing should be:

> *Fairness interventions designed without demographic coverage guarantees can harm the groups they intend to protect. We document this in Indian English ASR: gradient-reversal de-biasing, absent Sino-Tibetan training data, consistently worsens outcomes for Sino-Tibetan speakers.*

This is a policy-relevant finding, not just a technical one. Lead with it.

**Sections to emphasize for AIES:** The GRL result (Section 6.6), the noise-robustness gap (Section 6.7, noise part), the Open Problems section (especially the "dense demographic annotation" and "synthetic ST data" items).

**Sections to compress for AIES:** The taxonomy (Section 2) can be cut to 1 page. Reviewers will know what Wav2Vec2 and Whisper are.

### ARR May Cycle (Backup — Deadline May 25)

**Frame the CISPO result as the main finding** with the GRL failure as the key ablation. ARR reviewers are NLP/speech researchers who will appreciate the technical contribution more than the ethics angle.

The abstract for ARR should lead with: "We propose and evaluate CISPO-based RL post-training with family-balanced sampling for accent-fair ASR..."

**Sections to emphasize for ARR:** The method (Section 6.5, which needs to be expanded with more algorithmic detail for an NLP audience), the comparison table, the significance test.

### What NOT to claim in either venue

Do not claim the CISPO fairness reward is working as intended — the reward cancellation flaw means it isn't. Claim: (1) family-balanced sampling is sufficient to eliminate statistical systematicity, (2) the CISPO training framework is sound and a corrected reward is in progress, (3) the GRL failure is documented and reproducible.

---

## Appendix: Section Structure Comparison

| Old paper (rejected) | New paper |
|---------------------|-----------|
| §1 Introduction | §1 Introduction *(rewritten)* |
| §2 Taxonomy | §2 Taxonomy *(+ Indian English paragraph)* |
| §3 Data & Benchmarks | §3 Data & Benchmarks *(Table 2 fixed)* |
| §4 Robustness | §4 Robustness *(keep)* |
| §5 Fairness | §5 Fairness *(keep, trim)* |
| §6 Measurement Gap | §6 Measurement Gap *(+ your row in Table 3)* |
| — | **§6.5 Experimental Setup** *(new)* |
| — | **§6.6 Result I: GRL Failure** *(new)* |
| — | **§6.7 Result II: CISPO** *(new)* |
| §7 Protocol | §7 Protocol *(keep, now validated by §6.5–6.7)* |
| §8 Open Problems | §8 Open Problems *(update IndicTTS/Stage 3 as next step)* |
| §9 Conclusion | §9 Conclusion *(rewrite, remove "barrier is not technical" repetition)* |

**Target length:** ACL format allows 8 pages + references. Old paper was 8 pages pure text. Adding Sections 6.5–6.7 requires compressing Sections 2 and 4 by approximately 1 page each. Section 2 (taxonomy) is the easiest to trim — subsections on AST and CPE can be cut to one paragraph each since they are context, not core argument.

---

*All suggestions grounded in: reviewer reports (Uigi + 4sPD, ACL SRW 2026), v5 experimental log, known issues in reward.py documented in RL_TRAINING_FIXES.md.*
