# FR-CISPO contribution and authorship ledger

**Status:** append-only project record. This is an evidence ledger, not an
automatic author list. It exists to make collaboration fair while placement
timelines differ across teammates.

## Principles

- Credit follows verifiable intellectual and practical contribution, not
  seniority, availability, or a contributor's placement status.
- A person is considered for authorship only after making a substantial
  contribution, participating in review of the manuscript, approving the final
  version, and accepting responsibility for the part they contributed.
- A person who has not contributed to the publication work is not included
  merely because they were on the capstone team. Acknowledgment is appropriate
  for smaller help when the person agrees.
- The team and supervisor make the final authorship decision before submission.
  This document records evidence; it does not substitute for that decision.
- Do not put sensitive personal details, placement information, or private
  disputes in this repository. Link to a private record if a dispute must be
  documented.

## How to add an entry

Add one row for each material contribution as soon as it is completed. Use a
commit hash, immutable run ID, artifact hash, or review URL where possible.
Mark verification as `self`, `peer`, or `supervisor`; a contribution remains
`pending` until somebody other than its author has checked the claimed artifact.

| Date | Contributor | Contribution category | Concrete deliverable / decision | Evidence link or immutable ID | Verification | Status | Proposed CRediT roles |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-08-10 | _unassigned_ | Method audit | Identified inert-ratio, dropout-ratio, checkpoint, mask, and FP16 invariance risks; preserved corrected evidence | `docs/development-evidence-20260810.md` | pending | recorded | Methodology; Validation; Software |
| 2026-08-11 | _unassigned_ | Research operations | Defined gates, artifact policy, and a locked-but-unapproved H1 protocol | `docs/execution-runbook.md`; `experiments/H1-kl-control/protocol.md` | pending | recorded | Project administration; Methodology |

## Contribution categories and evidence expectations

| Category | Examples of substantial work | Evidence to record |
| --- | --- | --- |
| Conceptualization / Methodology | Research question, objective design, justified protocol, failure analysis | protocol, dated design note, review comments |
| Data curation | Official-ID recovery, taxonomy reconciliation, fold validation | source hash, manifest, validation report |
| Software | Reusable implementation, tests, reproducible command path | commit hash, test output, artifact path |
| Validation | Reproduction, invariance check, checkpoint round trip, audit | signed-off checklist or immutable diagnostic |
| Formal analysis | Bootstrap plan, error analysis, mathematically checked derivation | notebook/script, reviewed table or derivation |
| Writing | Drafted/revised sections with technical review and fact checking | document version, review history |
| Supervision / Project administration | Milestone decisions, risk management, coordination that changes research execution | dated decision log |

## Authorship review checkpoints

Run these reviews at three points: before a development-gated result is
announced, before a complete draft is circulated, and immediately before
submission.

| Check | Decision record |
| --- | --- |
| Scope | Which artifacts support the current manuscript claim? |
| Substantial contributions | Which ledger entries are independently verifiable and material to that claim? |
| Manuscript responsibility | Has each proposed author reviewed the relevant text and agreed to be accountable? |
| Order and roles | Record the agreed order and CRediT roles, with a short evidence-based rationale. |
| Acknowledgments | List non-author assistance only with consent. |
| Conflict handling | Raise a factual contribution disagreement promptly with the supervisor; preserve artifacts, not accusations. |

## Paper-readiness ledger

No person should be added to an author list from this table alone. It is a
completion view for the final team review.

| Requirement | Owner(s) | Evidence | Checked by | State |
| --- | --- | --- | --- | --- |
| Official 117-speaker provenance | _unassigned_ | source and fold manifests | _unassigned_ | blocked |
| Correct FP32 evaluation + reload reproducibility | _unassigned_ | prediction and diagnostic artifacts | _unassigned_ | partial |
| Locked protocol precedes result | _unassigned_ | protocol commit and run ID | _unassigned_ | pending |
| Three-seed development gate | _unassigned_ | immutable gate JSON | _unassigned_ | blocked |
| Five-fold OOF analysis + bootstrap | _unassigned_ | OOF summary and draws | _unassigned_ | blocked |
| Draft, limitations, and verified citations | _unassigned_ | paper source and bibliography audit | _unassigned_ | pending |
| Final authorship consent | all proposed authors | dated final-review record | supervisor | pending |

## Current limitations on credit claims

The 115 profile-cluster runs are not valid speaker-disjoint results. They may
be credited as engineering development or failure analysis when supported by
their artifacts, but they cannot be credited as a completed fairness evaluation
of 117 speakers. Likewise, a failed KL run is a contribution to the evidence
base only when its configuration, diagnostics, and failure reason are retained.
