from __future__ import annotations

import torch

from ast_asr.rollouts import (
    AcousticCondition,
    CandidateRollout,
    FrozenRolloutBatch,
    UtteranceRollout,
)


def test_frozen_rollout_round_trip_preserves_policy_evidence() -> None:
    batch = FrozenRolloutBatch(
        model_revision="sft-fold-0@abc123",
        utterances=(
            UtteranceRollout(
                utterance_id="utt-001",
                speaker_id="speaker-001",
                primary_language="Tamil",
                family="Dravidian",
                condition=AcousticCondition.CLEAN,
                reference="a test",
                candidates=(
                    CandidateRollout(
                        hypothesis="a test",
                        token_ids=(11, 12),
                        token_mask=(True, True),
                        old_token_log_probs=(-0.25, -0.75),
                        old_sequence_log_probability=-0.5,
                        wer=0.0,
                    ),
                ),
            ),
        ),
    )

    restored = FrozenRolloutBatch.from_dict(batch.to_dict())
    tensors = restored.objective_tensors()

    assert restored == batch
    assert tensors.old_token_log_probs.dtype == torch.float32
    assert tensors.token_mask.dtype == torch.bool
    assert tensors.old_sequence_log_probs.item() == -0.5
    assert tensors.candidate_wers.item() == 0.0
