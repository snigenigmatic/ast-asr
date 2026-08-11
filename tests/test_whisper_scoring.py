from __future__ import annotations

import torch

from ast_asr.whisper_policy import pack_decoder_targets


def test_decoder_target_packing_scores_only_generated_hypothesis_tokens() -> None:
    packed = pack_decoder_targets(
        prefix_token_ids=(1, 2, 3),
        target_token_ids=((10, 11), (12,)),
        pad_token_id=0,
    )

    torch.testing.assert_close(
        packed.decoder_input_ids,
        torch.tensor([[1, 2, 3, 10], [1, 2, 3, 12]]),
    )
    torch.testing.assert_close(
        packed.target_ids,
        torch.tensor([[2, 3, 10, 11], [2, 3, 12, 0]]),
    )
    torch.testing.assert_close(
        packed.score_mask,
        torch.tensor([[False, False, True, True], [False, False, True, False]]),
    )
