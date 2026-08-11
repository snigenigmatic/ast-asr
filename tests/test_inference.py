from __future__ import annotations

from types import SimpleNamespace

import torch

from ast_asr.inference import greedy_transcribe


class FakeProcessor:
    def __call__(self, audios, **kwargs):
        assert kwargs["return_attention_mask"] is True
        assert kwargs["padding"] == "max_length"
        assert kwargs["truncation"] is True
        lengths = [len(audio) for audio in audios]
        width = max(lengths)
        features = torch.zeros(len(audios), 1, width)
        mask = torch.zeros(len(audios), width, dtype=torch.long)
        for index, audio in enumerate(audios):
            features[index, 0, : len(audio)] = torch.as_tensor(audio)
            mask[index, : len(audio)] = 1
        return SimpleNamespace(input_features=features, attention_mask=mask)

    def batch_decode(self, sequences, *, skip_special_tokens):
        assert skip_special_tokens is True
        return [str(int(value)) for value in sequences.flatten().tolist()]


class FakeModel(torch.nn.Module):
    def __init__(self, dtype: torch.dtype = torch.float32):
        super().__init__()
        self.register_parameter("dtype_marker", torch.nn.Parameter(torch.ones((), dtype=dtype)))
        self.masks = []
        self.feature_dtypes = []

    def generate(self, *, input_features, attention_mask, **kwargs):
        self.masks.append(attention_mask.detach().cpu())
        self.feature_dtypes.append(input_features.dtype)
        assert kwargs["do_sample"] is False
        values = (input_features[:, 0, :] * attention_mask).sum(dim=1).round().long()
        return values.unsqueeze(1)


def test_greedy_inference_passes_masks_and_is_batch_invariant() -> None:
    audios = [torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0, 5.0])]
    processor = FakeProcessor()
    batched_model = FakeModel()
    solo_model = FakeModel()

    batched = greedy_transcribe(batched_model, processor, audios, batch_size=2)
    solo = greedy_transcribe(solo_model, processor, audios, batch_size=1)

    assert batched == solo == ["3", "12"]
    torch.testing.assert_close(
        batched_model.masks[0],
        torch.tensor([[1, 1, 0], [1, 1, 1]]),
    )


def test_greedy_inference_casts_features_to_model_dtype() -> None:
    model = FakeModel(dtype=torch.float16)
    greedy_transcribe(
        model,
        FakeProcessor(),
        [torch.tensor([1.0, 2.0])],
        batch_size=1,
    )

    assert model.feature_dtypes == [torch.float16]
