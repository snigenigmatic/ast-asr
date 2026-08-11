import torch

from ast_asr.evaluation import _prepare_evaluation_model


def test_evaluation_model_uses_fp32_and_eval_mode() -> None:
    model = torch.nn.Sequential(
        torch.nn.Linear(2, 2, dtype=torch.float16),
        torch.nn.Dropout(p=0.5),
    )
    model.train()

    prepared = _prepare_evaluation_model(model)

    assert prepared is model
    assert prepared.training is False
    assert next(prepared.parameters()).dtype is torch.float32
