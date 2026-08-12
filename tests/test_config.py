from __future__ import annotations

import json
from pathlib import Path

import pytest

from ast_asr.config import ExperimentConfig


def test_reference_config_pins_revisions_and_research_hyperparameters() -> None:
    config = ExperimentConfig.from_json(Path("configs/fr_cispo_tiny.json"))

    assert config.model.model_id == "openai/whisper-tiny"
    assert len(config.model.revision) == 40
    assert config.model.lora_rank == 8
    assert config.model.lora_alpha == 16
    assert config.model.lora_dropout == 0.0
    assert config.sft.maximum_epochs == 5
    assert config.sft.learning_rate == 1e-4
    assert config.policy.candidates == 4
    assert config.policy.rollout_cycles == 300
    assert config.policy.inner_updates == 4
    assert config.policy.learning_rate_grid == (1e-5, 3e-5, 1e-4)
    assert config.policy.reference_kl_beta == 0.0


@pytest.mark.parametrize("beta", (-0.01, float("nan"), float("inf")))
def test_reference_kl_beta_must_be_finite_and_nonnegative(
    tmp_path: Path,
    beta: float,
) -> None:
    raw = json.loads(Path("configs/fr_cispo_tiny.json").read_text(encoding="utf-8"))
    raw["policy"]["reference_kl_beta"] = beta
    path = tmp_path / "invalid-reference-kl.json"
    path.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(ValueError, match="reference KL beta"):
        ExperimentConfig.from_json(path)
