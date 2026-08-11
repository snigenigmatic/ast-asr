from __future__ import annotations

from ast_asr.ladder import TRAINING_LADDER
from ast_asr.objectives import (
    AdvantageKind,
    ClipRule,
    CorruptionPolicy,
    GroupWeighting,
    RatioUnit,
)


def test_training_ladder_changes_one_objective_axis_at_a_time() -> None:
    assert tuple(TRAINING_LADDER) == (
        "live-grpo",
        "cispo-mwer",
        "sequence-cispo-mwer",
        "fair-cispo",
        "fr-cispo",
    )
    assert TRAINING_LADDER["live-grpo"].advantage is AdvantageKind.STANDARDIZED
    assert TRAINING_LADDER["live-grpo"].clip_rule is ClipRule.PPO_SYMMETRIC
    assert TRAINING_LADDER["cispo-mwer"].ratio_unit is RatioUnit.TOKEN
    assert TRAINING_LADDER["sequence-cispo-mwer"].ratio_unit is RatioUnit.SEQUENCE
    assert TRAINING_LADDER["fair-cispo"].group_weighting is GroupWeighting.DUAL
    assert TRAINING_LADDER["fr-cispo"].corruption is CorruptionPolicy.PAIRED_WHITE
