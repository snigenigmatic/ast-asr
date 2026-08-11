"""Named, auditable objective arms for the development ladder."""

from __future__ import annotations

from .objectives import (
    AdvantageKind,
    ClipRule,
    CorruptionPolicy,
    GroupWeighting,
    ObjectiveSpec,
    RatioUnit,
)

TRAINING_LADDER: dict[str, ObjectiveSpec] = {
    "live-grpo": ObjectiveSpec(
        advantage=AdvantageKind.STANDARDIZED,
        ratio_unit=RatioUnit.TOKEN,
        clip_rule=ClipRule.PPO_SYMMETRIC,
        clip_lower=0.8,
        clip_upper=1.2,
    ),
    "cispo-mwer": ObjectiveSpec(
        advantage=AdvantageKind.CENTERED_MWER,
        ratio_unit=RatioUnit.TOKEN,
        clip_rule=ClipRule.CISPO_UPPER,
        clip_upper=2.0,
    ),
    "sequence-cispo-mwer": ObjectiveSpec(
        advantage=AdvantageKind.CENTERED_MWER,
        ratio_unit=RatioUnit.SEQUENCE,
        clip_rule=ClipRule.CISPO_UPPER,
        clip_upper=2.0,
    ),
    "fair-cispo": ObjectiveSpec(
        advantage=AdvantageKind.CENTERED_MWER,
        ratio_unit=RatioUnit.SEQUENCE,
        clip_rule=ClipRule.CISPO_UPPER,
        group_weighting=GroupWeighting.DUAL,
        clip_upper=2.0,
    ),
    "fr-cispo": ObjectiveSpec(
        advantage=AdvantageKind.CENTERED_MWER,
        ratio_unit=RatioUnit.SEQUENCE,
        clip_rule=ClipRule.CISPO_UPPER,
        group_weighting=GroupWeighting.DUAL,
        corruption=CorruptionPolicy.PAIRED_WHITE,
        clip_upper=2.0,
    ),
}
