from ast_asr.ladder import TRAINING_LADDER
from ast_asr.modal_smoke import build_synthetic_cases
from ast_asr.objectives import CorruptionPolicy
from ast_asr.rollouts import AcousticCondition


def test_synthetic_smoke_batch_exercises_all_registered_objectives() -> None:
    clean = build_synthetic_cases(
        seed=2026,
        corruption=CorruptionPolicy.CLEAN,
    )
    robust = build_synthetic_cases(
        seed=2026,
        corruption=CorruptionPolicy.PAIRED_WHITE,
    )

    assert set(TRAINING_LADDER) == {
        "live-grpo",
        "cispo-mwer",
        "sequence-cispo-mwer",
        "fair-cispo",
        "fr-cispo",
    }
    assert len(clean) == 3
    assert len(robust) == 6
    assert {case.condition for case in clean} == {AcousticCondition.CLEAN}
    assert {case.condition for case in robust} == {
        AcousticCondition.CLEAN,
        AcousticCondition.WHITE_TRAIN,
    }
    assert len({case.family for case in clean}) == 3
    assert all(len(case.hypotheses) == 4 for case in robust)
    assert all(case.reference in case.hypotheses for case in robust)
    assert all(case.audio.dtype.is_floating_point for case in robust)
    assert all(case.audio.ndim == 1 for case in robust)
