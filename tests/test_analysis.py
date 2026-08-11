from __future__ import annotations

import pytest

from ast_asr.analysis import aggregate_prediction_records
from ast_asr.metrics import PredictionRecord


def test_oof_aggregation_uses_edit_counts_and_one_fold_per_speaker() -> None:
    records = []
    for fold in range(5):
        family = "Dravidian" if fold < 3 else "Indo-Aryan"
        for condition, hypothesis in (
            ("clean", "one two"),
            ("white_10db", "one wrong" if fold == 0 else "one two"),
        ):
            records.append(
                PredictionRecord(
                    fold=fold,
                    utterance_id=f"utt-{fold}-{condition}",
                    speaker_id=f"speaker-{fold}",
                    primary_language="Tamil" if fold < 3 else "Hindi",
                    family=family,
                    condition=condition,
                    reference="one two",
                    hypothesis=hypothesis,
                    checkpoint_revision="checkpoint@abc",
                )
            )

    summary = aggregate_prediction_records(
        records,
        expected_speakers=5,
        bootstrap_samples=100,
        seed=7,
    )

    assert summary.metrics["clean_overall_wer"] == 0.0
    assert summary.metrics["worst_family_condition_wer"] == pytest.approx(1 / 6)
    assert len(summary.bootstrap["worst_family_condition_wer"]) == 2

    paired = [
        PredictionRecord(**{**record.to_dict(), "arm": "sft"})
        for record in records
    ] + [
        PredictionRecord(
            **{
                **record.to_dict(),
                "arm": "fr-cispo",
                "hypothesis": record.reference,
            }
        )
        for record in records
    ]
    paired_summary = aggregate_prediction_records(
        paired,
        expected_speakers=5,
        bootstrap_samples=10,
        seed=7,
    )
    assert paired_summary.metrics["paired_differences_from_baseline"]["fr-cispo-sft"][
        "worst_family_condition_wer"
    ] < 0
    assert "difference::fr-cispo-sft::worst_family_condition_wer" in paired_summary.bootstrap

    records[-1] = PredictionRecord(
        **{**records[-1].to_dict(), "fold": 0}
    )
    with pytest.raises(ValueError, match="more than one test fold"):
        aggregate_prediction_records(
            records,
            expected_speakers=5,
            bootstrap_samples=10,
            seed=7,
        )
