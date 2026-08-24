"""Tests for the SPIRE-SIES cross-corpus evaluation logic.

The module under test lives in scripts/ rather than src/ so that the H7
recovery authorization's frozen src/ and configs/ manifests stay byte-identical.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import spire_crosscorpus as sc

from ast_asr.metrics import EditCounts


def _row(
    uid: str,
    speaker: str,
    accent: str,
    family: str,
    *,
    split: str = "val",
    duration: float = 5.0,
    reference: str = "HELLO THERE WORLD",
    gender: str = "Female",
) -> dict[str, object]:
    return {
        "uid": uid,
        "speaker_id": speaker,
        "accent": accent,
        "language_family": family,
        "gender": gender,
        "reference": reference,
        "duration": duration,
        "split": split,
    }


def _result(
    uid: str,
    speaker: str,
    family: str,
    *,
    errors: int,
    words: int,
    gender: str = "Female",
) -> sc.UtteranceResult:
    return sc.UtteranceResult(
        uid=uid,
        speaker_id=speaker,
        family=family,
        gender=gender,
        counts=EditCounts(substitutions=errors, reference_words=words),
    )


class TestResolveFamily:
    def test_maps_spire_languages_to_two_families(self) -> None:
        assert sc.resolve_family("Tamil") == "Dravidian"
        assert sc.resolve_family("hindi") == "Indo-Aryan"

    def test_all_seventeen_spire_languages_are_known_and_two_family(self) -> None:
        spire_languages = [
            "bengali",
            "dogri",
            "gujarati",
            "hindi",
            "kannada",
            "kashmiri",
            "konkani",
            "maithili",
            "malayalam",
            "marathi",
            "nepali",
            "odia",
            "punjabi",
            "sindhi",
            "tamil",
            "telugu",
            "urdu",
        ]
        families = {sc.resolve_family(language) for language in spire_languages}
        assert families == {"Dravidian", "Indo-Aryan"}

    def test_rejects_unknown_language(self) -> None:
        with pytest.raises(sc.SpireContractError, match="unknown primary language"):
            sc.resolve_family("Klingon")

    def test_rejects_family_disagreement(self) -> None:
        with pytest.raises(sc.SpireContractError, match="disagrees with taxonomy"):
            sc.resolve_family("Tamil", "Indo-Aryan")

    def test_sino_tibetan_language_is_rejected_as_out_of_corpus(self) -> None:
        # Bodo is Svarah's only Sino-Tibetan language and SPIRE does not ship
        # it, so it must be rejected rather than silently adding a third family.
        with pytest.raises(sc.SpireContractError, match="unknown primary language"):
            sc.resolve_family("Bodo")


class TestValidateSplits:
    def _splits(self) -> dict[str, list[str]]:
        return {
            "train": [f"T{index}" for index in range(sc.EXPECTED_TRAIN_SPEAKERS)],
            "val": [f"V{index}" for index in range(sc.EXPECTED_VAL_SPEAKERS)],
        }

    def test_accepts_the_frozen_split_shape(self) -> None:
        val = sc.validate_splits(self._splits())
        assert len(val) == sc.EXPECTED_VAL_SPEAKERS
        assert "V0" in val

    def test_rejects_speaker_overlap(self) -> None:
        splits = self._splits()
        splits["val"][0] = "T0"
        with pytest.raises(sc.SpireContractError, match="not speaker-disjoint"):
            sc.validate_splits(splits)

    def test_rejects_unexpected_counts(self) -> None:
        splits = self._splits()
        splits["val"].pop()
        with pytest.raises(sc.SpireContractError, match="expected"):
            sc.validate_splits(splits)

    def test_rejects_extra_keys(self) -> None:
        splits = self._splits()
        splits["test"] = []
        with pytest.raises(sc.SpireContractError, match="exactly train and val"):
            sc.validate_splits(splits)


class TestAcceptRow:
    val_speakers = frozenset({"V1"})

    def test_accepts_val_row(self) -> None:
        utterance = sc.accept_row(
            _row("u1", "V1", "Tamil", "Dravidian"), self.val_speakers
        )
        assert utterance is not None
        assert utterance.family == "Dravidian"
        assert utterance.speaker_id == "V1"

    def test_skips_train_row(self) -> None:
        row = _row("u2", "T9", "Tamil", "Dravidian", split="train")
        assert sc.accept_row(row, self.val_speakers) is None

    def test_fails_when_val_label_contradicts_split_file(self) -> None:
        row = _row("u3", "UNLISTED", "Tamil", "Dravidian", split="val")
        with pytest.raises(sc.SpireContractError, match="absent from the split file"):
            sc.accept_row(row, self.val_speakers)

    def test_fails_when_val_speaker_is_labelled_train(self) -> None:
        row = _row("u4", "V1", "Tamil", "Dravidian", split="train")
        with pytest.raises(sc.SpireContractError, match="is a val speaker"):
            sc.accept_row(row, self.val_speakers)

    @pytest.mark.parametrize("duration", [0.5, 30.5])
    def test_filters_out_of_range_durations(self, duration: float) -> None:
        row = _row("u5", "V1", "Tamil", "Dravidian", duration=duration)
        assert sc.accept_row(row, self.val_speakers) is None

    def test_filters_empty_reference(self) -> None:
        row = _row("u6", "V1", "Tamil", "Dravidian", reference="   ")
        assert sc.accept_row(row, self.val_speakers) is None


class TestScoring:
    def test_uppercase_reference_normalizes_against_lowercase_hypothesis(self) -> None:
        utterance = sc.accept_row(
            _row("u1", "V1", "Tamil", "Dravidian"), frozenset({"V1"})
        )
        assert utterance is not None
        result = sc.score_utterance(utterance, "HELLO THERE WORLD", "hello there world")
        assert result.counts.errors == 0
        assert result.counts.wer == 0.0

    def test_pooling_is_error_weighted_not_utterance_averaged(self) -> None:
        results = [
            _result("a", "V1", "Dravidian", errors=1, words=1),
            _result("b", "V1", "Dravidian", errors=0, words=99),
        ]
        pooled = sc.pool(results, "family")
        # Error-weighted pooling gives 1/100, not the 0.5 an utterance mean gives.
        assert pooled["Dravidian"].wer == pytest.approx(0.01)


class TestSummarizeArm:
    def _results(self) -> list[sc.UtteranceResult]:
        return [
            _result("a", "V1", "Dravidian", errors=4, words=10),
            _result("b", "V2", "Indo-Aryan", errors=1, words=10),
            _result("c", "V3", "Indo-Aryan", errors=1, words=10),
        ]

    def test_reports_registered_endpoints(self) -> None:
        summary = sc.summarize_arm(self._results())
        assert summary["utterances"] == 3
        assert summary["speakers"] == 3
        assert summary["overall_wer"] == pytest.approx(6 / 30)
        assert summary["worst_family"] == "Dravidian"
        assert summary["worst_family_wer"] == pytest.approx(0.4)
        assert summary["family_gap"] == pytest.approx(0.3)
        assert summary["wer_by_gender"] == {"Female": pytest.approx(0.2)}

    def test_rejects_unexpected_family(self) -> None:
        results = [
            *self._results(),
            _result("d", "V4", "Sino-Tibetan", errors=1, words=2),
        ]
        with pytest.raises(sc.SpireContractError, match="unexpected families"):
            sc.summarize_arm(results)

    def test_rejects_empty_arm(self) -> None:
        with pytest.raises(sc.SpireContractError, match="empty arm"):
            sc.summarize_arm([])

    def test_worst_group_tie_breaks_by_name(self) -> None:
        assert sc.worst_group({"Indo-Aryan": 0.5, "Dravidian": 0.5})[0] == "Dravidian"


class TestPairedBootstrap:
    def _arms(self) -> tuple[list[sc.UtteranceResult], list[sc.UtteranceResult]]:
        control = [
            _result("a", "V1", "Dravidian", errors=5, words=10),
            _result("b", "V2", "Indo-Aryan", errors=2, words=10),
            _result("c", "V3", "Indo-Aryan", errors=3, words=10),
        ]
        treatment = [
            _result("a", "V1", "Dravidian", errors=3, words=10),
            _result("b", "V2", "Indo-Aryan", errors=2, words=10),
            _result("c", "V3", "Indo-Aryan", errors=3, words=10),
        ]
        return control, treatment

    def test_is_deterministic_for_a_fixed_seed(self) -> None:
        control, treatment = self._arms()
        first = sc.paired_speaker_bootstrap(control, treatment, resamples=200, seed=7)
        second = sc.paired_speaker_bootstrap(control, treatment, resamples=200, seed=7)
        assert first == second

    def test_reports_speaker_cluster_unit(self) -> None:
        control, treatment = self._arms()
        result = sc.paired_speaker_bootstrap(control, treatment, resamples=50, seed=1)
        assert result["cluster_unit"] == "corpus_speaker"
        assert result["clusters"] == 3

    def test_identical_arms_give_a_zero_delta(self) -> None:
        control, _ = self._arms()
        result = sc.paired_speaker_bootstrap(control, control, resamples=100, seed=3)
        assert result["overall_wer_delta"]["mean"] == pytest.approx(0.0)
        assert result["worst_family_wer_delta"]["high"] == pytest.approx(0.0)

    def test_improvement_gives_a_negative_mean_delta(self) -> None:
        control, treatment = self._arms()
        result = sc.paired_speaker_bootstrap(control, treatment, resamples=400, seed=5)
        assert result["worst_family_wer_delta"]["mean"] < 0.0

    def test_rejects_mismatched_utterances(self) -> None:
        control, treatment = self._arms()
        with pytest.raises(sc.SpireContractError, match="identical utterances"):
            sc.paired_speaker_bootstrap(control, treatment[:-1], resamples=10)

    def test_harm_detection_reads_the_upper_tail(self) -> None:
        assert sc.interval_includes_harm({"low": -0.05, "high": 0.02}) is True
        assert sc.interval_includes_harm({"low": -0.05, "high": -0.01}) is False


class TestPercentile:
    def test_matches_linear_interpolation(self) -> None:
        assert sc._percentile([0.0, 1.0, 2.0, 3.0], 50.0) == pytest.approx(1.5)

    def test_single_value(self) -> None:
        assert sc._percentile([4.2], 97.5) == pytest.approx(4.2)


class TestTaxonomyAgreement:
    """The local table must never drift from the repository taxonomy."""

    def test_local_table_agrees_with_repository_taxonomy(self) -> None:
        from ast_asr.taxonomy import SVARAH_LANGUAGE_FAMILIES

        for language, family in sc.SPIRE_LANGUAGE_FAMILIES.items():
            assert SVARAH_LANGUAGE_FAMILIES[language] == family

    def test_spire_omits_exactly_the_two_languages_svarah_adds(self) -> None:
        from ast_asr.taxonomy import SVARAH_LANGUAGE_FAMILIES

        missing = set(SVARAH_LANGUAGE_FAMILIES) - set(sc.SPIRE_LANGUAGE_FAMILIES)
        assert missing == {"Assamese", "Bodo"}
        # Bodo is the only Sino-Tibetan entry, which is why SPIRE is two-family.
        assert SVARAH_LANGUAGE_FAMILIES["Bodo"] == "Sino-Tibetan"
        assert set(sc.SPIRE_LANGUAGE_FAMILIES.values()) == set(
            sc.SPIRE_EXPECTED_FAMILIES
        )


class TestPreparationPathIsLightweight:
    """Corpus preparation must run in a CPU container without torch."""

    def test_prep_functions_work_with_torch_import_blocked(self) -> None:
        import subprocess

        code = (
            "import sys\n"
            "class Blocker:\n"
            "    def find_spec(self, name, path=None, target=None):\n"
            "        if name == 'torch' or name.startswith('torch.'):\n"
            "            raise ImportError('torch blocked on preparation path')\n"
            "        return None\n"
            "sys.meta_path.insert(0, Blocker())\n"
            f"sys.path.insert(0, {str(SCRIPTS)!r})\n"
            "import spire_crosscorpus as sc\n"
            "assert sc.resolve_family('Tamil', 'Dravidian') == 'Dravidian'\n"
            "val = sc.validate_splits({\n"
            "    'train': ['T%d' % i for i in range(sc.EXPECTED_TRAIN_SPEAKERS)],\n"
            "    'val': ['V%d' % i for i in range(sc.EXPECTED_VAL_SPEAKERS)],\n"
            "})\n"
            "row = {'uid': 'u', 'speaker_id': 'V1', 'accent': 'Tamil',\n"
            "       'language_family': 'Dravidian', 'gender': 'Female',\n"
            "       'reference': 'A B C', 'duration': 5.0, 'split': 'val'}\n"
            "assert sc.accept_row(row, val) is not None\n"
            "assert 'torch' not in sys.modules\n"
            "print('PREP_OK')\n"
        )
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr
        assert "PREP_OK" in result.stdout


class TestResultFromRecord:
    """predictions.jsonl records must round-trip without re-scoring text."""

    def _record(self) -> dict[str, object]:
        return {
            "uid": "u1",
            "speaker_id": "V1",
            "family": "Dravidian",
            "gender": "Female",
            "reference": "A B C",
            "hypothesis": "A B D",
            "substitutions": 1,
            "deletions": 0,
            "insertions": 0,
            "reference_words": 3,
        }

    def test_uses_stored_counts_verbatim(self) -> None:
        result = sc.result_from_record(self._record())
        assert result.uid == "u1"
        assert result.speaker_id == "V1"
        assert result.family == "Dravidian"
        assert result.counts.substitutions == 1
        assert result.counts.reference_words == 3
        assert result.counts.wer == pytest.approx(1 / 3)

    def test_round_trips_a_scored_utterance(self) -> None:
        utterance = sc.accept_row(
            _row("u9", "V1", "Tamil", "Dravidian", reference="ALPHA BETA GAMMA"),
            frozenset({"V1"}),
        )
        assert utterance is not None
        scored = sc.score_utterance(utterance, "ALPHA BETA GAMMA", "alpha beta delta")
        record = {
            "uid": scored.uid,
            "speaker_id": scored.speaker_id,
            "family": scored.family,
            "gender": scored.gender,
            "substitutions": scored.counts.substitutions,
            "deletions": scored.counts.deletions,
            "insertions": scored.counts.insertions,
            "reference_words": scored.counts.reference_words,
        }
        assert sc.result_from_record(record) == scored
