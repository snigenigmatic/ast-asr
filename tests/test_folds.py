from __future__ import annotations

from collections import Counter

from ast_asr.folds import SpeakerProfile, build_outer_fold_manifests


def test_five_fold_manifests_are_deterministic_disjoint_and_complete() -> None:
    profiles = tuple(
        SpeakerProfile(
            speaker_id=f"speaker-{index:03d}",
            primary_language=f"language-{index % 19:02d}",
            family=("Indo-Aryan", "Dravidian", "Sino-Tibetan")[index % 3],
            gender=("Female", "Male")[index % 2],
            age_group=("18-30", "30-45", "45-60", "60+")[index % 4],
            reference_word_count=100 + (index * 17) % 300,
        )
        for index in range(117)
    )

    first = build_outer_fold_manifests(profiles, seed=2026)
    second = build_outer_fold_manifests(tuple(reversed(profiles)), seed=2026)

    assert [manifest.content_hash for manifest in first] == [
        manifest.content_hash for manifest in second
    ]
    test_appearances: Counter[str] = Counter()
    for manifest in first:
        train = set(manifest.train_speakers)
        validation = set(manifest.validation_speakers)
        test = set(manifest.test_speakers)
        assert train.isdisjoint(validation)
        assert train.isdisjoint(test)
        assert validation.isdisjoint(test)
        assert len(train | validation | test) == 117
        test_appearances.update(test)

    assert set(test_appearances.values()) == {1}
    assert len(test_appearances) == 117
