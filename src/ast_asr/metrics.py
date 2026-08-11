"""Auditable ASR prediction records and concatenated edit counts."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import asdict, dataclass
from typing import Any

_PUNCTUATION = re.compile(r"[^\w\s']", flags=re.UNICODE)


def normalize_for_wer(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text).lower()
    normalized = _PUNCTUATION.sub(" ", normalized)
    return " ".join(normalized.split())


@dataclass(frozen=True, slots=True)
class EditCounts:
    substitutions: int = 0
    deletions: int = 0
    insertions: int = 0
    reference_words: int = 0

    @property
    def errors(self) -> int:
        return self.substitutions + self.deletions + self.insertions

    @property
    def wer(self) -> float:
        if self.reference_words == 0:
            return 0.0 if self.errors == 0 else float("inf")
        return self.errors / self.reference_words

    def __add__(self, other: EditCounts) -> EditCounts:
        return EditCounts(
            substitutions=self.substitutions + other.substitutions,
            deletions=self.deletions + other.deletions,
            insertions=self.insertions + other.insertions,
            reference_words=self.reference_words + other.reference_words,
        )

    def scaled(self, multiplier: int) -> EditCounts:
        return EditCounts(
            substitutions=self.substitutions * multiplier,
            deletions=self.deletions * multiplier,
            insertions=self.insertions * multiplier,
            reference_words=self.reference_words * multiplier,
        )


def word_edit_counts(reference: str, hypothesis: str) -> EditCounts:
    reference_words = normalize_for_wer(reference).split()
    hypothesis_words = normalize_for_wer(hypothesis).split()
    rows = len(reference_words) + 1
    columns = len(hypothesis_words) + 1
    table: list[list[tuple[int, int, int, int]]] = [
        [(0, 0, 0, 0) for _ in range(columns)] for _ in range(rows)
    ]
    for row in range(1, rows):
        table[row][0] = (row, 0, row, 0)
    for column in range(1, columns):
        table[0][column] = (column, 0, 0, column)
    for row in range(1, rows):
        for column in range(1, columns):
            if reference_words[row - 1] == hypothesis_words[column - 1]:
                table[row][column] = table[row - 1][column - 1]
                continue
            substitution = table[row - 1][column - 1]
            deletion = table[row - 1][column]
            insertion = table[row][column - 1]
            options = (
                (substitution[0] + 1, substitution[1] + 1, substitution[2], substitution[3]),
                (deletion[0] + 1, deletion[1], deletion[2] + 1, deletion[3]),
                (insertion[0] + 1, insertion[1], insertion[2], insertion[3] + 1),
            )
            table[row][column] = min(options)
    _, substitutions, deletions, insertions = table[-1][-1]
    return EditCounts(
        substitutions=substitutions,
        deletions=deletions,
        insertions=insertions,
        reference_words=len(reference_words),
    )


@dataclass(frozen=True, slots=True)
class PredictionRecord:
    fold: int
    utterance_id: str
    speaker_id: str
    primary_language: str
    family: str
    condition: str
    reference: str
    hypothesis: str
    checkpoint_revision: str
    arm: str = "unknown"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> PredictionRecord:
        return cls(
            fold=int(value["fold"]),
            utterance_id=str(value["utterance_id"]),
            speaker_id=str(value["speaker_id"]),
            primary_language=str(value["primary_language"]),
            family=str(value["family"]),
            condition=str(value["condition"]),
            reference=str(value["reference"]),
            hypothesis=str(value["hypothesis"]),
            checkpoint_revision=str(value["checkpoint_revision"]),
            arm=str(value.get("arm", "unknown")),
        )
