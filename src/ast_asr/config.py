"""Strict experiment configuration with pinned model and dataset revisions."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class ModelConfig:
    model_id: str
    revision: str
    lora_rank: int
    lora_alpha: int
    lora_dropout: float
    target_modules: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class DatasetConfig:
    dataset_id: str
    revision: str
    prepared_manifest: Path
    fold_directory: Path
    archive_root: Path
    musan_root: Path
    musan_revision: str


@dataclass(frozen=True, slots=True)
class SftConfig:
    maximum_epochs: int
    learning_rate: float
    train_batch_size: int
    evaluation_batch_size: int
    gradient_accumulation_steps: int
    warmup_ratio: float


@dataclass(frozen=True, slots=True)
class PolicyConfig:
    candidates: int
    rollout_cycles: int
    inner_updates: int
    rollout_temperature: float
    maximum_new_tokens: int
    gradient_clip: float
    learning_rate_grid: tuple[float, ...]
    risk_ema: float
    dual_learning_rate: float
    uniform_mix: float
    training_snr_db: tuple[float, float]
    reference_kl_beta: float = 0.0


@dataclass(frozen=True, slots=True)
class EvaluationConfig:
    conditions: tuple[str, ...]
    bootstrap_samples: int


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    schema_version: int
    model: ModelConfig
    dataset: DatasetConfig
    sft: SftConfig
    policy: PolicyConfig
    evaluation: EvaluationConfig

    def to_dict(self) -> dict[str, Any]:
        def jsonable(value: Any) -> Any:
            if isinstance(value, Path):
                return str(value)
            if isinstance(value, dict):
                return {key: jsonable(item) for key, item in value.items()}
            if isinstance(value, (list, tuple)):
                return [jsonable(item) for item in value]
            return value

        return jsonable(asdict(self))

    @classmethod
    def from_json(cls, path: Path) -> ExperimentConfig:
        raw: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
        model = raw["model"]
        dataset = raw["dataset"]
        sft = raw["sft"]
        policy = raw["policy"]
        evaluation = raw["evaluation"]
        config = cls(
            schema_version=int(raw["schema_version"]),
            model=ModelConfig(
                model_id=str(model["model_id"]),
                revision=str(model["revision"]),
                lora_rank=int(model["lora_rank"]),
                lora_alpha=int(model["lora_alpha"]),
                lora_dropout=float(model["lora_dropout"]),
                target_modules=tuple(str(item) for item in model["target_modules"]),
            ),
            dataset=DatasetConfig(
                dataset_id=str(dataset["dataset_id"]),
                revision=str(dataset["revision"]),
                prepared_manifest=Path(dataset["prepared_manifest"]),
                fold_directory=Path(dataset["fold_directory"]),
                archive_root=Path(dataset["archive_root"]),
                musan_root=Path(dataset["musan_root"]),
                musan_revision=str(dataset["musan_revision"]),
            ),
            sft=SftConfig(
                maximum_epochs=int(sft["maximum_epochs"]),
                learning_rate=float(sft["learning_rate"]),
                train_batch_size=int(sft["train_batch_size"]),
                evaluation_batch_size=int(sft["evaluation_batch_size"]),
                gradient_accumulation_steps=int(sft["gradient_accumulation_steps"]),
                warmup_ratio=float(sft["warmup_ratio"]),
            ),
            policy=PolicyConfig(
                candidates=int(policy["candidates"]),
                rollout_cycles=int(policy["rollout_cycles"]),
                inner_updates=int(policy["inner_updates"]),
                rollout_temperature=float(policy["rollout_temperature"]),
                maximum_new_tokens=int(policy["maximum_new_tokens"]),
                gradient_clip=float(policy["gradient_clip"]),
                learning_rate_grid=tuple(float(item) for item in policy["learning_rate_grid"]),
                risk_ema=float(policy["risk_ema"]),
                dual_learning_rate=float(policy["dual_learning_rate"]),
                uniform_mix=float(policy["uniform_mix"]),
                training_snr_db=tuple(float(item) for item in policy["training_snr_db"]),
                reference_kl_beta=float(policy.get("reference_kl_beta", 0.0)),
            ),
            evaluation=EvaluationConfig(
                conditions=tuple(str(item) for item in evaluation["conditions"]),
                bootstrap_samples=int(evaluation["bootstrap_samples"]),
            ),
        )
        config._validate()
        return config

    def _validate(self) -> None:
        if self.schema_version != 1:
            raise ValueError("unsupported experiment configuration schema")
        if len(self.model.revision) < 12 or self.model.revision in {"main", "latest"}:
            raise ValueError("model revision must be immutable")
        if len(self.dataset.revision) < 12 or self.dataset.revision in {"main", "latest"}:
            raise ValueError("dataset revision must be immutable")
        if self.model.lora_dropout != 0:
            raise ValueError("FR-CISPO requires LoRA dropout zero")
        if self.policy.inner_updates <= 1:
            raise ValueError("live off-policy training requires more than one inner update")
        if self.policy.training_snr_db[0] > self.policy.training_snr_db[1]:
            raise ValueError("training SNR range is reversed")
        if not math.isfinite(self.policy.reference_kl_beta):
            raise ValueError("reference KL beta must be finite")
        if self.policy.reference_kl_beta < 0:
            raise ValueError("reference KL beta must be nonnegative")
