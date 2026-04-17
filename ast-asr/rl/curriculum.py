"""
curriculum.py
3-stage curriculum scheduler and family-balanced sampler for RL post-training.

Stages:
  1. Easy  — Indo-Aryan only, short/clean audio
  2. Medium — Indo-Aryan + Dravidian, WER-bucketed sampling, light noise
  3. Hard  — All families (+ synthetic ST), oversampled ST, heavier noise

Each batch is guaranteed to contain all families present in the current stage,
so the fairness reward can compute meaningful ΔDP.
"""

from __future__ import annotations

import logging
import math
import random
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Sampler

logger = logging.getLogger(__name__)


@dataclass
class StageConfig:
    """Configuration for one curriculum stage."""
    name: str
    families: list[str]
    max_duration: float = 10.0
    family_weights: dict[str, float] = field(default_factory=dict)
    noise_prob: float = 0.0
    noise_snr_range: tuple[float, float] = (10.0, 20.0)
    min_step: int = 0
    max_step: int = 500


DEFAULT_STAGES = [
    StageConfig(
        name="easy",
        families=["Indo-Aryan"],
        max_duration=5.0,
        family_weights={"Indo-Aryan": 1.0},
        noise_prob=0.0,
        min_step=0,
        max_step=500,
    ),
    StageConfig(
        name="medium",
        families=["Indo-Aryan", "Dravidian"],
        max_duration=10.0,
        family_weights={"Indo-Aryan": 1.0, "Dravidian": 1.0},
        noise_prob=0.3,
        noise_snr_range=(10.0, 20.0),
        min_step=500,
        max_step=1500,
    ),
    StageConfig(
        name="hard",
        families=["Indo-Aryan", "Dravidian", "Sino-Tibetan"],
        max_duration=15.0,
        family_weights={"Indo-Aryan": 1.0, "Dravidian": 1.0, "Sino-Tibetan": 3.0},
        noise_prob=0.5,
        noise_snr_range=(5.0, 15.0),
        min_step=1500,
        max_step=3000,
    ),
]


class CurriculumScheduler:
    """
    Manages stage transitions based on step count and metrics.

    Transitions happen when:
      - Step count reaches the stage's max_step, OR
      - Metrics-based early transition criteria are met
    """

    def __init__(
        self,
        stages: list[StageConfig] | None = None,
        manifest: pd.DataFrame | None = None,
    ):
        self.stages = stages or DEFAULT_STAGES
        self.current_stage_idx = 0
        self.manifest = manifest
        self._transition_log: list[dict[str, Any]] = []

    @property
    def current_stage(self) -> StageConfig:
        return self.stages[self.current_stage_idx]

    @property
    def stage_name(self) -> str:
        return self.current_stage.name

    def maybe_advance(self, step: int, metrics: dict[str, float] | None = None) -> bool:
        """
        Check if we should advance to the next stage.

        Returns True if a transition occurred.
        """
        if self.current_stage_idx >= len(self.stages) - 1:
            return False  # already at final stage

        stage = self.current_stage
        should_advance = False

        # Hard transition: step count
        if step >= stage.max_step:
            should_advance = True
            reason = f"step {step} >= max_step {stage.max_step}"

        # Soft transition: metrics-based early advancement
        if not should_advance and metrics is not None:
            should_advance, reason = self._check_metrics_transition(metrics)

        if should_advance:
            old_name = stage.name
            self.current_stage_idx += 1
            new_name = self.current_stage.name
            self._transition_log.append({
                "step": step,
                "from": old_name,
                "to": new_name,
                "reason": reason,
            })
            logger.info(
                "Curriculum: %s → %s at step %d (%s)",
                old_name, new_name, step, reason,
            )
            return True

        return False

    def _check_metrics_transition(self, metrics: dict[str, float]) -> tuple[bool, str]:
        """Check metrics-based early transition criteria."""
        stage = self.current_stage

        if stage.name == "easy":
            # Transition when IA val WER plateaus
            ia_wer = metrics.get("ia_val_wer")
            if ia_wer is not None and ia_wer < 0.40:
                return True, f"IA val WER {ia_wer:.3f} < 0.40"

        elif stage.name == "medium":
            # Transition when ΔDP(IA, Drav) < 7pp
            delta_dp = metrics.get("delta_dp")
            if delta_dp is not None and delta_dp < 0.07:
                return True, f"ΔDP {delta_dp:.3f} < 0.07"

        return False, ""

    def filter_manifest(self, manifest: pd.DataFrame | None = None) -> pd.DataFrame:
        """Filter manifest to current stage's families and duration limits."""
        df = manifest if manifest is not None else self.manifest
        if df is None:
            raise ValueError("No manifest provided")

        stage = self.current_stage

        # Filter by family
        mask = df["language_family"].isin(stage.families)

        # Filter by duration
        if "duration" in df.columns:
            mask &= df["duration"] <= stage.max_duration

        filtered = df[mask].copy()
        logger.debug(
            "Stage '%s': %d/%d samples after filtering (families=%s, max_dur=%.1f)",
            stage.name, len(filtered), len(df), stage.families, stage.max_duration,
        )
        return filtered


class FamilyBalancedSampler(Sampler):
    """
    Yields batch indices that guarantee every batch contains all families
    from the current curriculum stage.

    For each batch:
      1. Determine per-family count based on family_weights
      2. Sample from each family's index pool
      3. Shuffle within the batch

    This ensures the fairness reward can compute meaningful per-family WER
    gaps within every single batch.
    """

    def __init__(
        self,
        manifest: pd.DataFrame,
        batch_size: int,
        family_weights: dict[str, float],
        seed: int = 42,
    ):
        self.manifest = manifest.reset_index(drop=True)
        self.batch_size = batch_size
        self.family_weights = family_weights
        self.rng = random.Random(seed)

        # Build per-family index pools
        self.family_indices: dict[str, list[int]] = {}
        for fam in family_weights:
            idx = self.manifest.index[
                self.manifest["language_family"] == fam
            ].tolist()
            if idx:
                self.family_indices[fam] = idx

        # Compute per-family samples per batch
        active_families = [f for f in family_weights if f in self.family_indices]
        if not active_families:
            raise ValueError("No samples found for any family in weights")

        total_weight = sum(family_weights[f] for f in active_families)
        self.per_family_count: dict[str, int] = {}
        remaining = batch_size
        for i, fam in enumerate(active_families):
            if i == len(active_families) - 1:
                # Last family gets the remainder
                count = remaining
            else:
                count = max(1, round(batch_size * family_weights[fam] / total_weight))
                remaining -= count
            self.per_family_count[fam] = max(1, count)

        # Ensure we have at least 1 sample per family
        actual_batch = sum(self.per_family_count.values())
        self._actual_batch_size = actual_batch

        logger.info(
            "FamilyBalancedSampler: batch_size=%d, per_family=%s",
            actual_batch,
            {f: c for f, c in self.per_family_count.items()},
        )

    def __iter__(self):
        # Shuffle each family's pool
        pools = {
            fam: list(idx) for fam, idx in self.family_indices.items()
        }
        for fam in pools:
            self.rng.shuffle(pools[fam])

        # Track position in each pool
        pos = {fam: 0 for fam in pools}

        n_batches = self._estimate_n_batches()
        for _ in range(n_batches):
            batch = []
            for fam, count in self.per_family_count.items():
                if fam not in pools:
                    continue
                pool = pools[fam]
                for _ in range(count):
                    if pos[fam] >= len(pool):
                        # Reshuffle and reset
                        self.rng.shuffle(pool)
                        pos[fam] = 0
                    batch.append(pool[pos[fam]])
                    pos[fam] += 1

            self.rng.shuffle(batch)
            yield batch

    def _estimate_n_batches(self) -> int:
        """Estimate total batches based on smallest family pool."""
        min_pool = min(
            len(idx) // self.per_family_count.get(fam, 1)
            for fam, idx in self.family_indices.items()
            if fam in self.per_family_count
        )
        return max(1, min_pool)

    def __len__(self) -> int:
        return self._estimate_n_batches()


def add_noise(audio: np.ndarray, snr_db: float, rng: np.random.Generator | None = None) -> np.ndarray:
    """Add white Gaussian noise to audio at the specified SNR."""
    rng = rng or np.random.default_rng()
    signal_power = np.mean(audio ** 2)
    if signal_power < 1e-10:
        return audio
    noise_power = signal_power / (10 ** (snr_db / 10))
    noise = rng.normal(0, np.sqrt(noise_power), audio.shape).astype(audio.dtype)
    return audio + noise
