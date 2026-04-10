"""
HybridAdversarialASR
--------------------
Wav2Vec 2.0 CTC ASR head with:
  - LoRA adapters on the Transformer's attention projections
  - a Gradient-Reversal-trained MLP adversary that tries to classify the
    speaker's language family from the encoder's pooled representation
  - an optional frozen AST branch whose pooled features can be distilled
    into the Wav2Vec2 encoder via a cosine-similarity auxiliary loss

Losses returned:
    total = ctc_loss + adv_loss + aux_weight * cosine_loss
The adversary loss flows back through the GRL, so the encoder is pushed
toward a representation in which the adversary cannot tell families apart.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .grl import GradientReversalLayer

logger = logging.getLogger(__name__)


@dataclass
class HybridConfig:
    wav2vec2_id: str = "facebook/wav2vec2-base-960h"
    ast_id: str = "MIT/ast-finetuned-audioset-10-10-0.4593"
    num_accent_classes: int = 3
    lambda_adv: float = 0.0  # schedule starts at 0
    use_ast_cosine: bool = False
    aux_weight: float = 0.1
    adversary_hidden: int = 256
    adversary_dropout: float = 0.1
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: tuple[str, ...] = field(
        default_factory=lambda: ("q_proj", "k_proj", "v_proj", "out_proj")
    )
    freeze_feature_extractor: bool = True
    use_adversary: bool = True  # False = pure CTC fine-tune, no GRL loss term


def _masked_mean(hidden: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    """
    Mean-pool `hidden` [B, T, D] over the time axis, honouring `mask` [B, T]
    if provided. Returns [B, D].
    """
    if mask is None:
        return hidden.mean(dim=1)
    mask = mask.to(hidden.dtype).unsqueeze(-1)  # [B, T, 1]
    summed = (hidden * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1.0)
    return summed / counts


class HybridAdversarialASR(nn.Module):
    """
    Main training module. Forward returns a dict with `loss`, `ctc_loss`,
    `adv_loss`, (optional) `aux_loss`, and `logits` so a custom Trainer can
    log each component separately.
    """

    def __init__(self, cfg: HybridConfig | None = None):
        super().__init__()
        self.cfg = cfg or HybridConfig()

        # Wav2Vec2 + LoRA ----------------------------------------------------
        from transformers import Wav2Vec2ForCTC
        from peft import LoraConfig, TaskType, get_peft_model

        base = Wav2Vec2ForCTC.from_pretrained(self.cfg.wav2vec2_id)
        hidden_size = base.config.hidden_size
        self.hidden_size = hidden_size

        # The `masked_spec_embed` parameter is missing from the pretrained
        # checkpoint and transformers leaves it uninitialized (NaN in
        # v5.x). SpecAugment then corrupts every forward pass in train mode.
        # Reinitialize it with the small-random scheme used elsewhere in
        # Wav2Vec2 (uniform in [0, 1)).
        msb = base.wav2vec2.masked_spec_embed
        if torch.isnan(msb).any() or not torch.isfinite(msb).all():
            with torch.no_grad():
                msb.uniform_()

        if self.cfg.freeze_feature_extractor:
            base.freeze_feature_encoder()

        lora_cfg = LoraConfig(
            r=self.cfg.lora_r,
            lora_alpha=self.cfg.lora_alpha,
            lora_dropout=self.cfg.lora_dropout,
            target_modules=list(self.cfg.lora_target_modules),
            bias="none",
            task_type=TaskType.FEATURE_EXTRACTION,
        )
        self.asr = get_peft_model(base, lora_cfg)

        # Make sure the CTC head (`lm_head`) remains trainable even though
        # PEFT freezes most base weights — without this the output
        # projection is locked and CTC cannot fine-tune.
        for name, param in self.asr.named_parameters():
            if "lm_head" in name:
                param.requires_grad_(True)

        # Gradient reversal + adversary -------------------------------------
        self.grl = GradientReversalLayer(lambda_=self.cfg.lambda_adv)
        self.adversary = nn.Sequential(
            nn.Linear(hidden_size, self.cfg.adversary_hidden),
            nn.ReLU(),
            nn.Dropout(self.cfg.adversary_dropout),
            nn.Linear(self.cfg.adversary_hidden, self.cfg.num_accent_classes),
        )

        # Optional AST branch -----------------------------------------------
        self.ast = None
        self.ast_proj = None
        if self.cfg.use_ast_cosine:
            from transformers import ASTModel

            ast = ASTModel.from_pretrained(self.cfg.ast_id)
            for p in ast.parameters():
                p.requires_grad_(False)
            ast.eval()
            self.ast = ast
            if ast.config.hidden_size != hidden_size:
                self.ast_proj = nn.Linear(ast.config.hidden_size, hidden_size, bias=False)

    # ── Lambda schedule helper (called from the training loop) ────────────

    @property
    def lambda_adv(self) -> float:
        return self.grl.lambda_

    @lambda_adv.setter
    def lambda_adv(self, value: float) -> None:
        self.grl.lambda_ = float(value)

    # ── HF Trainer passthroughs ────────────────────────────────────────────

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None) -> None:
        base = self.asr.base_model.model if hasattr(self.asr, "base_model") else self.asr
        if gradient_checkpointing_kwargs is None:
            base.gradient_checkpointing_enable()
        else:
            base.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)
        if hasattr(base, "enable_input_require_grads"):
            base.enable_input_require_grads()

    def gradient_checkpointing_disable(self) -> None:
        base = self.asr.base_model.model if hasattr(self.asr, "base_model") else self.asr
        base.gradient_checkpointing_disable()

    # ── Forward ────────────────────────────────────────────────────────────

    def forward(
        self,
        input_values: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        accent_labels: torch.Tensor | None = None,
        mel_spec: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        asr_out = self.asr(
            input_values=input_values,
            attention_mask=attention_mask,
            labels=labels,
            output_hidden_states=True,
            return_dict=True,
        )
        logits = asr_out.logits
        hidden = asr_out.hidden_states[-1]  # [B, T, D] last transformer layer

        ctc_loss = asr_out.loss if labels is not None else torch.tensor(0.0, device=hidden.device)

        # Build a hidden-time mask from the raw input attention mask.
        hidden_mask: torch.Tensor | None = None
        if attention_mask is not None:
            # Wav2Vec2 downsamples time by its CNN stride stack. `_get_feat_extract_output_lengths`
            # gives us the per-example post-downsampling lengths.
            base = self.asr.base_model.model if hasattr(self.asr, "base_model") else self.asr
            out_lens = base._get_feat_extract_output_lengths(attention_mask.sum(-1)).long()
            t = hidden.size(1)
            hidden_mask = torch.zeros((hidden.size(0), t), dtype=hidden.dtype, device=hidden.device)
            for i, L in enumerate(out_lens.tolist()):
                hidden_mask[i, : min(L, t)] = 1.0

        pooled = _masked_mean(hidden, hidden_mask)  # [B, D]

        adv_loss = torch.tensor(0.0, device=pooled.device)
        accent_logits = torch.zeros(
            hidden.size(0), self.cfg.num_accent_classes, device=pooled.device
        )
        if self.cfg.use_adversary:
            reversed_pooled = self.grl(pooled)
            accent_logits = self.adversary(reversed_pooled)
            if accent_labels is not None:
                adv_loss = F.cross_entropy(accent_logits, accent_labels)

        total = ctc_loss + adv_loss

        aux_loss = torch.tensor(0.0, device=pooled.device)
        if self.cfg.use_ast_cosine and mel_spec is not None and self.ast is not None:
            with torch.no_grad():
                ast_out = self.ast(input_values=mel_spec, return_dict=True)
                ast_pooled = ast_out.pooler_output  # [B, D_ast]
            if self.ast_proj is not None:
                ast_pooled = self.ast_proj(ast_pooled)
            # Encourage cosine similarity between Wav2Vec2 pooled and (frozen) AST pooled.
            aux_loss = 1.0 - F.cosine_similarity(pooled, ast_pooled, dim=-1).mean()
            total = total + self.cfg.aux_weight * aux_loss

        return {
            "loss": total,
            "ctc_loss": ctc_loss,
            "adv_loss": adv_loss,
            "aux_loss": aux_loss,
            "logits": logits,
            "accent_logits": accent_logits,
        }

    # ── Saving / loading adapters + adversary ─────────────────────────────

    def save_adapter(self, save_dir: str) -> None:
        """Persist only the LoRA adapter and the adversary head (no base weights)."""
        import os

        os.makedirs(save_dir, exist_ok=True)
        self.asr.save_pretrained(save_dir)  # writes the LoRA adapter
        torch.save(self.adversary.state_dict(), os.path.join(save_dir, "adversary.pt"))
        if self.ast_proj is not None:
            torch.save(self.ast_proj.state_dict(), os.path.join(save_dir, "ast_proj.pt"))
        logger.info("Saved hybrid adapter + adversary to %s", save_dir)

    def load_adapter(self, save_dir: str) -> None:
        import os

        from peft import PeftModel

        # Reload the LoRA adapter on top of the already-initialised base.
        base = self.asr.get_base_model() if hasattr(self.asr, "get_base_model") else self.asr
        self.asr = PeftModel.from_pretrained(base, save_dir, is_trainable=True)

        adv_path = os.path.join(save_dir, "adversary.pt")
        if os.path.exists(adv_path):
            self.adversary.load_state_dict(torch.load(adv_path, map_location="cpu"))

        proj_path = os.path.join(save_dir, "ast_proj.pt")
        if self.ast_proj is not None and os.path.exists(proj_path):
            self.ast_proj.load_state_dict(torch.load(proj_path, map_location="cpu"))

        logger.info("Loaded hybrid adapter + adversary from %s", save_dir)
