from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def classification_loss(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    if label_smoothing <= 0.0:
        return F.cross_entropy(outputs, targets)

    n_classes = outputs.size(-1)
    log_probs = F.log_softmax(outputs, dim=-1)
    smooth_pos = 1.0 - label_smoothing
    smooth_neg = label_smoothing / (n_classes - 1) if n_classes > 1 else 0.0
    target_probs = torch.full_like(log_probs, smooth_neg)
    target_probs.scatter_(1, targets.unsqueeze(1), smooth_pos)
    return (-target_probs * log_probs).sum(dim=-1).mean()


def consistency_loss(
    clean_logits: torch.Tensor,
    noisy_logits: torch.Tensor,
    temperature: float = 2.0,
    detach_clean: bool = True,
) -> torch.Tensor:
    if detach_clean:
        clean_logits = clean_logits.detach()

    clean_probs = F.softmax(clean_logits / temperature, dim=-1)
    noisy_log_probs = F.log_softmax(noisy_logits / temperature, dim=-1)

    kl_div = F.kl_div(noisy_log_probs, clean_probs, reduction="batchmean", log_target=False)
    return (temperature ** 2) * kl_div


def sgr_nat_loss(
    noisy_logits: torch.Tensor,
    targets: torch.Tensor,
    clean_logits: torch.Tensor | None = None,
    lambda_cons: float = 0.5,
    temperature: float = 2.0,
    detach_clean: bool = True,
    clean_ce_weight: float = 0.25,
    label_smoothing: float = 0.0,
) -> dict[str, torch.Tensor]:
    losses: dict[str, torch.Tensor] = {}

    ce_noisy = classification_loss(noisy_logits, targets, label_smoothing)
    losses["ce_noisy"] = ce_noisy

    if clean_logits is not None:
        ce_clean = classification_loss(clean_logits, targets, label_smoothing)
        losses["ce_clean"] = ce_clean

        cons = consistency_loss(clean_logits, noisy_logits, temperature, detach_clean)
        losses["cons"] = cons

        total = ce_noisy + clean_ce_weight * ce_clean + lambda_cons * cons
        losses["total"] = total
    else:
        losses["total"] = ce_noisy

    return losses
