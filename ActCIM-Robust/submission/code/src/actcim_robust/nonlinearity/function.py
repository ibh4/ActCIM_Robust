from __future__ import annotations

import torch
import torch.nn.functional as F


def nonlinearity(x: torch.Tensor, alpha: float = 0.0, eps: float = 1e-8) -> torch.Tensor:
    if alpha == 0.0:
        return x

    max_val = x.abs().amax().clamp_min(eps)
    x_norm = x / max_val
    y = alpha * (x_norm ** 3) + (1 - alpha) * x_norm
    return y * max_val


def nonlinearity_per_tensor(x: torch.Tensor, alpha: float = 0.0, eps: float = 1e-8) -> torch.Tensor:
    return nonlinearity(x, alpha=alpha, eps=eps)


def nonlinearity_per_sample(x: torch.Tensor, alpha: float = 0.0, eps: float = 1e-8) -> torch.Tensor:
    if alpha == 0.0:
        return x

    if x.dim() == 1:
        return nonlinearity_per_tensor(x, alpha=alpha, eps=eps)

    out = torch.empty_like(x)
    for i in range(x.shape[0]):
        out[i] = nonlinearity(x[i], alpha=alpha, eps=eps)
    return out


def nonlinearity_per_channel(x: torch.Tensor, alpha: float = 0.0, eps: float = 1e-8) -> torch.Tensor:
    if alpha == 0.0:
        return x

    if x.dim() < 3 or x.dim() > 4:
        return nonlinearity_per_tensor(x, alpha=alpha, eps=eps)

    out = x.clone()
    if x.dim() == 4:
        for c in range(x.shape[1]):
            out[:, c] = nonlinearity(x[:, c], alpha=alpha, eps=eps)
    elif x.dim() == 3:
        for c in range(x.shape[0]):
            out[c] = nonlinearity(x[c], alpha=alpha, eps=eps)
    return out


def compute_activation_stats(
    clean_x: torch.Tensor, nonlinear_x: torch.Tensor
) -> dict[str, float]:
    clean_flat = clean_x.detach().reshape(-1).float()
    noisy_flat = nonlinear_x.detach().reshape(-1).float()

    with torch.no_grad():
        l2_diff = torch.norm(clean_flat - noisy_flat, p=2)
        l2_clean = torch.norm(clean_flat, p=2)
        relative_l2 = (l2_diff / l2_clean).item() if l2_clean > 1e-12 else 0.0

        cosine_similarity = F.cosine_similarity(
            clean_flat.unsqueeze(0), noisy_flat.unsqueeze(0)
        ).item()

        mean_shift = (noisy_flat.mean() - clean_flat.mean()).item()

        std_ratio = (
            (noisy_flat.std() / clean_flat.std()).item()
            if clean_flat.std() > 1e-12
            else 1.0
        )

        abs_errors = (clean_flat - noisy_flat).abs()
        max_abs_error = abs_errors.max().item()
        p95_error = abs_errors.kthvalue(int(abs_errors.numel() * 0.95)).values.item()

        sign_flip_ratio = (
            ((clean_flat.sign() != noisy_flat.sign()) & (clean_flat != 0) & (noisy_flat != 0))
            .float()
            .mean()
            .item()
        )

        zero_fraction_clean = (clean_flat == 0).float().mean().item()
        zero_fraction_noisy = (noisy_flat == 0).float().mean().item()
        zero_fraction_change = zero_fraction_noisy - zero_fraction_clean

        saturation_ratio = ((abs_errors / (clean_flat.abs() + 1e-12)) > 0.95).float().mean().item()

    return {
        "relative_l2": relative_l2,
        "cosine_similarity": cosine_similarity,
        "mean_shift": mean_shift,
        "std_ratio": std_ratio,
        "max_abs_error": max_abs_error,
        "p95_error": p95_error,
        "sign_flip_ratio": sign_flip_ratio,
        "zero_fraction_change": zero_fraction_change,
        "saturation_ratio": saturation_ratio,
    }
