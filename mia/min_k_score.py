from __future__ import annotations

import math
import torch


def min_k_percent_score(
    correct_log_probs: torch.Tensor,
    mask: torch.Tensor,
    *,
    k_percent: float = 20.0,
    ignore_bos: bool = True,
    min_tokens: int = 2,
) -> float:
    values = correct_log_probs.reshape(-1)
    mask_flat = mask.reshape(-1).to(torch.float32)

    if ignore_bos and mask_flat.numel() > 0:
        mask_flat = mask_flat.clone()
        mask_flat[0] = 0.0

    valid_values = values[mask_flat > 0.5]

    if valid_values.numel() < min_tokens:
        return 0.0

    k = max(1, math.ceil(valid_values.numel() * (k_percent / 100.0)))

    lowest_values, _ = torch.topk(valid_values, k=k, largest=False)

    return float(lowest_values.mean().item())