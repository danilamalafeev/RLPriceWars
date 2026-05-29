from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class ObservationConfig:
    price_min: float | None = None
    price_max: float | None = None
    include_time: bool = False
    max_time: float = 1.0
    device: str | torch.device | None = None


def _detached_float_tensor(x, device=None) -> torch.Tensor:
    return torch.as_tensor(x, dtype=torch.float32, device=device).detach()


def _price_center_scale(price_history: torch.Tensor, config: ObservationConfig) -> tuple[torch.Tensor, torch.Tensor]:
    if config.price_min is not None and config.price_max is not None:
        low = torch.tensor(float(config.price_min), dtype=price_history.dtype, device=price_history.device)
        high = torch.tensor(float(config.price_max), dtype=price_history.dtype, device=price_history.device)
    else:
        low = torch.amin(price_history)
        high = torch.amax(price_history)
    scale = torch.clamp(high - low, min=1.0e-6)
    center = 0.5 * (high + low)
    return center, scale


def build_observation(
    price_history,
    current_prices,
    rewards,
    market_share,
    outside_share,
    margins,
    config: ObservationConfig | None = None,
    time_step: int | float | None = None,
) -> torch.Tensor:
    del rewards, outside_share
    config = config or ObservationConfig()
    device = config.device

    ph = _detached_float_tensor(price_history, device=device)
    cp = _detached_float_tensor(current_prices, device=device)
    ms = _detached_float_tensor(market_share, device=device)
    mg = _detached_float_tensor(margins, device=device)

    if ph.ndim != 3 or ph.shape[-1] != 2:
        raise ValueError("price_history must have shape [B, H, 2]")
    if cp.shape != (ph.shape[0], 2):
        raise ValueError("current_prices must have shape [B, 2]")

    center, scale = _price_center_scale(ph, config)
    norm_history = ((ph - center) / scale).reshape(ph.shape[0], -1)
    deltas = ((ph[:, 1:, :] - ph[:, :-1, :]) / scale).reshape(ph.shape[0], -1)
    price_gap = (torch.abs(cp[:, 0:1] - cp[:, 1:2]) / scale)
    mean_price = ((cp.mean(dim=1, keepdim=True) - center) / scale)
    norm_margins = mg / scale

    parts = [norm_history, deltas, price_gap, mean_price, ms, norm_margins]
    if config.include_time:
        t = 0.0 if time_step is None else float(time_step)
        time_feature = torch.full((ph.shape[0], 1), t / max(float(config.max_time), 1.0), dtype=torch.float32, device=ph.device)
        parts.append(time_feature)

    return torch.cat(parts, dim=1)


def observation_dim(history_length: int, include_time: bool = False) -> int:
    return history_length * 2 + max(history_length - 1, 0) * 2 + 1 + 1 + 2 + 2 + (1 if include_time else 0)
