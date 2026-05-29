from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class ReservoirConfig:
    input_dim: int
    reservoir_dim: int
    spectral_radius: float = 0.9
    input_scale: float = 0.1
    bias_scale: float = 0.01
    leak_rate: float = 0.5
    device: str | torch.device | None = None


def init_reservoir_buffers(generator: torch.Generator | None, config: ReservoirConfig) -> dict[str, torch.Tensor | float]:
    device = config.device
    W_in = torch.randn(config.input_dim, config.reservoir_dim, generator=generator, device=device) * config.input_scale
    W_res = torch.randn(config.reservoir_dim, config.reservoir_dim, generator=generator, device=device)
    eig = torch.linalg.eigvals(W_res).abs().max().real
    W_res = W_res * (config.spectral_radius / torch.clamp(eig, min=1.0e-8))
    b = torch.randn(config.reservoir_dim, generator=generator, device=device) * config.bias_scale
    return {
        "W_in": W_in.detach().requires_grad_(False),
        "W_res": W_res.detach().requires_grad_(False),
        "b": b.detach().requires_grad_(False),
        "leak_rate": float(config.leak_rate),
    }


def reservoir_update(features_t: torch.Tensor, h_prev: torch.Tensor, buffers: dict[str, torch.Tensor | float]) -> torch.Tensor:
    W_in = buffers["W_in"]
    W_res = buffers["W_res"]
    b = buffers["b"]
    leak_rate = float(buffers["leak_rate"])
    h_prev = h_prev.detach()
    drive = features_t.detach() @ W_in + h_prev @ W_res + b
    h_t = (1.0 - leak_rate) * h_prev + leak_rate * torch.tanh(drive)
    return h_t.detach()


def reservoir_observation(features_t: torch.Tensor, h_t: torch.Tensor) -> torch.Tensor:
    return torch.cat([features_t.detach(), h_t.detach()], dim=1)
