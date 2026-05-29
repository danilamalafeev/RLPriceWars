from __future__ import annotations

import math

import torch


def init_linear_policy(
    generator: torch.Generator | None,
    obs_dim: int,
    K: int,
    scale: float = 0.01,
    device: str | torch.device | None = None,
) -> dict[str, torch.Tensor]:
    W = torch.randn(obs_dim, K, generator=generator, device=device) * scale
    b = torch.zeros(K, device=device)
    return {"W": W.requires_grad_(True), "b": b.requires_grad_(True)}


def init_mlp_policy(
    generator: torch.Generator | None,
    obs_dim: int,
    hidden_dim: int,
    K: int,
    scale: float = 0.05,
    device: str | torch.device | None = None,
) -> dict[str, torch.Tensor]:
    W1 = torch.randn(obs_dim, hidden_dim, generator=generator, device=device) * (scale / math.sqrt(max(obs_dim, 1)))
    b1 = torch.zeros(hidden_dim, device=device)
    W2 = torch.randn(hidden_dim, K, generator=generator, device=device) * (scale / math.sqrt(max(hidden_dim, 1)))
    b2 = torch.zeros(K, device=device)
    return {
        "W1": W1.requires_grad_(True),
        "b1": b1.requires_grad_(True),
        "W2": W2.requires_grad_(True),
        "b2": b2.requires_grad_(True),
    }


def linear_policy_forward(params, buffers, obs: torch.Tensor, state=None):
    del buffers
    return obs @ params["W"] + params["b"], state


def mlp_policy_forward(params, buffers, obs: torch.Tensor, state=None):
    del buffers
    x = torch.tanh(obs @ params["W1"] + params["b1"])
    return x @ params["W2"] + params["b2"], state


def init_linear_value(
    generator: torch.Generator | None,
    obs_dim: int,
    scale: float = 0.01,
    device: str | torch.device | None = None,
) -> dict[str, torch.Tensor]:
    W = torch.randn(obs_dim, 1, generator=generator, device=device) * scale
    b = torch.zeros(1, device=device)
    return {"W": W.requires_grad_(True), "b": b.requires_grad_(True)}


def linear_value_forward(params, obs: torch.Tensor) -> torch.Tensor:
    return (obs @ params["W"] + params["b"]).squeeze(-1)


def init_mlp_value(
    generator: torch.Generator | None,
    obs_dim: int,
    hidden_dim: int,
    scale: float = 0.05,
    device: str | torch.device | None = None,
) -> dict[str, torch.Tensor]:
    W1 = torch.randn(obs_dim, hidden_dim, generator=generator, device=device) * (scale / math.sqrt(max(obs_dim, 1)))
    b1 = torch.zeros(hidden_dim, device=device)
    W2 = torch.randn(hidden_dim, 1, generator=generator, device=device) * (scale / math.sqrt(max(hidden_dim, 1)))
    b2 = torch.zeros(1, device=device)
    return {
        "W1": W1.requires_grad_(True),
        "b1": b1.requires_grad_(True),
        "W2": W2.requires_grad_(True),
        "b2": b2.requires_grad_(True),
    }


def mlp_value_forward(params, obs: torch.Tensor) -> torch.Tensor:
    x = torch.tanh(obs @ params["W1"] + params["b1"])
    return (x @ params["W2"] + params["b2"]).squeeze(-1)


def params_iter(params):
    if isinstance(params, dict):
        return params.values()
    if isinstance(params, (tuple, list)):
        return params
    raise TypeError("params must be a dict/list/tuple pytree of tensors")
