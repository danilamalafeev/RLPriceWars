from __future__ import annotations

from dataclasses import dataclass

import torch

from neural.functional_policies import params_iter


@dataclass(frozen=True)
class OptimizerConfig:
    lr: float = 1.0e-2
    gamma: float = 0.95
    entropy_coef: float = 0.0


def discounted_returns(rewards: torch.Tensor, gamma: float) -> torch.Tensor:
    if rewards.ndim != 3:
        raise ValueError("rewards must have shape [T, B, 2]")
    out = torch.zeros_like(rewards)
    running = torch.zeros_like(rewards[0])
    for t in range(rewards.shape[0] - 1, -1, -1):
        running = rewards[t] + gamma * running
        out[t] = running
    return out


def normalize_advantages(A: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    mean = A.mean()
    std = A.std(unbiased=False)
    return (A - mean) / (std + eps)


def reinforce_loss(logp: torch.Tensor, advantages: torch.Tensor, entropy: torch.Tensor | None = None, entropy_coef: float = 0.0) -> torch.Tensor:
    adv = advantages.detach()
    loss = -(logp * adv).mean()
    if entropy is not None and entropy_coef != 0.0:
        loss = loss - entropy_coef * entropy.mean()
    return loss


def _zero_grads(params) -> None:
    for p in params_iter(params):
        if p.grad is not None:
            p.grad.zero_()


def _sgd_step(params, lr: float) -> bool:
    changed = False
    with torch.no_grad():
        for p in params_iter(params):
            if p.grad is None:
                continue
            before = p.detach().clone()
            p -= lr * p.grad
            changed = changed or bool(torch.any(before != p).item())
    return changed


def train_pg_step(oracle_params, victim_params, rollout: dict[str, torch.Tensor], optimizer_config: OptimizerConfig | dict) -> dict[str, float]:
    if isinstance(optimizer_config, dict):
        optimizer_config = OptimizerConfig(**optimizer_config)

    returns = discounted_returns(rollout["rewards"], optimizer_config.gamma)
    advantages = normalize_advantages(returns.detach())

    _zero_grads(oracle_params)
    _zero_grads(victim_params)
    oracle_loss = reinforce_loss(
        rollout["logp"][:, :, 0],
        advantages[:, :, 0],
        rollout.get("entropy", None)[:, :, 0] if "entropy" in rollout else None,
        optimizer_config.entropy_coef,
    )
    victim_loss = reinforce_loss(
        rollout["logp"][:, :, 1],
        advantages[:, :, 1],
        rollout.get("entropy", None)[:, :, 1] if "entropy" in rollout else None,
        optimizer_config.entropy_coef,
    )
    total_loss = oracle_loss + victim_loss
    total_loss.backward()
    oracle_changed = _sgd_step(oracle_params, optimizer_config.lr)
    victim_changed = _sgd_step(victim_params, optimizer_config.lr)

    return {
        "loss_oracle": float(oracle_loss.detach().item()),
        "loss_victim": float(victim_loss.detach().item()),
        "loss_total": float(total_loss.detach().item()),
        "avg_return_oracle": float(returns[:, :, 0].mean().item()),
        "avg_return_victim": float(returns[:, :, 1].mean().item()),
        "oracle_changed": float(oracle_changed),
        "victim_changed": float(victim_changed),
    }


def train_pg_step_dual_lr(
    oracle_params,
    victim_params,
    rollout: dict[str, torch.Tensor],
    gamma: float,
    lr_oracle: float,
    lr_victim: float,
    entropy_coef: float = 0.0,
) -> dict[str, float]:
    returns = discounted_returns(rollout["rewards"], gamma)
    advantages = normalize_advantages(returns.detach())

    _zero_grads(oracle_params)
    _zero_grads(victim_params)
    oracle_entropy = rollout["entropy"][:, :, 0] if "entropy" in rollout else None
    victim_entropy = rollout["entropy"][:, :, 1] if "entropy" in rollout else None
    oracle_loss = reinforce_loss(rollout["logp"][:, :, 0], advantages[:, :, 0], oracle_entropy, entropy_coef)
    victim_loss = reinforce_loss(rollout["logp"][:, :, 1], advantages[:, :, 1], victim_entropy, entropy_coef)
    total_loss = oracle_loss + victim_loss
    total_loss.backward()
    oracle_changed = _sgd_step(oracle_params, lr_oracle)
    victim_changed = _sgd_step(victim_params, lr_victim)

    return {
        "loss_oracle": float(oracle_loss.detach().item()),
        "loss_victim": float(victim_loss.detach().item()),
        "loss_total": float(total_loss.detach().item()),
        "avg_return_oracle": float(returns[:, :, 0].mean().item()),
        "avg_return_victim": float(returns[:, :, 1].mean().item()),
        "oracle_changed": float(oracle_changed),
        "victim_changed": float(victim_changed),
    }


def _explained_variance(values: torch.Tensor, returns: torch.Tensor) -> torch.Tensor:
    target = returns.detach()
    var_y = torch.var(target, unbiased=False)
    if float(var_y.detach().item()) < 1e-12:
        return torch.tensor(0.0, dtype=values.dtype, device=values.device)
    return 1.0 - torch.var(target - values.detach(), unbiased=False) / var_y


def actor_critic_loss(
    logp: torch.Tensor,
    values: torch.Tensor,
    returns: torch.Tensor,
    entropy: torch.Tensor | None = None,
    entropy_coef: float = 0.0,
    value_coef: float = 0.5,
) -> dict[str, torch.Tensor]:
    advantages = returns.detach() - values.detach()
    norm_advantages = normalize_advantages(advantages)
    policy_loss = -(logp * norm_advantages).mean()
    value_loss = torch.mean((values - returns.detach()) ** 2)
    entropy_loss = torch.zeros((), dtype=logp.dtype, device=logp.device)
    if entropy is not None and entropy_coef != 0.0:
        entropy_loss = -entropy_coef * entropy.mean()
    loss = policy_loss + value_coef * value_loss + entropy_loss
    return {
        "loss": loss,
        "policy_loss": policy_loss,
        "value_loss": value_loss,
        "entropy_loss": entropy_loss,
        "explained_variance": _explained_variance(values, returns),
    }


def train_ac_step_dual_lr(
    oracle_policy_params,
    victim_policy_params,
    oracle_value_params,
    victim_value_params,
    rollout: dict[str, torch.Tensor],
    gamma: float,
    lr_policy_oracle: float,
    lr_policy_victim: float,
    lr_value_oracle: float,
    lr_value_victim: float,
    entropy_coef: float = 0.0,
    value_coef: float = 0.5,
) -> dict[str, float]:
    if "values" not in rollout:
        raise ValueError("actor-critic training requires rollout['values']")
    returns = discounted_returns(rollout["rewards"], gamma)

    _zero_grads(oracle_policy_params)
    _zero_grads(victim_policy_params)
    _zero_grads(oracle_value_params)
    _zero_grads(victim_value_params)

    oracle = actor_critic_loss(
        rollout["logp"][:, :, 0],
        rollout["values"][:, :, 0],
        returns[:, :, 0],
        rollout.get("entropy", None)[:, :, 0] if "entropy" in rollout else None,
        entropy_coef,
        value_coef,
    )
    victim = actor_critic_loss(
        rollout["logp"][:, :, 1],
        rollout["values"][:, :, 1],
        returns[:, :, 1],
        rollout.get("entropy", None)[:, :, 1] if "entropy" in rollout else None,
        entropy_coef,
        value_coef,
    )
    total_loss = oracle["loss"] + victim["loss"]
    total_loss.backward()
    oracle_policy_changed = _sgd_step(oracle_policy_params, lr_policy_oracle)
    victim_policy_changed = _sgd_step(victim_policy_params, lr_policy_victim)
    oracle_value_changed = _sgd_step(oracle_value_params, lr_value_oracle)
    victim_value_changed = _sgd_step(victim_value_params, lr_value_victim)

    return {
        "loss_oracle": float(oracle["loss"].detach().item()),
        "loss_victim": float(victim["loss"].detach().item()),
        "loss_total": float(total_loss.detach().item()),
        "policy_loss_oracle": float(oracle["policy_loss"].detach().item()),
        "policy_loss_victim": float(victim["policy_loss"].detach().item()),
        "value_loss_oracle": float(oracle["value_loss"].detach().item()),
        "value_loss_victim": float(victim["value_loss"].detach().item()),
        "entropy_loss_oracle": float(oracle["entropy_loss"].detach().item()),
        "entropy_loss_victim": float(victim["entropy_loss"].detach().item()),
        "explained_variance_oracle": float(oracle["explained_variance"].detach().item()),
        "explained_variance_victim": float(victim["explained_variance"].detach().item()),
        "avg_return_oracle": float(returns[:, :, 0].mean().item()),
        "avg_return_victim": float(returns[:, :, 1].mean().item()),
        "oracle_changed": float(oracle_policy_changed),
        "victim_changed": float(victim_policy_changed),
        "oracle_value_changed": float(oracle_value_changed),
        "victim_value_changed": float(victim_value_changed),
    }
