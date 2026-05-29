from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

import calvano_market_cpp as cm
from calvano_market import CalvanoMarketConfig, StaticBenchmarks, build_static_benchmarks
from neural.functional_policies import init_mlp_policy, init_mlp_value, mlp_policy_forward, mlp_value_forward
from neural.losses import actor_critic_loss, discounted_returns
from neural.observations import ObservationConfig, build_observation, observation_dim
from neural.reservoir import ReservoirConfig, init_reservoir_buffers, reservoir_observation, reservoir_update


@dataclass(frozen=True)
class QVictimOracleConfig:
    oracle_kind: str = "dqn"  # "actor_critic", "dqn", "dqn_jepa", "dqn_regret", "tabular_cfr", "tabular_multi_cfr", "tabular_lola", "tabular_model_lola", or "tabular_rollout_lola"
    seed: int = 0
    B: int = 64
    H: int = 8
    K: int = 15
    total_steps: int = 50_000
    rollout_steps: int = 32
    eval_every: int = 5_000
    eval_steps: int = 5_000
    log_every: int = 1_000

    victim_alpha: float = 0.15
    victim_beta: float = 4e-6
    victim_delta: float = 0.95

    gamma: float = 0.95
    lr: float = 1e-3
    lr_value: float = 1e-3
    hidden_dim: int = 128
    value_hidden_dim: int = 128
    value_coef: float = 0.5
    entropy_coef: float = 0.0
    reservoir_dim: int = 256
    reservoir_spectral_radius: float = 0.9
    reservoir_leak_rate: float = 0.5

    replay_capacity: int = 100_000
    batch_size: int = 256
    train_every: int = 4
    target_update_every: int = 1_000
    oracle_epsilon_start: float = 1.0
    oracle_epsilon_end: float = 0.05
    oracle_epsilon_decay_steps: int = 50_000
    jepa_latent_dim: int = 64
    jepa_coef: float = 0.1
    regret_coef: float = 0.1
    cfr_state_mode: str = "joint_last_action"
    cfr_regret_decay: float = 1.0
    cfr_value_lr: float = 0.1
    cfr_gamma: float = 0.95
    lola_gamma: float = 0.95
    lola_tau: float = 0.05
    lola_epsilon: float = 0.05
    lola_state_mode: str = "joint_last_action"
    model_lola_gamma: float = 0.95
    model_lola_tau: float = 0.05
    model_lola_epsilon: float = 0.02
    model_lola_victim_policy: str = "epsilon_greedy"
    model_lola_future_policy: str = "epsilon_greedy"
    model_lola_victim_softmax_tau: float = 0.05
    rollout_lola_horizon: int = 20
    rollout_lola_num_particles: int = 32
    rollout_lola_tau: float = 0.05
    rollout_lola_epsilon: float = 0.02
    rollout_lola_victim_policy: str = "epsilon_greedy"
    rollout_lola_oracle_rollout_policy: str = "greedy_best_response"
    rollout_lola_discount: float = 0.95
    rollout_lola_include_immediate: bool = True

    asymmetry_coef: float = 0.0
    device: str = "cpu"
    out_dir: str | None = None


def make_calvano_vec_env(B: int, H: int, K: int, seed: int):
    market_config = CalvanoMarketConfig(m=K)
    benchmarks = build_static_benchmarks(market_config)
    price_grid = benchmarks.price_grid.astype(np.float32)
    env_config = {
        "B": B,
        "A": 2,
        "K": K,
        "H": H,
        "price_grid": price_grid,
        "qualities": np.array([2.0, 2.0], dtype=np.float32),
        "costs": np.array([1.0, 1.0], dtype=np.float32),
        "outside_quality": 0.0,
        "mu": 0.25,
        "demand_scale": 1.0,
        "random_seed": int(seed),
    }
    env = cm.create_env(env_config)
    cm.reset(env, int(seed))
    return env, price_grid, benchmarks, benchmarks.profit_matrix


def oracle_epsilon(config: QVictimOracleConfig, step: int) -> float:
    progress = min(float(step) / max(float(config.oracle_epsilon_decay_steps), 1.0), 1.0)
    return float(config.oracle_epsilon_start + progress * (config.oracle_epsilon_end - config.oracle_epsilon_start))


def victim_epsilon(beta: float, t: int | np.ndarray) -> np.ndarray:
    return np.exp(-beta * np.asarray(t, dtype=np.float64))


def init_victim_state(
    B: int,
    K: int,
    benchmarks_or_profit_matrix: StaticBenchmarks | np.ndarray,
    delta: float,
    rng: np.random.Generator | None = None,
    seed: int | None = None,
) -> dict[str, Any]:
    if rng is None:
        rng = np.random.default_rng(seed)
    profit_matrix = (
        benchmarks_or_profit_matrix.profit_matrix
        if isinstance(benchmarks_or_profit_matrix, StaticBenchmarks)
        else np.asarray(benchmarks_or_profit_matrix)
    )
    S = K * K
    q0 = np.zeros((S, K), dtype=np.float64)
    for a_v in range(K):
        q0[:, a_v] = float(np.mean(profit_matrix[:, a_v, 1])) / (1.0 - delta)
    return {
        "Q": np.repeat(q0[None, :, :], B, axis=0),
        "state_id": rng.integers(0, S, size=B, dtype=np.int64),
        "t": np.zeros(B, dtype=np.int64),
    }


def victim_select_actions(
    victim: dict[str, Any],
    K: int,
    beta: float,
    rng: np.random.Generator,
    greedy: bool = False,
    epsilon_override: float | None = None,
) -> np.ndarray:
    B = victim["Q"].shape[0]
    if epsilon_override is not None:
        eps = np.full(B, float(epsilon_override), dtype=np.float64)
    elif greedy:
        eps = np.zeros(B, dtype=np.float64)
    else:
        eps = victim_epsilon(beta, victim["t"])
    state = victim["state_id"]
    q = victim["Q"][np.arange(B), state, :]
    greedy_actions = np.argmax(q, axis=1).astype(np.int64)
    random_actions = rng.integers(0, K, size=B, dtype=np.int64)
    explore = rng.random(B) < eps
    return np.where(explore, random_actions, greedy_actions).astype(np.int64)


def update_victim_q(
    victim: dict[str, Any],
    oracle_actions: np.ndarray,
    victim_actions: np.ndarray,
    rewards_victim: np.ndarray,
    alpha: float,
    delta: float,
    K: int,
) -> None:
    B = victim["Q"].shape[0]
    old_state = victim["state_id"].copy()
    next_state = (oracle_actions.astype(np.int64) * K + victim_actions.astype(np.int64)).astype(np.int64)
    rows = np.arange(B)
    old = victim["Q"][rows, old_state, victim_actions]
    next_max = np.max(victim["Q"][rows, next_state, :], axis=1)
    target = rewards_victim + delta * next_max
    victim["Q"][rows, old_state, victim_actions] = (1.0 - alpha) * old + alpha * target
    victim["state_id"] = next_state
    victim["t"] += 1


def victim_q_update(
    state: dict[str, Any],
    state_id: np.ndarray,
    victim_actions: np.ndarray,
    rewards_victim: np.ndarray,
    next_state_id: np.ndarray,
    alpha: float,
    delta: float,
) -> None:
    B = state["Q"].shape[0]
    rows = np.arange(B)
    old = state["Q"][rows, state_id, victim_actions]
    next_max = np.max(state["Q"][rows, next_state_id, :], axis=1)
    target = rewards_victim + delta * next_max
    state["Q"][rows, state_id, victim_actions] = (1.0 - alpha) * old + alpha * target
    state["state_id"] = next_state_id.astype(np.int64)
    state["t"] += 1


def victim_policy_from_q(
    victim: dict[str, Any],
    K: int,
    beta: float | None = None,
    greedy: bool = True,
) -> np.ndarray:
    q = victim["Q"][np.arange(victim["Q"].shape[0]), victim["state_id"], :]
    greedy_actions = np.argmax(q, axis=1).astype(np.int64)
    if greedy:
        return greedy_actions
    eps = victim_epsilon(0.0 if beta is None else beta, victim["t"])
    uniform = np.full((q.shape[0], K), 1.0 / K, dtype=np.float64)
    probs = eps[:, None] * uniform
    probs[np.arange(q.shape[0]), greedy_actions] += 1.0 - eps
    return np.asarray([np.random.choice(K, p=probs[b]) for b in range(q.shape[0])], dtype=np.int64)


def victim_policy_probs_from_q(
    q_values: np.ndarray,
    K: int,
    mode: str,
    epsilon: np.ndarray | float | None = None,
    tau: float = 0.05,
) -> np.ndarray:
    q = np.asarray(q_values, dtype=np.float64)
    if q.shape[-1] != K:
        raise ValueError(f"q_values last dimension must be K={K}, got {q.shape[-1]}")
    if mode == "greedy":
        greedy_actions = np.argmax(q, axis=-1)
        probs = np.zeros_like(q, dtype=np.float64)
        np.put_along_axis(probs, greedy_actions[..., None], 1.0, axis=-1)
        return probs
    if mode == "epsilon_greedy":
        greedy_actions = np.argmax(q, axis=-1)
        if epsilon is None:
            eps = np.zeros(q.shape[:-1], dtype=np.float64)
        else:
            eps = np.asarray(epsilon, dtype=np.float64)
            if eps.shape == ():
                eps = np.full(q.shape[:-1], float(eps), dtype=np.float64)
            else:
                while eps.ndim < q.ndim - 1:
                    eps = eps[..., None]
                eps = np.broadcast_to(eps, q.shape[:-1]).astype(np.float64, copy=False)
        eps = np.clip(eps, 0.0, 1.0)
        probs = np.broadcast_to((eps / float(K))[..., None], q.shape).astype(np.float64, copy=True)
        np.put_along_axis(probs, greedy_actions[..., None], np.take_along_axis(probs, greedy_actions[..., None], axis=-1) + (1.0 - eps)[..., None], axis=-1)
        return probs
    if mode == "softmax":
        denom = max(float(tau), 1e-12)
        z = q / denom
        z = z - np.max(z, axis=-1, keepdims=True)
        exp_z = np.exp(z)
        return exp_z / np.clip(np.sum(exp_z, axis=-1, keepdims=True), 1e-12, None)
    raise ValueError(f"unknown victim policy mode: {mode}")


def init_tabular_cfr_state(B: int, K: int, state_mode: str, device: torch.device) -> dict[str, torch.Tensor]:
    if state_mode == "victim_last_action":
        S = K
    elif state_mode == "joint_last_action":
        S = K * K
    else:
        raise ValueError(f"unknown cfr_state_mode: {state_mode}")
    return {
        "regret_table": torch.zeros((B, S, K), dtype=torch.float32, device=device),
        "value_table": torch.zeros((B, S), dtype=torch.float32, device=device),
        "state_id": torch.zeros(B, dtype=torch.long, device=device),
    }


def tabular_cfr_state_id(
    oracle_actions: np.ndarray,
    victim_actions: np.ndarray,
    K: int,
    state_mode: str,
    device: torch.device,
) -> torch.Tensor:
    oracle_actions = np.asarray(oracle_actions, dtype=np.int64)
    victim_actions = np.asarray(victim_actions, dtype=np.int64)
    if state_mode == "victim_last_action":
        state_id = victim_actions
    elif state_mode == "joint_last_action":
        state_id = oracle_actions * K + victim_actions
    else:
        raise ValueError(f"unknown cfr_state_mode: {state_mode}")
    return torch.as_tensor(state_id, dtype=torch.long, device=device)


def tabular_cfr_strategy_probs(cfr_state: dict[str, torch.Tensor], K: int, epsilon: float = 0.0) -> torch.Tensor:
    regret_table = cfr_state["regret_table"]
    state_id = cfr_state["state_id"]
    B = regret_table.shape[0]
    device = regret_table.device
    rows = torch.arange(B, device=device)
    regrets = regret_table[rows, state_id, :]
    positive = torch.clamp(regrets, min=0.0)
    sums = positive.sum(dim=1, keepdim=True)
    uniform = torch.full((B, K), 1.0 / K, dtype=regret_table.dtype, device=device)
    probs = torch.where(sums > 0.0, positive / torch.clamp(sums, min=1e-12), uniform)
    if epsilon > 0.0:
        probs = (1.0 - float(epsilon)) * probs + float(epsilon) * uniform
    return probs


def tabular_cfr_select_actions(
    cfr_state: dict[str, torch.Tensor],
    K: int,
    epsilon: float,
    generator: torch.Generator,
) -> torch.Tensor:
    probs = tabular_cfr_strategy_probs(cfr_state, K, epsilon)
    return torch.multinomial(probs, 1, generator=generator).squeeze(1).to(torch.long)


def oracle_counterfactual_profit(
    profit_matrix: np.ndarray,
    victim_actions: np.ndarray,
    device: torch.device,
) -> torch.Tensor:
    victim_actions = np.asarray(victim_actions, dtype=np.int64)
    cf_profit = np.asarray(profit_matrix[:, victim_actions, 0], dtype=np.float32).T
    return torch.as_tensor(cf_profit, dtype=torch.float32, device=device)


def victim_counterfactual_profit(
    profit_matrix: np.ndarray,
    victim_actions: np.ndarray,
    device: torch.device,
) -> torch.Tensor:
    victim_actions = np.asarray(victim_actions, dtype=np.int64)
    cf_profit = np.asarray(profit_matrix[:, victim_actions, 1], dtype=np.float32).T
    return torch.as_tensor(cf_profit, dtype=torch.float32, device=device)


def counterfactual_victim_q_after_update(
    victim: dict[str, Any],
    oracle_actions_cf: np.ndarray,
    victim_actions: np.ndarray,
    rewards_victim_cf: np.ndarray,
    alpha: float,
    delta: float,
    K: int,
) -> np.ndarray:
    Q = victim["Q"]
    B = Q.shape[0]
    oracle_actions_cf = np.asarray(oracle_actions_cf, dtype=np.int64)
    rewards_victim_cf = np.asarray(rewards_victim_cf, dtype=np.float64)
    if oracle_actions_cf.ndim == 1:
        oracle_actions_cf = oracle_actions_cf.reshape(B, 1)
    if rewards_victim_cf.ndim == 1:
        rewards_victim_cf = rewards_victim_cf.reshape(B, 1)
    victim_actions = np.asarray(victim_actions, dtype=np.int64)
    old_state = victim["state_id"].astype(np.int64)
    rows = np.arange(B)[:, None]
    next_state_cf = oracle_actions_cf * K + victim_actions[:, None]
    old_q = Q[np.arange(B), old_state, victim_actions][:, None]
    next_max = np.max(Q[rows, next_state_cf, :], axis=2)
    target = rewards_victim_cf + delta * next_max
    return (1.0 - alpha) * old_q + alpha * target


def tabular_cfr_counterfactual_next_state_ids(
    oracle_actions_all: torch.Tensor,
    victim_actions: np.ndarray,
    K: int,
    state_mode: str,
    device: torch.device,
) -> torch.Tensor:
    victim_actions_t = torch.as_tensor(victim_actions, dtype=torch.long, device=device)
    B = victim_actions_t.shape[0]
    if state_mode == "victim_last_action":
        return victim_actions_t.view(B, 1).expand(B, K)
    if state_mode != "joint_last_action":
        raise ValueError(f"unknown cfr_state_mode: {state_mode}")
    oracle_actions_all = oracle_actions_all.to(device=device, dtype=torch.long)
    if oracle_actions_all.ndim == 1:
        oracle_actions_all = oracle_actions_all.view(1, K).expand(B, K)
    elif oracle_actions_all.shape != (B, K):
        raise ValueError(f"oracle_actions_all must have shape [K] or [B, K], got {tuple(oracle_actions_all.shape)}")
    return oracle_actions_all * K + victim_actions_t.view(B, 1)


def tabular_multi_cfr_cf_value(
    cfr_state: dict[str, torch.Tensor],
    cf_profit: torch.Tensor,
    next_state_cf: torch.Tensor,
    gamma: float,
) -> torch.Tensor:
    B, K = cf_profit.shape
    device = cf_profit.device
    rows = torch.arange(B, device=device).unsqueeze(1).expand(B, K)
    future_v = cfr_state["value_table"][rows, next_state_cf]
    return cf_profit + float(gamma) * future_v


def tabular_cfr_update(
    cfr_state: dict[str, torch.Tensor],
    prev_state_id: torch.Tensor,
    oracle_actions: torch.Tensor,
    cf_profit: torch.Tensor,
    regret_decay: float,
) -> None:
    regret_table = cfr_state["regret_table"]
    B = regret_table.shape[0]
    rows = torch.arange(B, device=regret_table.device)
    oracle_actions = oracle_actions.to(device=regret_table.device, dtype=torch.long)
    prev_state_id = prev_state_id.to(device=regret_table.device, dtype=torch.long)
    actual_profit = cf_profit[rows, oracle_actions]
    regret_vector = cf_profit - actual_profit.unsqueeze(1)
    with torch.no_grad():
        regret_table[rows, prev_state_id, :] *= float(regret_decay)
        regret_table[rows, prev_state_id, :] += regret_vector


def tabular_multi_cfr_value_update(
    cfr_state: dict[str, torch.Tensor],
    prev_state_id: torch.Tensor,
    rewards_oracle: torch.Tensor,
    next_state_real: torch.Tensor,
    value_lr: float,
    gamma: float,
) -> None:
    value_table = cfr_state["value_table"]
    B = value_table.shape[0]
    rows = torch.arange(B, device=value_table.device)
    prev_state_id = prev_state_id.to(device=value_table.device, dtype=torch.long)
    next_state_real = next_state_real.to(device=value_table.device, dtype=torch.long)
    rewards_oracle = rewards_oracle.to(device=value_table.device, dtype=value_table.dtype)
    with torch.no_grad():
        target = rewards_oracle + float(gamma) * value_table[rows, next_state_real]
        old = value_table[rows, prev_state_id]
        value_table[rows, prev_state_id] = (1.0 - float(value_lr)) * old + float(value_lr) * target


def tabular_lola_select_actions(
    victim: dict[str, Any],
    profit_matrix: np.ndarray,
    K: int,
    gamma: float,
    tau: float,
    epsilon: float,
    generator: torch.Generator,
    device: torch.device,
    victim_alpha: float = 0.15,
    victim_delta: float = 0.95,
) -> tuple[torch.Tensor, dict[str, float]]:
    victim_actions_pred = victim_policy_from_q(victim, K, greedy=True)
    cf_profit_o = oracle_counterfactual_profit(profit_matrix, victim_actions_pred, device)
    cf_profit_v = victim_counterfactual_profit(profit_matrix, victim_actions_pred, device)
    oracle_actions_all = np.broadcast_to(np.arange(K, dtype=np.int64), (victim_actions_pred.shape[0], K))
    Q = victim["Q"]
    B = Q.shape[0]
    rows = np.arange(B)[:, None]
    next_state_cf = oracle_actions_all * K + victim_actions_pred[:, None]
    future_q = Q[rows, next_state_cf, :].copy()
    updated_entry = counterfactual_victim_q_after_update(
        victim,
        oracle_actions_all,
        victim_actions_pred,
        cf_profit_v.cpu().numpy(),
        alpha=victim_alpha,
        delta=victim_delta,
        K=K,
    )
    old_state = victim["state_id"].astype(np.int64)
    mask = next_state_cf == old_state[:, None]
    if np.any(mask):
        b_idx, a_idx = np.nonzero(mask)
        future_q[b_idx, a_idx, victim_actions_pred[b_idx]] = updated_entry[b_idx, a_idx]
    future_victim_action = np.argmax(future_q, axis=2).astype(np.int64)
    best_profit_by_victim_action = np.max(profit_matrix[:, :, 0], axis=0).astype(np.float32)
    future_value = torch.as_tensor(best_profit_by_victim_action[future_victim_action], dtype=torch.float32, device=device)
    lola_values = cf_profit_o + float(gamma) * future_value
    probs = torch.softmax(lola_values / max(float(tau), 1e-6), dim=1)
    if epsilon > 0.0:
        uniform = torch.full((B, K), 1.0 / K, dtype=probs.dtype, device=device)
        probs = (1.0 - float(epsilon)) * probs + float(epsilon) * uniform
    actions = torch.multinomial(probs, 1, generator=generator).squeeze(1).to(torch.long)
    entropy = -(probs * torch.clamp(probs, min=1e-12).log()).sum(dim=1)
    metrics = {
        "lola_immediate_value": float(cf_profit_o.gather(1, actions.view(-1, 1)).mean().detach().item()),
        "lola_future_value": float(future_value.gather(1, actions.view(-1, 1)).mean().detach().item()),
        "lola_total_value": float(lola_values.gather(1, actions.view(-1, 1)).mean().detach().item()),
        "lola_entropy": float(entropy.mean().detach().item()),
        "lola_value_mean": float(lola_values.mean().detach().item()),
    }
    return actions, metrics


def tabular_model_lola_values(
    victim: dict[str, Any],
    profit_matrix: np.ndarray,
    K: int,
    alpha: float,
    delta: float,
    beta: float,
    gamma_lola: float,
    victim_policy_mode: str,
    future_policy_mode: str,
    victim_softmax_tau: float,
) -> tuple[np.ndarray, dict[str, float]]:
    Q = victim["Q"]
    B = Q.shape[0]
    rows = np.arange(B)
    old_state = victim["state_id"].astype(np.int64)
    t = victim["t"]
    q_current = Q[rows, old_state, :]
    eps_current = np.exp(-float(beta) * np.asarray(t, dtype=np.float64))
    pi_v = victim_policy_probs_from_q(
        q_current,
        K,
        victim_policy_mode,
        epsilon=eps_current,
        tau=victim_softmax_tau,
    )

    oracle_actions = np.arange(K, dtype=np.int64)
    victim_actions = np.arange(K, dtype=np.int64)
    next_state = oracle_actions[:, None] * K + victim_actions[None, :]
    profit_o_now = np.asarray(profit_matrix[:, :, 0], dtype=np.float64)
    profit_v_now = np.asarray(profit_matrix[:, :, 1], dtype=np.float64)

    eps_next = np.exp(-float(beta) * (np.asarray(t, dtype=np.float64) + 1.0))
    pi_v_future_all = victim_policy_probs_from_q(
        Q,
        K,
        future_policy_mode,
        epsilon=eps_next,
        tau=victim_softmax_tau,
    )
    best_future_profit_o = np.max(np.asarray(profit_matrix[:, :, 0], dtype=np.float64), axis=0)
    future_o_value_all = np.sum(pi_v_future_all * best_future_profit_o.reshape((1, 1, K)), axis=2)
    future_o_value = future_o_value_all[rows[:, None, None], next_state[None, :, :]].copy()
    future_entropy_all = -np.sum(pi_v_future_all * np.log(np.clip(pi_v_future_all, 1e-12, 1.0)), axis=2)
    future_entropy = future_entropy_all[rows[:, None, None], next_state[None, :, :]].copy()

    a_o_same = old_state // K
    a_v_same = old_state % K
    old_q_same = q_current[rows, a_v_same]
    next_max_same = np.max(q_current, axis=1)
    target_same = profit_v_now[a_o_same, a_v_same] + float(delta) * next_max_same
    updated_same = (1.0 - float(alpha)) * old_q_same + float(alpha) * target_same
    q_same = q_current.copy()
    q_same[rows, a_v_same] = updated_same
    pi_same = victim_policy_probs_from_q(
        q_same,
        K,
        future_policy_mode,
        epsilon=eps_next,
        tau=victim_softmax_tau,
    )
    future_o_value[rows, a_o_same, a_v_same] = np.sum(pi_same * best_future_profit_o.reshape((1, K)), axis=1)
    future_entropy[rows, a_o_same, a_v_same] = -np.sum(pi_same * np.log(np.clip(pi_same, 1e-12, 1.0)), axis=1)

    candidate_value_given_av = profit_o_now[None, :, :] + float(gamma_lola) * future_o_value
    lola_values = np.sum(pi_v[:, None, :] * candidate_value_given_av, axis=2)

    current_entropy = -np.sum(pi_v * np.log(np.clip(pi_v, 1e-12, 1.0)), axis=1)
    immediate_values = np.sum(pi_v[:, None, :] * profit_o_now[None, :, :], axis=2)
    future_values = np.sum(pi_v[:, None, :] * future_o_value, axis=2)
    metrics = {
        "model_lola_immediate_value": float(np.mean(immediate_values)),
        "model_lola_future_value": float(np.mean(future_values)),
        "model_lola_total_value": float(np.mean(lola_values)),
        "model_lola_current_victim_entropy": float(np.mean(current_entropy)),
        "model_lola_future_victim_entropy": float(np.mean(future_entropy)),
    }
    return lola_values.astype(np.float64, copy=False), metrics


def tabular_model_lola_select_actions(
    lola_values: np.ndarray,
    tau: float,
    epsilon: float,
    generator: torch.Generator,
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, float]]:
    values_t = torch.as_tensor(lola_values, dtype=torch.float32, device=device)
    probs = torch.softmax(values_t / max(float(tau), 1e-6), dim=-1)
    uniform = torch.full_like(probs, 1.0 / probs.shape[-1])
    probs = (1.0 - float(epsilon)) * probs + float(epsilon) * uniform
    actions = torch.multinomial(probs, 1, generator=generator).squeeze(1).to(torch.long)
    entropy = -(probs * torch.clamp(probs, min=1e-12).log()).sum(dim=1)
    metrics = {
        "model_lola_entropy": float(entropy.mean().detach().item()),
        "model_lola_value_mean": float(values_t.mean().detach().item()),
        "model_lola_value_std": float(values_t.std(unbiased=False).detach().item()),
    }
    return actions, metrics


def tabular_rollout_lola_values(
    victim: dict[str, Any],
    profit_matrix: np.ndarray,
    K: int,
    alpha: float,
    delta: float,
    beta: float,
    horizon: int,
    num_particles: int,
    victim_policy_mode: str,
    oracle_rollout_policy: str,
    discount: float,
    include_immediate: bool,
    rng: np.random.Generator,
    price_grid: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, float]]:
    Q = np.asarray(victim["Q"], dtype=np.float64)
    B, S, _ = Q.shape
    horizon = max(int(horizon), 1)
    num_particles = max(int(num_particles), 1)
    rows_b = np.arange(B)[:, None]
    rows_p = np.arange(num_particles)[None, :]
    profit_o_matrix = np.asarray(profit_matrix[:, :, 0], dtype=np.float64)
    profit_v_matrix = np.asarray(profit_matrix[:, :, 1], dtype=np.float64)
    prices = None if price_grid is None else np.asarray(price_grid, dtype=np.float64)

    values = np.zeros((B, K), dtype=np.float64)
    first_profit_acc = np.zeros((B, K), dtype=np.float64)
    future_profit_acc = np.zeros((B, K), dtype=np.float64)
    oracle_price_acc = np.zeros((B, K), dtype=np.float64)
    victim_price_acc = np.zeros((B, K), dtype=np.float64)

    old_state = np.asarray(victim["state_id"], dtype=np.int64)
    old_t = np.asarray(victim["t"], dtype=np.int64)

    for a0 in range(K):
        Q_clone = np.repeat(Q[:, None, :, :], num_particles, axis=1).copy()
        state_clone = np.repeat(old_state[:, None], num_particles, axis=1)
        t_clone = np.repeat(old_t[:, None], num_particles, axis=1)
        total = np.zeros((B, num_particles), dtype=np.float64)
        first_profit = np.zeros((B, num_particles), dtype=np.float64)
        future_profit = np.zeros((B, num_particles), dtype=np.float64)
        oracle_price_sum = np.zeros((B, num_particles), dtype=np.float64)
        victim_price_sum = np.zeros((B, num_particles), dtype=np.float64)

        for ell in range(horizon):
            q_current = Q_clone[rows_b, rows_p, state_clone, :]
            eps = np.exp(-float(beta) * t_clone.astype(np.float64))
            pi_v = victim_policy_probs_from_q(q_current, K, victim_policy_mode, epsilon=eps)

            if ell == 0 or oracle_rollout_policy == "fixed_first_action":
                action_o = np.full((B, num_particles), a0, dtype=np.int64)
            elif oracle_rollout_policy == "greedy_best_response":
                expected_profit = np.einsum("...v,ov->...o", pi_v, profit_o_matrix)
                action_o = np.argmax(expected_profit, axis=-1).astype(np.int64)
            else:
                raise ValueError(f"unknown rollout_lola_oracle_rollout_policy: {oracle_rollout_policy}")

            cdf = np.cumsum(pi_v, axis=-1)
            u = rng.random((B, num_particles))
            action_v = np.sum(u[..., None] > cdf, axis=-1).astype(np.int64)
            action_v = np.minimum(action_v, K - 1)

            profit_o = profit_o_matrix[action_o, action_v]
            profit_v = profit_v_matrix[action_o, action_v]
            if ell == 0:
                first_profit = profit_o
            else:
                future_profit += (float(discount) ** ell) * profit_o
            if include_immediate or ell > 0:
                total += (float(discount) ** ell) * profit_o
            if prices is not None:
                oracle_price_sum += prices[action_o]
                victim_price_sum += prices[action_v]

            next_state = (action_o * K + action_v).astype(np.int64)
            old_q = Q_clone[rows_b, rows_p, state_clone, action_v]
            next_max = np.max(Q_clone[rows_b, rows_p, next_state, :], axis=-1)
            target = profit_v + float(delta) * next_max
            Q_clone[rows_b, rows_p, state_clone, action_v] = (1.0 - float(alpha)) * old_q + float(alpha) * target
            state_clone = next_state
            t_clone += 1

        values[:, a0] = np.mean(total, axis=1)
        first_profit_acc[:, a0] = np.mean(first_profit, axis=1)
        future_profit_acc[:, a0] = np.mean(future_profit, axis=1)
        if prices is not None:
            oracle_price_acc[:, a0] = np.mean(oracle_price_sum / float(horizon), axis=1)
            victim_price_acc[:, a0] = np.mean(victim_price_sum / float(horizon), axis=1)

    metrics = {
        "rollout_lola_first_step_profit": float(np.mean(first_profit_acc)),
        "rollout_lola_future_profit": float(np.mean(future_profit_acc)),
        "rollout_lola_victim_price_simulated": float(np.mean(victim_price_acc)) if prices is not None else float("nan"),
        "rollout_lola_oracle_price_simulated": float(np.mean(oracle_price_acc)) if prices is not None else float("nan"),
    }
    return values, metrics


def tabular_rollout_lola_select_actions(
    rollout_values: np.ndarray,
    tau: float,
    epsilon: float,
    generator: torch.Generator,
    device: torch.device,
    price_grid: np.ndarray | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    values_t = torch.as_tensor(rollout_values, dtype=torch.float32, device=device)
    probs = torch.softmax(values_t / max(float(tau), 1e-6), dim=-1)
    uniform = torch.full_like(probs, 1.0 / probs.shape[-1])
    probs = (1.0 - float(epsilon)) * probs + float(epsilon) * uniform
    actions = torch.multinomial(probs, 1, generator=generator).squeeze(1).to(torch.long)
    entropy = -(probs * torch.clamp(probs, min=1e-12).log()).sum(dim=1)
    best_actions = torch.argmax(values_t, dim=1).detach().cpu().numpy().astype(np.int64)
    if price_grid is None:
        best_action_price = float("nan")
    else:
        best_action_price = float(np.mean(np.asarray(price_grid, dtype=np.float64)[best_actions]))
    metrics = {
        "rollout_lola_value_mean": float(values_t.mean().detach().item()),
        "rollout_lola_value_std": float(values_t.std(unbiased=False).detach().item()),
        "rollout_lola_entropy": float(entropy.mean().detach().item()),
        "rollout_lola_best_action_price": best_action_price,
    }
    return actions, metrics


def tabular_cfr_metrics(cfr_state: dict[str, torch.Tensor], K: int) -> dict[str, float]:
    regret_table = cfr_state["regret_table"]
    state_id = cfr_state["state_id"]
    rows = torch.arange(regret_table.shape[0], device=regret_table.device)
    regrets = regret_table[rows, state_id, :]
    probs = tabular_cfr_strategy_probs(cfr_state, K, epsilon=0.0)
    entropy = -(probs * torch.clamp(probs, min=1e-12).log()).sum(dim=1)
    return {
        "avg_positive_regret": float(torch.clamp(regrets, min=0.0).mean().detach().item()),
        "avg_regret_abs": float(torch.abs(regrets).mean().detach().item()),
        "avg_strategy_entropy": float(entropy.mean().detach().item()),
        "avg_value": float(cfr_state["value_table"][rows, state_id].mean().detach().item()),
    }


def _market_features(env, obs_config: ObservationConfig, t: int, device: torch.device) -> torch.Tensor:
    return build_observation(
        cm.get_price_history_view(env),
        cm.get_current_prices(env),
        cm.get_rewards(env),
        cm.get_market_share(env),
        cm.get_outside_share(env),
        cm.get_margins(env),
        obs_config,
        time_step=t,
    ).to(device)


def init_dqn_params(
    generator: torch.Generator,
    obs_dim: int,
    hidden_dim: int,
    K: int,
    device: torch.device | str = "cpu",
) -> dict[str, torch.Tensor]:
    device = torch.device(device)
    W1 = torch.randn(obs_dim, hidden_dim, generator=generator, device=device) * (0.05 / np.sqrt(max(obs_dim, 1)))
    b1 = torch.zeros(hidden_dim, device=device)
    W2 = torch.randn(hidden_dim, K, generator=generator, device=device) * (0.05 / np.sqrt(max(hidden_dim, 1)))
    b2 = torch.zeros(K, device=device)
    return {"W1": W1.requires_grad_(True), "b1": b1.requires_grad_(True), "W2": W2.requires_grad_(True), "b2": b2.requires_grad_(True)}


def dqn_forward(params: dict[str, torch.Tensor], obs: torch.Tensor) -> torch.Tensor:
    x = torch.tanh(obs @ params["W1"] + params["b1"])
    return x @ params["W2"] + params["b2"]


def init_jepa_params(
    generator: torch.Generator,
    obs_dim: int,
    hidden_dim: int,
    latent_dim: int,
    K: int,
    device: torch.device | str = "cpu",
) -> dict[str, torch.Tensor]:
    device = torch.device(device)
    enc_W1 = torch.randn(obs_dim, hidden_dim, generator=generator, device=device) * (0.05 / np.sqrt(max(obs_dim, 1)))
    enc_b1 = torch.zeros(hidden_dim, device=device)
    enc_W2 = torch.randn(hidden_dim, latent_dim, generator=generator, device=device) * (0.05 / np.sqrt(max(hidden_dim, 1)))
    enc_b2 = torch.zeros(latent_dim, device=device)
    pred_in_dim = latent_dim + K
    pred_W1 = torch.randn(pred_in_dim, hidden_dim, generator=generator, device=device) * (0.05 / np.sqrt(max(pred_in_dim, 1)))
    pred_b1 = torch.zeros(hidden_dim, device=device)
    pred_W2 = torch.randn(hidden_dim, latent_dim, generator=generator, device=device) * (0.05 / np.sqrt(max(hidden_dim, 1)))
    pred_b2 = torch.zeros(latent_dim, device=device)
    return {
        "enc_W1": enc_W1.requires_grad_(True),
        "enc_b1": enc_b1.requires_grad_(True),
        "enc_W2": enc_W2.requires_grad_(True),
        "enc_b2": enc_b2.requires_grad_(True),
        "pred_W1": pred_W1.requires_grad_(True),
        "pred_b1": pred_b1.requires_grad_(True),
        "pred_W2": pred_W2.requires_grad_(True),
        "pred_b2": pred_b2.requires_grad_(True),
    }


def jepa_encode(params: dict[str, torch.Tensor], obs: torch.Tensor) -> torch.Tensor:
    x = torch.tanh(obs @ params["enc_W1"] + params["enc_b1"])
    return x @ params["enc_W2"] + params["enc_b2"]


def jepa_predict(params: dict[str, torch.Tensor], latent: torch.Tensor, action: torch.Tensor, K: int) -> torch.Tensor:
    action_onehot = torch.nn.functional.one_hot(action.to(torch.int64), num_classes=K).to(dtype=latent.dtype, device=latent.device)
    x = torch.cat([latent, action_onehot], dim=1)
    h = torch.tanh(x @ params["pred_W1"] + params["pred_b1"])
    return h @ params["pred_W2"] + params["pred_b2"]


def init_regret_params(
    generator: torch.Generator,
    obs_dim: int,
    hidden_dim: int,
    K: int,
    device: torch.device | str = "cpu",
) -> dict[str, torch.Tensor]:
    device = torch.device(device)
    W1 = torch.randn(obs_dim, hidden_dim, generator=generator, device=device) * (0.05 / np.sqrt(max(obs_dim, 1)))
    b1 = torch.zeros(hidden_dim, device=device)
    W2 = torch.randn(hidden_dim, K, generator=generator, device=device) * (0.05 / np.sqrt(max(hidden_dim, 1)))
    b2 = torch.zeros(K, device=device)
    return {"W1": W1.requires_grad_(True), "b1": b1.requires_grad_(True), "W2": W2.requires_grad_(True), "b2": b2.requires_grad_(True)}


def regret_forward(params: dict[str, torch.Tensor], obs: torch.Tensor) -> torch.Tensor:
    x = torch.tanh(obs @ params["W1"] + params["b1"])
    return x @ params["W2"] + params["b2"]


def clone_params(params: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {k: v.detach().clone() for k, v in params.items()}


def sync_params(target: dict[str, torch.Tensor], source: dict[str, torch.Tensor]) -> None:
    with torch.no_grad():
        for k, v in source.items():
            target[k].copy_(v.detach())


def init_replay_buffer(capacity: int, obs_dim: int, K: int = 0, device: torch.device | str = "cpu") -> dict[str, Any]:
    device = torch.device(device)
    return {
        "obs": torch.empty((capacity, obs_dim), dtype=torch.float32, device=device),
        "action": torch.empty(capacity, dtype=torch.int64, device=device),
        "victim_action": torch.empty(capacity, dtype=torch.int64, device=device),
        "reward": torch.empty(capacity, dtype=torch.float32, device=device),
        "next_obs": torch.empty((capacity, obs_dim), dtype=torch.float32, device=device),
        "done": torch.empty(capacity, dtype=torch.float32, device=device),
        "cf_profit": torch.empty((capacity, K), dtype=torch.float32, device=device),
        "pos": 0,
        "size": 0,
        "capacity": capacity,
    }


def replay_add(
    buffer: dict[str, Any],
    obs: torch.Tensor,
    action: torch.Tensor,
    reward: torch.Tensor,
    next_obs: torch.Tensor,
    done: torch.Tensor,
    victim_action: torch.Tensor | None = None,
    cf_profit: torch.Tensor | None = None,
) -> None:
    n = obs.shape[0]
    cap = buffer["capacity"]
    idx = (torch.arange(n, device=obs.device) + int(buffer["pos"])) % cap
    buffer["obs"][idx] = obs.detach()
    buffer["action"][idx] = action.detach()
    if victim_action is None:
        buffer["victim_action"][idx] = torch.zeros(n, dtype=torch.int64, device=obs.device)
    else:
        buffer["victim_action"][idx] = victim_action.detach()
    buffer["reward"][idx] = reward.detach()
    buffer["next_obs"][idx] = next_obs.detach()
    buffer["done"][idx] = done.detach().to(dtype=buffer["done"].dtype)
    if buffer["cf_profit"].shape[1] > 0:
        if cf_profit is None:
            buffer["cf_profit"][idx] = torch.zeros_like(buffer["cf_profit"][idx])
        else:
            buffer["cf_profit"][idx] = cf_profit.detach()
    buffer["pos"] = int((buffer["pos"] + n) % cap)
    buffer["size"] = int(min(cap, buffer["size"] + n))


def replay_sample(buffer: dict[str, Any], batch_size: int, generator: torch.Generator) -> dict[str, torch.Tensor]:
    idx = torch.randint(0, int(buffer["size"]), (batch_size,), generator=generator, device=buffer["obs"].device)
    return {k: buffer[k][idx] for k in ["obs", "action", "victim_action", "reward", "next_obs", "done", "cf_profit"]}


def dqn_train_step(
    params: dict[str, torch.Tensor],
    target_params: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    optimizer: torch.optim.Optimizer,
    gamma: float,
) -> float:
    q = dqn_forward(params, batch["obs"])
    q_sa = q.gather(1, batch["action"].view(-1, 1)).squeeze(1)
    with torch.no_grad():
        q_next = dqn_forward(target_params, batch["next_obs"]).max(dim=1).values
        done = batch["done"].to(dtype=batch["reward"].dtype)
        target = batch["reward"] + gamma * (1.0 - done) * q_next
    loss = torch.mean((q_sa - target) ** 2)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return float(loss.detach().item())


def dqn_jepa_train_step(
    params: dict[str, torch.Tensor],
    target_params: dict[str, torch.Tensor],
    jepa_params: dict[str, torch.Tensor],
    target_jepa_params: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    optimizer: torch.optim.Optimizer,
    gamma: float,
    jepa_coef: float,
    K: int,
) -> dict[str, float]:
    q = dqn_forward(params, batch["obs"])
    q_sa = q.gather(1, batch["action"].view(-1, 1)).squeeze(1)
    with torch.no_grad():
        q_next = dqn_forward(target_params, batch["next_obs"]).max(dim=1).values
        done = batch["done"].to(dtype=batch["reward"].dtype)
        target = batch["reward"] + gamma * (1.0 - done) * q_next
        target_latent = jepa_encode(target_jepa_params, batch["next_obs"])
    dqn_loss = torch.mean((q_sa - target) ** 2)
    latent = jepa_encode(jepa_params, batch["obs"])
    pred_latent = jepa_predict(jepa_params, latent, batch["action"], K)
    jepa_loss = torch.mean((pred_latent - target_latent) ** 2)
    total_loss = dqn_loss + float(jepa_coef) * jepa_loss
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()
    return {
        "dqn_loss": float(dqn_loss.detach().item()),
        "jepa_loss": float(jepa_loss.detach().item()),
        "total_loss": float(total_loss.detach().item()),
        "q_mean": float(q.detach().mean().item()),
        "q_max": float(q.detach().max().item()),
    }


def dqn_regret_train_step(
    params: dict[str, torch.Tensor],
    target_params: dict[str, torch.Tensor],
    regret_params: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    optimizer: torch.optim.Optimizer,
    gamma: float,
    regret_coef: float,
) -> dict[str, float]:
    q = dqn_forward(params, batch["obs"])
    q_sa = q.gather(1, batch["action"].view(-1, 1)).squeeze(1)
    with torch.no_grad():
        q_next = dqn_forward(target_params, batch["next_obs"]).max(dim=1).values
        done = batch["done"].to(dtype=batch["reward"].dtype)
        target = batch["reward"] + gamma * (1.0 - done) * q_next
        cf_profit = batch["cf_profit"].to(dtype=q.dtype)
    dqn_loss = torch.mean((q_sa - target) ** 2)
    shared_features = torch.tanh(batch["obs"] @ params["W1"] + params["b1"])
    regret_pred = shared_features @ regret_params["W2"] + regret_params["b2"]
    regret_loss = torch.mean((regret_pred - cf_profit) ** 2)
    total_loss = dqn_loss + float(regret_coef) * regret_loss
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()
    return {
        "dqn_loss": float(dqn_loss.detach().item()),
        "regret_loss": float(regret_loss.detach().item()),
        "total_loss": float(total_loss.detach().item()),
        "q_mean": float(q.detach().mean().item()),
        "q_max": float(q.detach().max().item()),
    }


def _warm_history(env, victim: dict[str, Any], H: int, K: int, rng: np.random.Generator) -> None:
    B = victim["Q"].shape[0]
    actions = rng.integers(0, K, size=(B, 2), dtype=np.int64)
    for _ in range(H):
        cm.step(env, actions)
    victim["state_id"] = (actions[:, 0] * K + actions[:, 1]).astype(np.int64)


def _oracle_reward(rewards: np.ndarray, asymmetry_coef: float) -> np.ndarray:
    return rewards[:, 0] + asymmetry_coef * (rewards[:, 0] - rewards[:, 1])


def _nanmean_or_default(values: list[float], default: float = 0.0) -> float:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0 or np.all(np.isnan(arr)):
        return float(default)
    return float(np.nanmean(arr))


def evaluate(config: QVictimOracleConfig, params, buffers, benchmarks: StaticBenchmarks, policy_value=None) -> dict[str, float]:
    device = torch.device(config.device)
    rng = np.random.default_rng(config.seed + 10_000)
    env, price_grid, _, profit_matrix = make_calvano_vec_env(config.B, config.H, config.K, config.seed + 10_000)
    victim = init_victim_state(config.B, config.K, benchmarks, config.victim_delta, rng)
    _warm_history(env, victim, config.H, config.K, rng)
    obs_cfg = None
    h = None
    tabular_cfr_kinds = {"tabular_cfr", "tabular_multi_cfr"}
    tabular_direct_kinds = tabular_cfr_kinds | {"tabular_lola", "tabular_model_lola", "tabular_rollout_lola"}
    eval_gen = None
    if config.oracle_kind not in tabular_direct_kinds:
        obs_cfg = ObservationConfig(price_min=float(price_grid[0]), price_max=float(price_grid[-1]), device=device)
        h = torch.zeros(config.B, config.reservoir_dim, dtype=torch.float32, device=device)
    else:
        eval_gen = torch.Generator(device=device).manual_seed(config.seed + 20_000)
        eval_rollout_rng = np.random.default_rng(config.seed + 30_000)
    if config.oracle_kind in tabular_cfr_kinds:
        eval_cfr_state = {
            "regret_table": params["regret_table"],
            "value_table": params["value_table"],
            "state_id": torch.zeros(config.B, dtype=torch.long, device=device),
        }
    reward_sum = np.zeros(2, dtype=np.float64)
    price_sum = np.zeros(2, dtype=np.float64)
    victim_eps_sum = 0.0
    with torch.no_grad():
        for t in range(config.eval_steps):
            if config.oracle_kind in tabular_cfr_kinds:
                action_o_t = tabular_cfr_select_actions(eval_cfr_state, config.K, epsilon=0.0, generator=eval_gen)
                action_o = action_o_t.cpu().numpy().astype(np.int64)
            elif config.oracle_kind == "tabular_lola":
                action_o_t, _ = tabular_lola_select_actions(
                    victim,
                    profit_matrix,
                    config.K,
                    config.lola_gamma,
                    config.lola_tau,
                    epsilon=0.0,
                    generator=eval_gen,
                    device=device,
                    victim_alpha=config.victim_alpha,
                    victim_delta=config.victim_delta,
                )
                action_o = action_o_t.cpu().numpy().astype(np.int64)
            elif config.oracle_kind == "tabular_model_lola":
                lola_values, _ = tabular_model_lola_values(
                    victim,
                    profit_matrix,
                    config.K,
                    config.victim_alpha,
                    config.victim_delta,
                    config.victim_beta,
                    config.model_lola_gamma,
                    config.model_lola_victim_policy,
                    config.model_lola_future_policy,
                    config.model_lola_victim_softmax_tau,
                )
                action_o_t, _ = tabular_model_lola_select_actions(
                    lola_values,
                    config.model_lola_tau,
                    epsilon=0.0,
                    generator=eval_gen,
                    device=device,
                )
                action_o = action_o_t.cpu().numpy().astype(np.int64)
            elif config.oracle_kind == "tabular_rollout_lola":
                rollout_values, _ = tabular_rollout_lola_values(
                    victim,
                    profit_matrix,
                    config.K,
                    config.victim_alpha,
                    config.victim_delta,
                    config.victim_beta,
                    config.rollout_lola_horizon,
                    config.rollout_lola_num_particles,
                    config.rollout_lola_victim_policy,
                    config.rollout_lola_oracle_rollout_policy,
                    config.rollout_lola_discount,
                    config.rollout_lola_include_immediate,
                    eval_rollout_rng,
                    price_grid,
                )
                action_o_t, _ = tabular_rollout_lola_select_actions(
                    rollout_values,
                    config.rollout_lola_tau,
                    epsilon=0.0,
                    generator=eval_gen,
                    device=device,
                    price_grid=price_grid,
                )
                action_o = action_o_t.cpu().numpy().astype(np.int64)
            else:
                assert obs_cfg is not None and h is not None
                features = _market_features(env, obs_cfg, t, device)
                h = reservoir_update(features, h, buffers["reservoir"])
                obs = reservoir_observation(features, h)
                if config.oracle_kind in {"dqn", "dqn_jepa", "dqn_regret"}:
                    action_o = torch.argmax(dqn_forward(params, obs), dim=1).cpu().numpy().astype(np.int64)
                else:
                    logits, _ = mlp_policy_forward(params, {}, obs, None)
                    action_o = torch.argmax(logits, dim=1).cpu().numpy().astype(np.int64)
            eps_v = victim_epsilon(config.victim_beta, victim["t"])
            action_v = victim_select_actions(victim, config.K, beta=config.victim_beta, rng=rng, greedy=False)
            actions = np.stack([action_o, action_v], axis=1).astype(np.int64)
            cm.step(env, actions)
            rewards = np.asarray(cm.get_rewards(env), dtype=np.float64)
            prices = np.asarray(cm.get_current_prices(env), dtype=np.float64)
            update_victim_q(victim, action_o, action_v, rewards[:, 1], config.victim_alpha, config.victim_delta, config.K)
            if config.oracle_kind in tabular_cfr_kinds:
                eval_cfr_state["state_id"] = tabular_cfr_state_id(action_o, action_v, config.K, config.cfr_state_mode, device)
            reward_sum += rewards.sum(axis=0)
            price_sum += prices.sum(axis=0)
            victim_eps_sum += float(np.mean(eps_v))
    denom_count = max(config.eval_steps * config.B, 1)
    avg_profit = reward_sum / denom_count
    avg_price = price_sum / denom_count
    denom = np.maximum(np.abs(benchmarks.pi_m - benchmarks.pi_n), 1e-12)
    gains = (avg_profit - benchmarks.pi_n) / denom
    market_price = float(np.mean(avg_price))
    return {
        "eval_avg_profit_oracle": float(avg_profit[0]),
        "eval_avg_profit_victim": float(avg_profit[1]),
        "eval_profit_asymmetry": float(avg_profit[0] - avg_profit[1]),
        "eval_avg_price_oracle": float(avg_price[0]),
        "eval_avg_price_victim": float(avg_price[1]),
        "eval_market_price_mean": market_price,
        "eval_oracle_profit_gain": float(gains[0]),
        "eval_victim_profit_gain": float(gains[1]),
        "eval_distance_to_nash_price": float(abs(market_price - benchmarks.p_n)),
        "eval_distance_to_monopoly_price": float(abs(market_price - benchmarks.p_m)),
        "victim_avg_epsilon": victim_eps_sum / max(config.eval_steps, 1),
    }


def run_experiment(config: QVictimOracleConfig) -> dict[str, Any]:
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    device = torch.device(config.device)
    rng = np.random.default_rng(config.seed)
    torch_gen = torch.Generator(device=device).manual_seed(config.seed)
    tabular_cfr_kinds = {"tabular_cfr", "tabular_multi_cfr"}
    tabular_direct_kinds = tabular_cfr_kinds | {"tabular_lola", "tabular_model_lola", "tabular_rollout_lola"}
    rollout_rng = np.random.default_rng(config.seed + 40_000)
    env, price_grid, benchmarks, profit_matrix = make_calvano_vec_env(config.B, config.H, config.K, config.seed)
    victim = init_victim_state(config.B, config.K, benchmarks, config.victim_delta, rng)
    _warm_history(env, victim, config.H, config.K, rng)
    base_dim = observation_dim(config.H)
    obs_dim = base_dim + config.reservoir_dim
    if config.oracle_kind in tabular_direct_kinds:
        reservoir = None
        buffers = {}
        obs_cfg = None
        h = None
    else:
        reservoir = init_reservoir_buffers(
            torch_gen,
            ReservoirConfig(base_dim, config.reservoir_dim, config.reservoir_spectral_radius, leak_rate=config.reservoir_leak_rate, device=device),
        )
        buffers = {"reservoir": reservoir}
        obs_cfg = ObservationConfig(price_min=float(price_grid[0]), price_max=float(price_grid[-1]), device=device)
        h = torch.zeros(config.B, config.reservoir_dim, dtype=torch.float32, device=device)

    if config.oracle_kind in {"dqn", "dqn_jepa", "dqn_regret"}:
        params = init_dqn_params(torch_gen, obs_dim, config.hidden_dim, config.K, device)
        target_params = clone_params(params)
        if config.oracle_kind == "dqn_jepa":
            jepa_params = init_jepa_params(torch_gen, obs_dim, config.hidden_dim, config.jepa_latent_dim, config.K, device)
            target_jepa_params = clone_params(jepa_params)
            regret_params = None
            optimizer = torch.optim.Adam(list(params.values()) + list(jepa_params.values()), lr=config.lr)
        elif config.oracle_kind == "dqn_regret":
            jepa_params = None
            target_jepa_params = None
            regret_params = init_regret_params(torch_gen, obs_dim, config.hidden_dim, config.K, device)
            optimizer = torch.optim.Adam(list(params.values()) + list(regret_params.values()), lr=config.lr)
        else:
            jepa_params = None
            target_jepa_params = None
            regret_params = None
            optimizer = torch.optim.Adam(list(params.values()), lr=config.lr)
        replay = init_replay_buffer(config.replay_capacity, obs_dim, config.K, device)
    elif config.oracle_kind == "actor_critic":
        params = init_mlp_policy(torch_gen, obs_dim, config.hidden_dim, config.K, device=device)
        value_params = init_mlp_value(torch_gen, obs_dim, config.value_hidden_dim, device=device)
        optimizer = torch.optim.Adam(list(params.values()), lr=config.lr)
        value_optimizer = torch.optim.Adam(list(value_params.values()), lr=config.lr_value)
    elif config.oracle_kind in tabular_cfr_kinds:
        params = init_tabular_cfr_state(config.B, config.K, config.cfr_state_mode, device)
    elif config.oracle_kind in {"tabular_lola", "tabular_model_lola", "tabular_rollout_lola"}:
        params = {}
    else:
        raise ValueError(f"unknown oracle_kind: {config.oracle_kind}")

    train_rows = []
    eval_rows = []
    pending = {
        "reward": [],
        "price": [],
        "asym": [],
        "loss": [],
        "dqn_loss": [],
        "jepa_loss": [],
        "regret_loss": [],
        "total_loss": [],
        "q_mean": [],
        "q_max": [],
        "entropy": [],
        "avg_positive_regret": [],
        "avg_regret_abs": [],
        "avg_strategy_entropy": [],
        "avg_value": [],
        "lola_immediate_value": [],
        "lola_future_value": [],
        "lola_total_value": [],
        "lola_entropy": [],
        "model_lola_value": [],
        "model_lola_value_std": [],
        "model_lola_entropy": [],
        "model_lola_immediate_value": [],
        "model_lola_future_value": [],
        "model_lola_total_value": [],
        "model_lola_current_victim_entropy": [],
        "model_lola_future_victim_entropy": [],
        "rollout_lola_value_mean": [],
        "rollout_lola_value_std": [],
        "rollout_lola_entropy": [],
        "rollout_lola_best_action_price": [],
        "rollout_lola_first_step_profit": [],
        "rollout_lola_future_profit": [],
        "rollout_lola_victim_price_simulated": [],
        "rollout_lola_oracle_price_simulated": [],
        "victim_pred_accuracy": [],
        "eps_o": [],
        "eps_v": [],
    }
    rollout_logp: list[torch.Tensor] = []
    rollout_values: list[torch.Tensor] = []
    rollout_rewards: list[torch.Tensor] = []
    for step in range(1, config.total_steps + 1):
        eps_o = oracle_epsilon(config, step)
        prev_cfr_state_id = None
        avg_positive_regret_val = np.nan
        avg_regret_abs_val = np.nan
        avg_strategy_entropy_val = np.nan
        avg_value_val = np.nan
        lola_immediate_value_val = np.nan
        lola_future_value_val = np.nan
        lola_total_value_val = np.nan
        lola_entropy_val = np.nan
        model_lola_value_val = np.nan
        model_lola_value_std_val = np.nan
        model_lola_entropy_val = np.nan
        model_lola_immediate_value_val = np.nan
        model_lola_future_value_val = np.nan
        model_lola_total_value_val = np.nan
        model_lola_current_victim_entropy_val = np.nan
        model_lola_future_victim_entropy_val = np.nan
        rollout_lola_value_mean_val = np.nan
        rollout_lola_value_std_val = np.nan
        rollout_lola_entropy_val = np.nan
        rollout_lola_best_action_price_val = np.nan
        rollout_lola_first_step_profit_val = np.nan
        rollout_lola_future_profit_val = np.nan
        rollout_lola_victim_price_simulated_val = np.nan
        rollout_lola_oracle_price_simulated_val = np.nan
        victim_pred_accuracy_val = np.nan
        if config.oracle_kind in tabular_cfr_kinds:
            prev_cfr_state_id = params["state_id"].clone()
            action_o_t = tabular_cfr_select_actions(params, config.K, eps_o, torch_gen)
            cfr_stats = tabular_cfr_metrics(params, config.K)
            avg_positive_regret_val = cfr_stats["avg_positive_regret"]
            avg_regret_abs_val = cfr_stats["avg_regret_abs"]
            avg_strategy_entropy_val = cfr_stats["avg_strategy_entropy"]
            avg_value_val = cfr_stats["avg_value"]
            entropy_val = avg_strategy_entropy_val
            q_mean_val = float("nan")
            q_max_val = float("nan")
        elif config.oracle_kind == "tabular_lola":
            victim_actions_pred = victim_policy_from_q(victim, config.K, greedy=True)
            action_o_t, lola_metrics = tabular_lola_select_actions(
                victim,
                profit_matrix,
                config.K,
                config.lola_gamma,
                config.lola_tau,
                config.lola_epsilon,
                torch_gen,
                device,
                victim_alpha=config.victim_alpha,
                victim_delta=config.victim_delta,
            )
            lola_immediate_value_val = lola_metrics["lola_immediate_value"]
            lola_future_value_val = lola_metrics["lola_future_value"]
            lola_total_value_val = lola_metrics["lola_total_value"]
            lola_entropy_val = lola_metrics["lola_entropy"]
            entropy_val = lola_entropy_val
            q_mean_val = float("nan")
            q_max_val = float("nan")
        elif config.oracle_kind == "tabular_model_lola":
            q_current = victim["Q"][np.arange(config.B), victim["state_id"], :]
            pi_v_current = victim_policy_probs_from_q(
                q_current,
                config.K,
                config.model_lola_victim_policy,
                epsilon=victim_epsilon(config.victim_beta, victim["t"]),
                tau=config.model_lola_victim_softmax_tau,
            )
            victim_actions_pred = np.argmax(pi_v_current, axis=1).astype(np.int64)
            lola_values, model_value_metrics = tabular_model_lola_values(
                victim,
                profit_matrix,
                config.K,
                config.victim_alpha,
                config.victim_delta,
                config.victim_beta,
                config.model_lola_gamma,
                config.model_lola_victim_policy,
                config.model_lola_future_policy,
                config.model_lola_victim_softmax_tau,
            )
            action_o_t, model_select_metrics = tabular_model_lola_select_actions(
                lola_values,
                config.model_lola_tau,
                config.model_lola_epsilon,
                torch_gen,
                device,
            )
            model_lola_immediate_value_val = model_value_metrics["model_lola_immediate_value"]
            model_lola_future_value_val = model_value_metrics["model_lola_future_value"]
            model_lola_total_value_val = model_value_metrics["model_lola_total_value"]
            model_lola_current_victim_entropy_val = model_value_metrics["model_lola_current_victim_entropy"]
            model_lola_future_victim_entropy_val = model_value_metrics["model_lola_future_victim_entropy"]
            model_lola_entropy_val = model_select_metrics["model_lola_entropy"]
            model_lola_value_val = model_select_metrics["model_lola_value_mean"]
            model_lola_value_std_val = model_select_metrics["model_lola_value_std"]
            entropy_val = model_lola_entropy_val
            q_mean_val = float("nan")
            q_max_val = float("nan")
        elif config.oracle_kind == "tabular_rollout_lola":
            q_current = victim["Q"][np.arange(config.B), victim["state_id"], :]
            pi_v_current = victim_policy_probs_from_q(
                q_current,
                config.K,
                config.rollout_lola_victim_policy,
                epsilon=victim_epsilon(config.victim_beta, victim["t"]),
            )
            victim_actions_pred = np.argmax(pi_v_current, axis=1).astype(np.int64)
            rollout_values, rollout_value_metrics = tabular_rollout_lola_values(
                victim,
                profit_matrix,
                config.K,
                config.victim_alpha,
                config.victim_delta,
                config.victim_beta,
                config.rollout_lola_horizon,
                config.rollout_lola_num_particles,
                config.rollout_lola_victim_policy,
                config.rollout_lola_oracle_rollout_policy,
                config.rollout_lola_discount,
                config.rollout_lola_include_immediate,
                rollout_rng,
                price_grid,
            )
            action_o_t, rollout_select_metrics = tabular_rollout_lola_select_actions(
                rollout_values,
                config.rollout_lola_tau,
                config.rollout_lola_epsilon,
                torch_gen,
                device,
                price_grid,
            )
            rollout_lola_value_mean_val = rollout_select_metrics["rollout_lola_value_mean"]
            rollout_lola_value_std_val = rollout_select_metrics["rollout_lola_value_std"]
            rollout_lola_entropy_val = rollout_select_metrics["rollout_lola_entropy"]
            rollout_lola_best_action_price_val = rollout_select_metrics["rollout_lola_best_action_price"]
            rollout_lola_first_step_profit_val = rollout_value_metrics["rollout_lola_first_step_profit"]
            rollout_lola_future_profit_val = rollout_value_metrics["rollout_lola_future_profit"]
            rollout_lola_victim_price_simulated_val = rollout_value_metrics["rollout_lola_victim_price_simulated"]
            rollout_lola_oracle_price_simulated_val = rollout_value_metrics["rollout_lola_oracle_price_simulated"]
            entropy_val = rollout_lola_entropy_val
            q_mean_val = float("nan")
            q_max_val = float("nan")
        else:
            assert obs_cfg is not None and h is not None and reservoir is not None
            features = _market_features(env, obs_cfg, step, device)
            h = reservoir_update(features, h, reservoir)
            obs = reservoir_observation(features, h).detach()
        if config.oracle_kind in {"dqn", "dqn_jepa", "dqn_regret"}:
            q_values = dqn_forward(params, obs)
            greedy = torch.argmax(q_values, dim=1)
            random = torch.randint(0, config.K, (config.B,), generator=torch_gen, device=device)
            explore = torch.rand(config.B, generator=torch_gen, device=device) < eps_o
            action_o_t = torch.where(explore, random, greedy).to(torch.int64)
            entropy_val = float("nan")
            q_mean_val = float(q_values.detach().mean().item())
            q_max_val = float(q_values.detach().max().item())
        elif config.oracle_kind == "actor_critic":
            logits, _ = mlp_policy_forward(params, {}, obs, None)
            dist = torch.distributions.Categorical(logits=logits)
            action_o_t = dist.sample(generator=torch_gen) if False else torch.multinomial(torch.softmax(logits, -1), 1, generator=torch_gen).squeeze(1)
            rollout_logp.append(dist.log_prob(action_o_t))
            rollout_values.append(mlp_value_forward(value_params, obs))
            entropy_val = float(dist.entropy().detach().mean().item())
            q_mean_val = float("nan")
            q_max_val = float("nan")

        action_o = action_o_t.detach().cpu().numpy().astype(np.int64)
        eps_v_arr = victim_epsilon(config.victim_beta, victim["t"])
        action_v = victim_select_actions(victim, config.K, beta=config.victim_beta, rng=rng, greedy=False)
        if config.oracle_kind in {"tabular_lola", "tabular_model_lola", "tabular_rollout_lola"}:
            victim_pred_accuracy_val = float(np.mean(victim_actions_pred == action_v))
        actions = np.stack([action_o, action_v], axis=1).astype(np.int64)
        cm.step(env, actions)
        rewards = np.asarray(cm.get_rewards(env), dtype=np.float64)
        prices = np.asarray(cm.get_current_prices(env), dtype=np.float64)
        reward_o = _oracle_reward(rewards, config.asymmetry_coef)
        update_victim_q(victim, action_o, action_v, rewards[:, 1], config.victim_alpha, config.victim_delta, config.K)

        if config.oracle_kind in tabular_direct_kinds:
            h_next = None
            next_obs = None
        else:
            assert obs_cfg is not None and h is not None and reservoir is not None
            next_features = _market_features(env, obs_cfg, step, device)
            h_next = reservoir_update(next_features, h, reservoir)
            next_obs = reservoir_observation(next_features, h_next).detach()

        loss_val = np.nan
        dqn_loss_val = np.nan
        jepa_loss_val = np.nan
        regret_loss_val = np.nan
        total_loss_val = np.nan
        if config.oracle_kind in tabular_cfr_kinds:
            assert prev_cfr_state_id is not None
            cf_profit_o_t = oracle_counterfactual_profit(profit_matrix, action_v, device)
            next_state_real = tabular_cfr_state_id(action_o, action_v, config.K, config.cfr_state_mode, device)
            if config.oracle_kind == "tabular_multi_cfr":
                next_state_cf = tabular_cfr_counterfactual_next_state_ids(
                    torch.arange(config.K, dtype=torch.long, device=device),
                    action_v,
                    config.K,
                    config.cfr_state_mode,
                    device,
                )
                cf_update_values = tabular_multi_cfr_cf_value(params, cf_profit_o_t, next_state_cf, config.cfr_gamma)
                tabular_cfr_update(params, prev_cfr_state_id, action_o_t, cf_update_values, config.cfr_regret_decay)
                tabular_multi_cfr_value_update(
                    params,
                    prev_cfr_state_id,
                    torch.as_tensor(rewards[:, 0], dtype=torch.float32, device=device),
                    next_state_real,
                    config.cfr_value_lr,
                    config.cfr_gamma,
                )
            else:
                tabular_cfr_update(params, prev_cfr_state_id, action_o_t, cf_profit_o_t, config.cfr_regret_decay)
            params["state_id"] = next_state_real
            loss_val = 0.0
            total_loss_val = 0.0
            cfr_stats = tabular_cfr_metrics(params, config.K)
            avg_positive_regret_val = cfr_stats["avg_positive_regret"]
            avg_regret_abs_val = cfr_stats["avg_regret_abs"]
            avg_strategy_entropy_val = cfr_stats["avg_strategy_entropy"]
            avg_value_val = cfr_stats["avg_value"]
            entropy_val = avg_strategy_entropy_val
        elif config.oracle_kind in {"dqn", "dqn_jepa", "dqn_regret"}:
            assert h is not None and reservoir is not None
            cf_profit_o = np.asarray(profit_matrix[:, action_v, 0], dtype=np.float32).T
            replay_add(
                replay,
                obs,
                action_o_t,
                torch.as_tensor(reward_o, dtype=torch.float32, device=device),
                next_obs,
                torch.zeros(config.B, dtype=torch.float32, device=device),
                torch.as_tensor(action_v, dtype=torch.int64, device=device),
                torch.as_tensor(cf_profit_o, dtype=torch.float32, device=device),
            )
            if step % config.train_every == 0 and replay["size"] >= config.batch_size:
                batch = replay_sample(replay, config.batch_size, torch_gen)
                if config.oracle_kind == "dqn_jepa":
                    assert jepa_params is not None and target_jepa_params is not None
                    train_metrics = dqn_jepa_train_step(
                        params,
                        target_params,
                        jepa_params,
                        target_jepa_params,
                        batch,
                        optimizer,
                        config.gamma,
                        config.jepa_coef,
                        config.K,
                    )
                    dqn_loss_val = train_metrics["dqn_loss"]
                    jepa_loss_val = train_metrics["jepa_loss"]
                    total_loss_val = train_metrics["total_loss"]
                    loss_val = total_loss_val
                    q_mean_val = train_metrics["q_mean"]
                    q_max_val = train_metrics["q_max"]
                elif config.oracle_kind == "dqn_regret":
                    assert regret_params is not None
                    train_metrics = dqn_regret_train_step(
                        params,
                        target_params,
                        regret_params,
                        batch,
                        optimizer,
                        config.gamma,
                        config.regret_coef,
                    )
                    dqn_loss_val = train_metrics["dqn_loss"]
                    regret_loss_val = train_metrics["regret_loss"]
                    total_loss_val = train_metrics["total_loss"]
                    loss_val = total_loss_val
                    q_mean_val = train_metrics["q_mean"]
                    q_max_val = train_metrics["q_max"]
                else:
                    dqn_loss_val = dqn_train_step(params, target_params, batch, optimizer, config.gamma)
                    loss_val = dqn_loss_val
                    total_loss_val = dqn_loss_val
                    with torch.no_grad():
                        q_after = dqn_forward(params, obs)
                        q_mean_val = float(q_after.mean().item())
                        q_max_val = float(q_after.max().item())
            if step % config.target_update_every == 0:
                sync_params(target_params, params)
                if config.oracle_kind == "dqn_jepa":
                    assert jepa_params is not None and target_jepa_params is not None
                    sync_params(target_jepa_params, jepa_params)
        elif config.oracle_kind in {"tabular_lola", "tabular_model_lola", "tabular_rollout_lola"}:
            loss_val = 0.0
            total_loss_val = 0.0
        else:
            rollout_rewards.append(torch.as_tensor(reward_o, dtype=torch.float32, device=device))
            if len(rollout_rewards) >= config.rollout_steps:
                rewards_t = torch.stack(rollout_rewards, dim=0).unsqueeze(-1).repeat(1, 1, 2)
                returns = discounted_returns(rewards_t, config.gamma)[:, :, 0]
                logp = torch.stack(rollout_logp, dim=0)
                values = torch.stack(rollout_values, dim=0)
                parts = actor_critic_loss(logp, values, returns, entropy=None, entropy_coef=config.entropy_coef, value_coef=config.value_coef)
                optimizer.zero_grad()
                value_optimizer.zero_grad()
                parts["loss"].backward()
                optimizer.step()
                value_optimizer.step()
                loss_val = float(parts["loss"].detach().item())
                total_loss_val = loss_val
                rollout_logp.clear()
                rollout_values.clear()
                rollout_rewards.clear()

        if config.oracle_kind not in tabular_direct_kinds:
            h = h_next.detach()
        pending["reward"].append(rewards.mean(axis=0))
        pending["price"].append(prices.mean(axis=0))
        pending["asym"].append(float(np.mean(rewards[:, 0] - rewards[:, 1])))
        pending["loss"].append(loss_val)
        pending["dqn_loss"].append(dqn_loss_val)
        pending["jepa_loss"].append(jepa_loss_val)
        pending["regret_loss"].append(regret_loss_val)
        pending["total_loss"].append(total_loss_val)
        pending["q_mean"].append(q_mean_val)
        pending["q_max"].append(q_max_val)
        pending["entropy"].append(entropy_val)
        pending["avg_positive_regret"].append(avg_positive_regret_val)
        pending["avg_regret_abs"].append(avg_regret_abs_val)
        pending["avg_strategy_entropy"].append(avg_strategy_entropy_val)
        pending["avg_value"].append(avg_value_val)
        pending["lola_immediate_value"].append(lola_immediate_value_val)
        pending["lola_future_value"].append(lola_future_value_val)
        pending["lola_total_value"].append(lola_total_value_val)
        pending["lola_entropy"].append(lola_entropy_val)
        pending["model_lola_value"].append(model_lola_value_val)
        pending["model_lola_value_std"].append(model_lola_value_std_val)
        pending["model_lola_entropy"].append(model_lola_entropy_val)
        pending["model_lola_immediate_value"].append(model_lola_immediate_value_val)
        pending["model_lola_future_value"].append(model_lola_future_value_val)
        pending["model_lola_total_value"].append(model_lola_total_value_val)
        pending["model_lola_current_victim_entropy"].append(model_lola_current_victim_entropy_val)
        pending["model_lola_future_victim_entropy"].append(model_lola_future_victim_entropy_val)
        pending["rollout_lola_value_mean"].append(rollout_lola_value_mean_val)
        pending["rollout_lola_value_std"].append(rollout_lola_value_std_val)
        pending["rollout_lola_entropy"].append(rollout_lola_entropy_val)
        pending["rollout_lola_best_action_price"].append(rollout_lola_best_action_price_val)
        pending["rollout_lola_first_step_profit"].append(rollout_lola_first_step_profit_val)
        pending["rollout_lola_future_profit"].append(rollout_lola_future_profit_val)
        pending["rollout_lola_victim_price_simulated"].append(rollout_lola_victim_price_simulated_val)
        pending["rollout_lola_oracle_price_simulated"].append(rollout_lola_oracle_price_simulated_val)
        pending["victim_pred_accuracy"].append(victim_pred_accuracy_val)
        pending["eps_o"].append(eps_o)
        pending["eps_v"].append(float(np.mean(eps_v_arr)))

        if step % config.log_every == 0 or step == config.total_steps:
            rewards_avg = np.mean(np.stack(pending["reward"]), axis=0)
            prices_avg = np.mean(np.stack(pending["price"]), axis=0)
            loss_mean = _nanmean_or_default(pending["loss"])
            dqn_loss_mean = _nanmean_or_default(pending["dqn_loss"], default=float("nan"))
            jepa_loss_mean = _nanmean_or_default(pending["jepa_loss"], default=float("nan"))
            regret_loss_mean = _nanmean_or_default(pending["regret_loss"], default=float("nan"))
            total_loss_mean = _nanmean_or_default(pending["total_loss"], default=float("nan"))
            train_rows.append(
                {
                    "step": step,
                    "oracle_epsilon": float(np.mean(pending["eps_o"])),
                    "victim_epsilon": float(np.mean(pending["eps_v"])),
                    "avg_profit_oracle": float(rewards_avg[0]),
                    "avg_profit_victim": float(rewards_avg[1]),
                    "profit_asymmetry": float(np.mean(pending["asym"])),
                    "avg_price_oracle": float(prices_avg[0]),
                    "avg_price_victim": float(prices_avg[1]),
                    "loss": loss_mean,
                    "dqn_loss": dqn_loss_mean,
                    "jepa_loss": jepa_loss_mean,
                    "regret_loss": regret_loss_mean,
                    "total_loss": total_loss_mean,
                    "q_mean": _nanmean_or_default(pending["q_mean"]),
                    "q_max": _nanmean_or_default(pending["q_max"]),
                    "entropy": _nanmean_or_default(pending["entropy"]),
                    "avg_positive_regret": _nanmean_or_default(pending["avg_positive_regret"], default=float("nan")),
                    "avg_regret_abs": _nanmean_or_default(pending["avg_regret_abs"], default=float("nan")),
                    "avg_strategy_entropy": _nanmean_or_default(pending["avg_strategy_entropy"], default=float("nan")),
                    "avg_value": _nanmean_or_default(pending["avg_value"], default=float("nan")),
                    "lola_immediate_value": _nanmean_or_default(pending["lola_immediate_value"], default=float("nan")),
                    "lola_future_value": _nanmean_or_default(pending["lola_future_value"], default=float("nan")),
                    "lola_total_value": _nanmean_or_default(pending["lola_total_value"], default=float("nan")),
                    "lola_entropy": _nanmean_or_default(pending["lola_entropy"], default=float("nan")),
                    "model_lola_value": _nanmean_or_default(pending["model_lola_value"], default=float("nan")),
                    "model_lola_value_std": _nanmean_or_default(pending["model_lola_value_std"], default=float("nan")),
                    "model_lola_entropy": _nanmean_or_default(pending["model_lola_entropy"], default=float("nan")),
                    "model_lola_immediate_value": _nanmean_or_default(pending["model_lola_immediate_value"], default=float("nan")),
                    "model_lola_future_value": _nanmean_or_default(pending["model_lola_future_value"], default=float("nan")),
                    "model_lola_total_value": _nanmean_or_default(pending["model_lola_total_value"], default=float("nan")),
                    "model_lola_current_victim_entropy": _nanmean_or_default(
                        pending["model_lola_current_victim_entropy"],
                        default=float("nan"),
                    ),
                    "model_lola_future_victim_entropy": _nanmean_or_default(
                        pending["model_lola_future_victim_entropy"],
                        default=float("nan"),
                    ),
                    "rollout_lola_value_mean": _nanmean_or_default(pending["rollout_lola_value_mean"], default=float("nan")),
                    "rollout_lola_value_std": _nanmean_or_default(pending["rollout_lola_value_std"], default=float("nan")),
                    "rollout_lola_entropy": _nanmean_or_default(pending["rollout_lola_entropy"], default=float("nan")),
                    "rollout_lola_best_action_price": _nanmean_or_default(pending["rollout_lola_best_action_price"], default=float("nan")),
                    "rollout_lola_first_step_profit": _nanmean_or_default(pending["rollout_lola_first_step_profit"], default=float("nan")),
                    "rollout_lola_future_profit": _nanmean_or_default(pending["rollout_lola_future_profit"], default=float("nan")),
                    "rollout_lola_victim_price_simulated": _nanmean_or_default(
                        pending["rollout_lola_victim_price_simulated"],
                        default=float("nan"),
                    ),
                    "rollout_lola_oracle_price_simulated": _nanmean_or_default(
                        pending["rollout_lola_oracle_price_simulated"],
                        default=float("nan"),
                    ),
                    "victim_pred_accuracy": _nanmean_or_default(pending["victim_pred_accuracy"], default=float("nan")),
                }
            )
            for v in pending.values():
                v.clear()

        if step % config.eval_every == 0 or step == config.total_steps:
            row = {"step": step}
            row.update(evaluate(config, params, buffers, benchmarks))
            eval_rows.append(row)

    train_df = pd.DataFrame.from_records(train_rows)
    eval_df = pd.DataFrame.from_records(eval_rows)
    summary = make_summary(config, eval_df, benchmarks)
    if config.out_dir is not None:
        out = Path(config.out_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "config.json").write_text(json.dumps(asdict(config), indent=2, sort_keys=True), encoding="utf-8")
        train_df.to_csv(out / "train_metrics.csv", index=False)
        eval_df.to_csv(out / "eval_metrics.csv", index=False)
        (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return {"train_metrics": train_df, "eval_metrics": eval_df, "summary": summary}


def make_summary(config: QVictimOracleConfig, eval_df: pd.DataFrame, benchmarks: StaticBenchmarks) -> dict[str, Any]:
    final = eval_df.iloc[-1] if len(eval_df) else {}
    asym = eval_df["eval_profit_asymmetry"] if "eval_profit_asymmetry" in eval_df else pd.Series(dtype=float)
    return {
        "oracle_kind": config.oracle_kind,
        "seed": config.seed,
        "final_eval_avg_profit_oracle": None if eval_df.empty else float(final["eval_avg_profit_oracle"]),
        "final_eval_avg_profit_victim": None if eval_df.empty else float(final["eval_avg_profit_victim"]),
        "final_eval_profit_asymmetry": None if eval_df.empty else float(final["eval_profit_asymmetry"]),
        "final_eval_avg_price_oracle": None if eval_df.empty else float(final["eval_avg_price_oracle"]),
        "final_eval_avg_price_victim": None if eval_df.empty else float(final["eval_avg_price_victim"]),
        "final_eval_market_price_mean": None if eval_df.empty else float(final["eval_market_price_mean"]),
        "final_eval_oracle_profit_gain": None if eval_df.empty else float(final["eval_oracle_profit_gain"]),
        "final_eval_victim_profit_gain": None if eval_df.empty else float(final["eval_victim_profit_gain"]),
        "final_eval_distance_to_nash_price": None if eval_df.empty else float(final["eval_distance_to_nash_price"]),
        "final_eval_distance_to_monopoly_price": None if eval_df.empty else float(final["eval_distance_to_monopoly_price"]),
        "max_eval_profit_asymmetry": None if eval_df.empty else float(asym.max()),
        "mean_last_5_eval_profit_asymmetry": None if eval_df.empty else float(asym.tail(5).mean()),
        "benchmarks": {
            "p_n": float(benchmarks.p_n),
            "p_m": float(benchmarks.p_m),
            "pi_n": [float(x) for x in benchmarks.pi_n],
            "pi_m": [float(x) for x in benchmarks.pi_m],
        },
    }


DQNOracleConfig = QVictimOracleConfig
run_dqn_oracle_vs_qvictim = run_experiment


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Reservoir Oracle vs adaptive Calvano Q-learning Victim")
    p.add_argument(
        "--oracle-kind",
        choices=[
            "actor_critic",
            "dqn",
            "dqn_jepa",
            "dqn_regret",
            "tabular_cfr",
            "tabular_multi_cfr",
            "tabular_lola",
            "tabular_model_lola",
            "tabular_rollout_lola",
        ],
        default="dqn",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--B", type=int, default=64)
    p.add_argument("--H", type=int, default=8)
    p.add_argument("--K", type=int, default=15)
    p.add_argument("--total-steps", type=int, default=50_000)
    p.add_argument("--rollout-steps", type=int, default=32)
    p.add_argument("--eval-every", type=int, default=5_000)
    p.add_argument("--eval-steps", type=int, default=5_000)
    p.add_argument("--log-every", type=int, default=1_000)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--lr-value", type=float, default=1e-3)
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--value-hidden-dim", type=int, default=128)
    p.add_argument("--reservoir-dim", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--train-every", type=int, default=4)
    p.add_argument("--target-update-every", type=int, default=1000)
    p.add_argument("--oracle-epsilon-decay-steps", type=int, default=50_000)
    p.add_argument("--jepa-latent-dim", type=int, default=64)
    p.add_argument("--jepa-coef", type=float, default=0.1)
    p.add_argument("--regret-coef", type=float, default=0.1)
    p.add_argument("--cfr-state-mode", choices=["victim_last_action", "joint_last_action"], default="joint_last_action")
    p.add_argument("--cfr-regret-decay", type=float, default=1.0)
    p.add_argument("--cfr-value-lr", type=float, default=0.1)
    p.add_argument("--cfr-gamma", type=float, default=0.95)
    p.add_argument("--lola-gamma", type=float, default=0.95)
    p.add_argument("--lola-tau", type=float, default=0.05)
    p.add_argument("--lola-epsilon", type=float, default=0.05)
    p.add_argument("--lola-state-mode", choices=["victim_last_action", "joint_last_action"], default="joint_last_action")
    p.add_argument("--model-lola-gamma", type=float, default=0.95)
    p.add_argument("--model-lola-tau", type=float, default=0.05)
    p.add_argument("--model-lola-epsilon", type=float, default=0.02)
    p.add_argument("--model-lola-victim-policy", choices=["greedy", "epsilon_greedy", "softmax"], default="epsilon_greedy")
    p.add_argument("--model-lola-future-policy", choices=["greedy", "epsilon_greedy", "softmax"], default="epsilon_greedy")
    p.add_argument("--model-lola-victim-softmax-tau", type=float, default=0.05)
    p.add_argument("--rollout-lola-horizon", type=int, default=20)
    p.add_argument("--rollout-lola-num-particles", type=int, default=32)
    p.add_argument("--rollout-lola-tau", type=float, default=0.05)
    p.add_argument("--rollout-lola-epsilon", type=float, default=0.02)
    p.add_argument("--rollout-lola-victim-policy", choices=["greedy", "epsilon_greedy", "softmax"], default="epsilon_greedy")
    p.add_argument("--rollout-lola-oracle-rollout-policy", choices=["greedy_best_response", "fixed_first_action"], default="greedy_best_response")
    p.add_argument("--rollout-lola-discount", type=float, default=0.95)
    p.add_argument("--rollout-lola-include-immediate", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--asymmetry-coef", type=float, default=0.0)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--out-dir", type=str, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = QVictimOracleConfig(
        oracle_kind=args.oracle_kind,
        seed=args.seed,
        B=args.B,
        H=args.H,
        K=args.K,
        total_steps=args.total_steps,
        rollout_steps=args.rollout_steps,
        eval_every=args.eval_every,
        eval_steps=args.eval_steps,
        log_every=args.log_every,
        lr=args.lr,
        lr_value=args.lr_value,
        hidden_dim=args.hidden_dim,
        value_hidden_dim=args.value_hidden_dim,
        reservoir_dim=args.reservoir_dim,
        batch_size=args.batch_size,
        train_every=args.train_every,
        target_update_every=args.target_update_every,
        oracle_epsilon_decay_steps=args.oracle_epsilon_decay_steps,
        jepa_latent_dim=args.jepa_latent_dim,
        jepa_coef=args.jepa_coef,
        regret_coef=args.regret_coef,
        cfr_state_mode=args.cfr_state_mode,
        cfr_regret_decay=args.cfr_regret_decay,
        cfr_value_lr=args.cfr_value_lr,
        cfr_gamma=args.cfr_gamma,
        lola_gamma=args.lola_gamma,
        lola_tau=args.lola_tau,
        lola_epsilon=args.lola_epsilon,
        lola_state_mode=args.lola_state_mode,
        model_lola_gamma=args.model_lola_gamma,
        model_lola_tau=args.model_lola_tau,
        model_lola_epsilon=args.model_lola_epsilon,
        model_lola_victim_policy=args.model_lola_victim_policy,
        model_lola_future_policy=args.model_lola_future_policy,
        model_lola_victim_softmax_tau=args.model_lola_victim_softmax_tau,
        rollout_lola_horizon=args.rollout_lola_horizon,
        rollout_lola_num_particles=args.rollout_lola_num_particles,
        rollout_lola_tau=args.rollout_lola_tau,
        rollout_lola_epsilon=args.rollout_lola_epsilon,
        rollout_lola_victim_policy=args.rollout_lola_victim_policy,
        rollout_lola_oracle_rollout_policy=args.rollout_lola_oracle_rollout_policy,
        rollout_lola_discount=args.rollout_lola_discount,
        rollout_lola_include_immediate=args.rollout_lola_include_immediate,
        asymmetry_coef=args.asymmetry_coef,
        device=args.device,
        out_dir=args.out_dir,
    )
    result = run_experiment(cfg)
    s = result["summary"]
    print(f"oracle_kind={s['oracle_kind']} seed={s['seed']}")
    print(f"final_eval_avg_profit_oracle={s['final_eval_avg_profit_oracle']}")
    print(f"final_eval_avg_profit_victim={s['final_eval_avg_profit_victim']}")
    print(f"final_eval_profit_asymmetry={s['final_eval_profit_asymmetry']}")
    print(f"final_eval_avg_price_oracle={s['final_eval_avg_price_oracle']}")
    print(f"final_eval_avg_price_victim={s['final_eval_avg_price_victim']}")
    print(f"final_eval_market_price_mean={s['final_eval_market_price_mean']}")


if __name__ == "__main__":
    main()
