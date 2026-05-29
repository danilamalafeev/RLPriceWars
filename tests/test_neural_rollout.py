import numpy as np
import torch

import calvano_market_cpp as cm
from neural.functional_policies import (
    init_linear_policy,
    init_mlp_policy,
    linear_policy_forward,
    mlp_policy_forward,
)
from neural.losses import OptimizerConfig, train_pg_step
from neural.observations import ObservationConfig, build_observation, observation_dim
from neural.reservoir import ReservoirConfig, init_reservoir_buffers, reservoir_observation, reservoir_update
from neural.rollout import RolloutConfig, collect_duopoly_rollout, sample_actions


def make_env(B=4, H=5, K=7):
    price_grid = np.linspace(1.0, 3.0, K, dtype=np.float32)
    config = {
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
        "random_seed": 123,
    }
    env = cm.create_env(config)
    cm.reset(env)
    return env, price_grid


def test_policy_shapes():
    B, Z, K = 4, 17, 7
    obs = torch.randn(B, Z)
    gen = torch.Generator().manual_seed(1)

    linear = init_linear_policy(gen, Z, K)
    logits, state = linear_policy_forward(linear, {}, obs, None)
    assert logits.shape == (B, K)
    assert state is None

    mlp = init_mlp_policy(gen, Z, hidden_dim=11, K=K)
    logits, state = mlp_policy_forward(mlp, {}, obs, None)
    assert logits.shape == (B, K)
    assert state is None


def test_sample_actions():
    logits = torch.randn(5, 7, requires_grad=True)
    actions, logp, entropy = sample_actions(logits, torch.Generator().manual_seed(2))
    assert actions.shape == (5,)
    assert logp.shape == (5,)
    assert entropy.shape == (5,)
    assert logp.requires_grad


def test_observation_builder():
    B, H = 3, 5
    price_history = np.linspace(1.0, 2.0, B * H * 2, dtype=np.float32).reshape(B, H, 2)
    current_prices = price_history[:, -1, :]
    rewards = np.zeros((B, 2), dtype=np.float32)
    market_share = np.full((B, 2), 0.4, dtype=np.float32)
    outside_share = np.full(B, 0.2, dtype=np.float32)
    margins = current_prices - 1.0
    obs = build_observation(
        price_history,
        current_prices,
        rewards,
        market_share,
        outside_share,
        margins,
        ObservationConfig(price_min=1.0, price_max=2.0),
    )
    assert obs.shape == (B, observation_dim(H))
    assert torch.isfinite(obs).all()
    assert not obs.requires_grad


def test_reservoir():
    B, F, R = 4, 9, 6
    gen = torch.Generator().manual_seed(3)
    buffers = init_reservoir_buffers(gen, ReservoirConfig(input_dim=F, reservoir_dim=R))
    features = torch.randn(B, F)
    h_prev = torch.zeros(B, R)
    h_t = reservoir_update(features, h_prev, buffers)
    obs = reservoir_observation(features, h_t)
    assert h_t.shape == (B, R)
    assert obs.shape == (B, F + R)
    assert not buffers["W_in"].requires_grad
    assert not buffers["W_res"].requires_grad
    assert torch.isfinite(h_t).all()


def test_rollout_shapes():
    B, H, K, T = 4, 5, 7, 8
    env, price_grid = make_env(B=B, H=H, K=K)
    Z = observation_dim(H)
    gen = torch.Generator().manual_seed(4)
    oracle = init_linear_policy(gen, Z, K)
    victim = init_linear_policy(gen, Z, K)

    rollout = collect_duopoly_rollout(
        env,
        linear_policy_forward,
        oracle,
        {},
        linear_policy_forward,
        victim,
        {},
        ObservationConfig(price_min=float(price_grid[0]), price_max=float(price_grid[-1])),
        RolloutConfig(T=T, B=B, H=H, K=K),
        gen,
    )
    assert rollout["obs"].shape == (T, B, Z)
    assert rollout["obs_oracle"].shape == (T, B, Z)
    assert rollout["obs_victim"].shape == (T, B, Z)
    assert rollout["logp"].shape == (T, B, 2)
    assert rollout["entropy"].shape == (T, B, 2)
    assert rollout["rewards"].shape == (T, B, 2)
    assert rollout["actions"].shape == (T, B, 2)


def test_rollout_supports_different_reservoir_dims():
    B, H, K, T = 3, 4, 5, 6
    env, price_grid = make_env(B=B, H=H, K=K)
    F = observation_dim(H)
    gen = torch.Generator().manual_seed(44)
    oracle_res = init_reservoir_buffers(gen, ReservoirConfig(input_dim=F, reservoir_dim=3))
    victim_res = init_reservoir_buffers(gen, ReservoirConfig(input_dim=F, reservoir_dim=5))
    oracle = init_linear_policy(gen, F + 3, K)
    victim = init_linear_policy(gen, F + 5, K)

    rollout = collect_duopoly_rollout(
        env,
        linear_policy_forward,
        oracle,
        {"reservoir": oracle_res},
        linear_policy_forward,
        victim,
        {"reservoir": victim_res},
        ObservationConfig(price_min=float(price_grid[0]), price_max=float(price_grid[-1])),
        RolloutConfig(T=T, B=B, H=H, K=K, use_reservoir=True),
        gen,
    )

    assert rollout["obs"].shape == (T, B, F + 3)
    assert rollout["obs_oracle"].shape == (T, B, F + 3)
    assert rollout["obs_victim"].shape == (T, B, F + 5)
    assert rollout["logp"].shape == (T, B, 2)


def test_logp_graph():
    B, H, K, T = 4, 5, 7, 6
    env, price_grid = make_env(B=B, H=H, K=K)
    Z = observation_dim(H)
    gen = torch.Generator().manual_seed(5)
    oracle = init_mlp_policy(gen, Z, hidden_dim=8, K=K)
    victim = init_mlp_policy(gen, Z, hidden_dim=8, K=K)

    rollout = collect_duopoly_rollout(
        env,
        mlp_policy_forward,
        oracle,
        {},
        mlp_policy_forward,
        victim,
        {},
        ObservationConfig(price_min=float(price_grid[0]), price_max=float(price_grid[-1])),
        RolloutConfig(T=T, B=B, H=H, K=K),
        gen,
    )
    loss = -rollout["logp"][:, :, 0].mean()
    loss.backward()
    grads = [p.grad for p in oracle.values()]
    assert all(g is not None for g in grads)
    assert any(torch.any(torch.abs(g) > 0) for g in grads)


def test_environment_detached():
    B, H, K, T = 3, 4, 5, 5
    env, price_grid = make_env(B=B, H=H, K=K)
    Z = observation_dim(H)
    gen = torch.Generator().manual_seed(6)
    oracle = init_linear_policy(gen, Z, K)
    victim = init_linear_policy(gen, Z, K)
    rollout = collect_duopoly_rollout(
        env,
        linear_policy_forward,
        oracle,
        {},
        linear_policy_forward,
        victim,
        {},
        ObservationConfig(price_min=float(price_grid[0]), price_max=float(price_grid[-1])),
        RolloutConfig(T=T, B=B, H=H, K=K),
        gen,
    )
    assert not rollout["rewards"].requires_grad
    assert not rollout["prices"].requires_grad


def test_pg_step():
    B, H, K, T = 4, 5, 7, 8
    env, price_grid = make_env(B=B, H=H, K=K)
    Z = observation_dim(H)
    gen = torch.Generator().manual_seed(7)
    oracle = init_mlp_policy(gen, Z, hidden_dim=10, K=K)
    victim = init_mlp_policy(gen, Z, hidden_dim=10, K=K)
    oracle_before = {k: v.detach().clone() for k, v in oracle.items()}
    victim_before = {k: v.detach().clone() for k, v in victim.items()}

    rollout = collect_duopoly_rollout(
        env,
        mlp_policy_forward,
        oracle,
        {},
        mlp_policy_forward,
        victim,
        {},
        ObservationConfig(price_min=float(price_grid[0]), price_max=float(price_grid[-1])),
        RolloutConfig(T=T, B=B, H=H, K=K),
        gen,
    )
    metrics = train_pg_step(oracle, victim, rollout, OptimizerConfig(lr=0.05, gamma=0.95))
    assert np.isfinite(metrics["loss_total"])
    assert any(not torch.allclose(oracle_before[k], oracle[k]) for k in oracle)
    assert any(not torch.allclose(victim_before[k], victim[k]) for k in victim)


def test_tiny_three_pg_updates_no_nan():
    B, H, K, T = 8, 5, 7, 16
    env, price_grid = make_env(B=B, H=H, K=K)
    Z = observation_dim(H)
    gen = torch.Generator().manual_seed(8)
    oracle = init_mlp_policy(gen, Z, hidden_dim=12, K=K)
    victim = init_mlp_policy(gen, Z, hidden_dim=12, K=K)
    rewards = []
    for _ in range(3):
        rollout = collect_duopoly_rollout(
            env,
            mlp_policy_forward,
            oracle,
            {},
            mlp_policy_forward,
            victim,
            {},
            ObservationConfig(price_min=float(price_grid[0]), price_max=float(price_grid[-1])),
            RolloutConfig(T=T, B=B, H=H, K=K),
            gen,
        )
        metrics = train_pg_step(oracle, victim, rollout, OptimizerConfig(lr=0.02, gamma=0.95))
        rewards.append(float(rollout["rewards"].mean().item()))
        assert np.isfinite(metrics["loss_total"])
    assert np.isfinite(rewards).all()
