from __future__ import annotations

import json
import subprocess
import sys

import numpy as np
import torch

from experiments.dqn_oracle_vs_qvictim import (
    DQNOracleConfig,
    clone_params,
    dqn_forward,
    dqn_jepa_train_step,
    dqn_regret_train_step,
    dqn_train_step,
    init_dqn_params,
    init_jepa_params,
    init_regret_params,
    init_static_cooperative_victim_state,
    init_tabular_cfr_state,
    init_replay_buffer,
    init_victim_state,
    jepa_encode,
    jepa_predict,
    make_calvano_vec_env,
    oracle_counterfactual_profit,
    parse_args as parse_oracle_args,
    replay_add,
    replay_sample,
    regret_forward,
    run_dqn_oracle_vs_qvictim,
    tabular_cfr_counterfactual_next_state_ids,
    tabular_cfr_select_actions,
    tabular_cfr_state_id,
    tabular_cfr_update,
    tabular_multi_cfr_cf_value,
    tabular_multi_cfr_value_update,
    tabular_lola_select_actions,
    tabular_model_lola_select_actions,
    tabular_model_lola_values,
    tabular_rollout_lola_select_actions,
    tabular_rollout_lola_values,
    tabular_rollout_lola_values_torch,
    victim_policy_probs_from_q,
    victim_policy_from_q,
    update_victim_q,
    victim_q_update,
    victim_select_actions,
)


def test_victim_state_shapes():
    B, K = 4, 5
    _, _, _, profit_matrix = make_calvano_vec_env(B, H=4, K=K, seed=0)
    state = init_victim_state(B, K, profit_matrix, delta=0.95, seed=1)
    assert state["Q"].shape == (B, K * K, K)
    assert state["state_id"].shape == (B,)


def test_victim_q_initialization_eq8():
    B, K = 3, 5
    _, _, _, profit_matrix = make_calvano_vec_env(B, H=4, K=K, seed=0)
    state = init_victim_state(B, K, profit_matrix, delta=0.95, seed=1)
    for av in range(K):
        expected = np.mean(profit_matrix[:, av, 1]) / (1.0 - 0.95)
        np.testing.assert_allclose(state["Q"][:, :, av], expected)


def test_victim_action_selection():
    B, K = 4, 5
    _, _, _, profit_matrix = make_calvano_vec_env(B, H=4, K=K, seed=0)
    state = init_victim_state(B, K, profit_matrix, delta=0.95, seed=1)
    state["Q"][:] = 1.0
    actions = victim_select_actions(state, K, beta=0.0, rng=np.random.default_rng(2), epsilon_override=0.0)
    assert np.all(actions == 0)
    random_actions = victim_select_actions(state, K, beta=0.0, rng=np.random.default_rng(2), epsilon_override=1.0)
    assert random_actions.shape == (B,)
    assert np.all((0 <= random_actions) & (random_actions < K))


def test_static_cooperative_victim_is_non_adaptive():
    B, K = 4, 5
    _, _, benchmarks, _ = make_calvano_vec_env(B, H=4, K=K, seed=0)
    state = init_static_cooperative_victim_state(B, K, benchmarks, seed=1)
    before_q = state["Q"].copy()
    actions0 = victim_select_actions(state, K, beta=0.0, rng=np.random.default_rng(2), epsilon_override=1.0)
    update_victim_q(
        state,
        oracle_actions=np.arange(B) % K,
        victim_actions=np.arange(B) % K,
        rewards_victim=np.linspace(0.0, 1.0, B),
        alpha=1.0,
        delta=0.0,
        K=K,
    )
    actions1 = victim_select_actions(state, K, beta=0.0, rng=np.random.default_rng(3), epsilon_override=1.0)
    assert np.all(actions0 == int(benchmarks.monopoly_actions[1]))
    assert np.all(actions1 == actions0)
    np.testing.assert_allclose(state["Q"], before_q)


def test_cli_default_victim_kind_is_adaptive_q(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["dqn_oracle_vs_qvictim"])
    args = parse_oracle_args()
    assert args.victim_kind == "adaptive_q"


def test_cli_accepts_rollout_lola_backend_torch(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["dqn_oracle_vs_qvictim", "--rollout-lola-backend", "torch"])
    args = parse_oracle_args()
    assert args.rollout_lola_backend == "torch"


def test_victim_policy_from_q_shape():
    B, K = 4, 5
    _, _, _, profit_matrix = make_calvano_vec_env(B, H=4, K=K, seed=0)
    state = init_victim_state(B, K, profit_matrix, delta=0.95, seed=1)
    state["Q"][:] = 0.0
    state["Q"][:, :, 3] = 1.0
    actions = victim_policy_from_q(state, K, greedy=True)
    assert actions.shape == (B,)
    assert np.all(actions == 3)


def test_victim_policy_probs_from_q_greedy():
    q = np.array([[0.0, 2.0, 1.0], [3.0, 1.0, 3.0]], dtype=np.float64)
    probs = victim_policy_probs_from_q(q, K=3, mode="greedy")
    expected = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float64)
    np.testing.assert_allclose(probs, expected)


def test_victim_policy_probs_from_q_epsilon_greedy():
    q = np.array([[[0.0, 2.0, 1.0], [3.0, 1.0, 0.0]], [[1.0, 0.0, 2.0], [0.0, 4.0, 3.0]]], dtype=np.float64)
    probs = victim_policy_probs_from_q(q, K=3, mode="epsilon_greedy", epsilon=np.array([0.3, 0.6]))
    assert probs.shape == q.shape
    np.testing.assert_allclose(probs.sum(axis=-1), 1.0)
    np.testing.assert_allclose(probs[0, 0], np.array([0.1, 0.8, 0.1]))
    np.testing.assert_allclose(probs[1, 0], np.array([0.2, 0.2, 0.6]))


def test_victim_policy_probs_from_q_softmax():
    q = np.array([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]], dtype=np.float64)
    probs = victim_policy_probs_from_q(q, K=3, mode="softmax", tau=1.0)
    assert probs.shape == q.shape
    np.testing.assert_allclose(probs.sum(axis=-1), 1.0)
    assert probs[0, 2] > probs[0, 1] > probs[0, 0]
    np.testing.assert_allclose(probs[1], np.full(3, 1.0 / 3.0))


def test_victim_q_update():
    B, K = 1, 3
    Q = np.zeros((B, K * K, K), dtype=np.float64)
    Q[0, 2, :] = np.array([1.0, 2.0, 3.0])
    state = {"Q": Q, "state_id": np.array([0], dtype=np.int64), "t": np.zeros(B, dtype=np.int64)}
    victim_q_update(
        state,
        state_id=np.array([0]),
        victim_actions=np.array([1]),
        rewards_victim=np.array([0.5]),
        next_state_id=np.array([2]),
        alpha=0.2,
        delta=0.9,
    )
    expected = 0.8 * 0.0 + 0.2 * (0.5 + 0.9 * 3.0)
    np.testing.assert_allclose(state["Q"][0, 0, 1], expected)
    assert state["state_id"][0] == 2


def test_dqn_forward_shape():
    B, Z, K = 4, 11, 5
    gen = torch.Generator().manual_seed(3)
    params = init_dqn_params(gen, Z, hidden_dim=7, K=K)
    q = dqn_forward(params, torch.randn(B, Z))
    assert q.shape == (B, K)


def test_replay_buffer_add_sample():
    B, Z, K = 4, 6, 5
    buffer = init_replay_buffer(capacity=20, obs_dim=Z, K=K)
    obs = torch.randn(B, Z)
    next_obs = torch.randn(B, Z)
    action = torch.randint(0, K, (B,))
    victim_action = torch.randint(0, K, (B,))
    reward = torch.randn(B)
    done = torch.zeros(B, dtype=torch.bool)
    cf_profit = torch.randn(B, K)
    replay_add(buffer, obs, action, reward, next_obs, done, victim_action, cf_profit)
    assert buffer["size"] == B
    batch = replay_sample(buffer, batch_size=3, generator=torch.Generator().manual_seed(4))
    assert batch["obs"].shape == (3, Z)
    assert batch["action"].shape == (3,)
    assert batch["victim_action"].shape == (3,)
    assert batch["cf_profit"].shape == (3, K)


def test_dqn_train_step():
    B, Z, K = 8, 6, 5
    gen = torch.Generator().manual_seed(5)
    params = init_dqn_params(gen, Z, hidden_dim=7, K=K)
    target = {k: v.detach().clone() for k, v in params.items()}
    before = {k: v.detach().clone() for k, v in params.items()}
    optimizer = torch.optim.Adam(list(params.values()), lr=1e-2)
    batch = {
        "obs": torch.randn(B, Z),
        "action": torch.randint(0, K, (B,)),
        "reward": torch.randn(B),
        "next_obs": torch.randn(B, Z),
        "done": torch.zeros(B, dtype=torch.bool),
    }
    loss = dqn_train_step(params, target, batch, optimizer, gamma=0.95)
    assert np.isfinite(loss)
    assert any(not torch.allclose(before[k], params[k]) for k in params)


def test_jepa_forward_shapes():
    B, Z, H, L, K = 4, 6, 7, 5, 3
    gen = torch.Generator().manual_seed(6)
    params = init_jepa_params(gen, Z, hidden_dim=H, latent_dim=L, K=K)
    obs = torch.randn(B, Z)
    action = torch.randint(0, K, (B,))
    latent = jepa_encode(params, obs)
    pred = jepa_predict(params, latent, action, K)
    assert latent.shape == (B, L)
    assert pred.shape == (B, L)


def test_regret_forward_shape():
    B, Z, K = 4, 6, 5
    gen = torch.Generator().manual_seed(8)
    params = init_regret_params(gen, Z, hidden_dim=7, K=K)
    pred = regret_forward(params, torch.randn(B, Z))
    assert pred.shape == (B, K)


def test_tabular_cfr_state_shapes():
    device = torch.device("cpu")
    state_v = init_tabular_cfr_state(B=4, K=5, state_mode="victim_last_action", device=device)
    assert state_v["regret_table"].shape == (4, 5, 5)
    assert state_v["value_table"].shape == (4, 5)
    assert state_v["state_id"].shape == (4,)
    state_j = init_tabular_cfr_state(B=4, K=5, state_mode="joint_last_action", device=device)
    assert state_j["regret_table"].shape == (4, 25, 5)
    assert state_j["value_table"].shape == (4, 25)
    assert state_j["state_id"].shape == (4,)


def test_tabular_multi_cfr_state_has_value_table():
    state = init_tabular_cfr_state(B=3, K=4, state_mode="joint_last_action", device=torch.device("cpu"))
    assert "value_table" in state
    assert state["value_table"].shape == (3, 16)
    assert torch.all(state["value_table"] == 0.0)


def test_tabular_cfr_state_id_victim_last_action():
    state_id = tabular_cfr_state_id(
        oracle_actions=np.array([0, 2, 4]),
        victim_actions=np.array([1, 3, 0]),
        K=5,
        state_mode="victim_last_action",
        device=torch.device("cpu"),
    )
    torch.testing.assert_close(state_id, torch.tensor([1, 3, 0], dtype=torch.long))


def test_tabular_cfr_state_id_joint_last_action():
    state_id = tabular_cfr_state_id(
        oracle_actions=np.array([0, 2, 4]),
        victim_actions=np.array([1, 3, 0]),
        K=5,
        state_mode="joint_last_action",
        device=torch.device("cpu"),
    )
    torch.testing.assert_close(state_id, torch.tensor([1, 13, 20], dtype=torch.long))


def test_tabular_cfr_select_actions_uniform_when_zero_regret():
    B, K = 5000, 5
    state = init_tabular_cfr_state(B=B, K=K, state_mode="victim_last_action", device=torch.device("cpu"))
    actions = tabular_cfr_select_actions(state, K, epsilon=0.0, generator=torch.Generator().manual_seed(10))
    counts = torch.bincount(actions, minlength=K).to(torch.float32) / B
    torch.testing.assert_close(counts, torch.full((K,), 1.0 / K), atol=0.025, rtol=0.0)


def test_tabular_cfr_update_accumulates_regret():
    state = init_tabular_cfr_state(B=2, K=3, state_mode="victim_last_action", device=torch.device("cpu"))
    prev_state_id = torch.tensor([0, 1], dtype=torch.long)
    oracle_actions = torch.tensor([1, 2], dtype=torch.long)
    cf_profit = torch.tensor([[1.0, 2.0, 0.5], [0.3, 0.4, 0.1]], dtype=torch.float32)
    tabular_cfr_update(state, prev_state_id, oracle_actions, cf_profit, regret_decay=1.0)
    expected0 = torch.tensor([-1.0, 0.0, -1.5])
    expected1 = torch.tensor([0.2, 0.3, 0.0])
    torch.testing.assert_close(state["regret_table"][0, 0], expected0)
    torch.testing.assert_close(state["regret_table"][1, 1], expected1)
    tabular_cfr_update(state, prev_state_id, oracle_actions, cf_profit, regret_decay=1.0)
    torch.testing.assert_close(state["regret_table"][0, 0], 2.0 * expected0)
    torch.testing.assert_close(state["regret_table"][1, 1], 2.0 * expected1)


def test_counterfactual_next_state_ids_joint_last_action():
    next_state = tabular_cfr_counterfactual_next_state_ids(
        torch.arange(4),
        victim_actions=np.array([1, 3]),
        K=4,
        state_mode="joint_last_action",
        device=torch.device("cpu"),
    )
    expected = torch.tensor([[1, 5, 9, 13], [3, 7, 11, 15]], dtype=torch.long)
    torch.testing.assert_close(next_state, expected)


def test_counterfactual_next_state_ids_victim_last_action():
    next_state = tabular_cfr_counterfactual_next_state_ids(
        torch.arange(4),
        victim_actions=np.array([1, 3]),
        K=4,
        state_mode="victim_last_action",
        device=torch.device("cpu"),
    )
    expected = torch.tensor([[1, 1, 1, 1], [3, 3, 3, 3]], dtype=torch.long)
    torch.testing.assert_close(next_state, expected)


def test_tabular_multi_cfr_cf_value_shape():
    state = init_tabular_cfr_state(B=2, K=3, state_mode="victim_last_action", device=torch.device("cpu"))
    state["value_table"][0] = torch.tensor([0.0, 1.0, 2.0])
    state["value_table"][1] = torch.tensor([3.0, 4.0, 5.0])
    cf_profit = torch.ones(2, 3)
    next_state_cf = torch.tensor([[0, 1, 2], [2, 1, 0]], dtype=torch.long)
    cf_value = tabular_multi_cfr_cf_value(state, cf_profit, next_state_cf, gamma=0.5)
    assert cf_value.shape == (2, 3)
    expected = torch.tensor([[1.0, 1.5, 2.0], [3.5, 3.0, 2.5]])
    torch.testing.assert_close(cf_value, expected)


def test_tabular_multi_cfr_value_update():
    state = init_tabular_cfr_state(B=2, K=3, state_mode="victim_last_action", device=torch.device("cpu"))
    state["value_table"][0, 2] = 4.0
    state["value_table"][1, 1] = 2.0
    tabular_multi_cfr_value_update(
        state,
        prev_state_id=torch.tensor([0, 1]),
        rewards_oracle=torch.tensor([1.0, 2.0]),
        next_state_real=torch.tensor([2, 1]),
        value_lr=0.5,
        gamma=0.25,
    )
    torch.testing.assert_close(state["value_table"][0, 0], torch.tensor(1.0))
    torch.testing.assert_close(state["value_table"][1, 1], torch.tensor(2.25))


def test_tabular_lola_select_actions_shape():
    B, K = 4, 5
    _, _, _, profit_matrix = make_calvano_vec_env(B, H=4, K=K, seed=0)
    victim = init_victim_state(B, K, profit_matrix, delta=0.95, seed=1)
    actions, metrics = tabular_lola_select_actions(
        victim,
        profit_matrix,
        K,
        gamma=0.95,
        tau=0.05,
        epsilon=0.05,
        generator=torch.Generator().manual_seed(11),
        device=torch.device("cpu"),
    )
    assert actions.shape == (B,)
    assert torch.all((0 <= actions) & (actions < K))
    assert {"lola_immediate_value", "lola_future_value", "lola_total_value", "lola_entropy"}.issubset(metrics)


def test_tabular_lola_select_actions_probs_finite():
    B, K = 4, 5
    _, _, _, profit_matrix = make_calvano_vec_env(B, H=4, K=K, seed=0)
    victim = init_victim_state(B, K, profit_matrix, delta=0.95, seed=1)
    _, metrics = tabular_lola_select_actions(
        victim,
        profit_matrix,
        K,
        gamma=0.95,
        tau=0.05,
        epsilon=0.05,
        generator=torch.Generator().manual_seed(12),
        device=torch.device("cpu"),
    )
    assert all(np.isfinite(v) for v in metrics.values())


def test_tabular_model_lola_values_shape():
    B, K = 4, 5
    _, _, _, profit_matrix = make_calvano_vec_env(B, H=4, K=K, seed=0)
    victim = init_victim_state(B, K, profit_matrix, delta=0.95, seed=1)
    values, metrics = tabular_model_lola_values(
        victim,
        profit_matrix,
        K,
        alpha=0.15,
        delta=0.95,
        beta=4e-6,
        gamma_lola=0.95,
        victim_policy_mode="epsilon_greedy",
        future_policy_mode="epsilon_greedy",
        victim_softmax_tau=0.05,
    )
    assert values.shape == (B, K)
    assert {
        "model_lola_immediate_value",
        "model_lola_future_value",
        "model_lola_total_value",
        "model_lola_current_victim_entropy",
        "model_lola_future_victim_entropy",
    }.issubset(metrics)


def test_tabular_model_lola_values_finite():
    B, K = 4, 5
    _, _, _, profit_matrix = make_calvano_vec_env(B, H=4, K=K, seed=0)
    victim = init_victim_state(B, K, profit_matrix, delta=0.95, seed=1)
    values, metrics = tabular_model_lola_values(
        victim,
        profit_matrix,
        K,
        alpha=0.15,
        delta=0.95,
        beta=4e-6,
        gamma_lola=0.95,
        victim_policy_mode="softmax",
        future_policy_mode="softmax",
        victim_softmax_tau=0.05,
    )
    assert np.isfinite(values).all()
    assert all(np.isfinite(v) for v in metrics.values())


def test_tabular_model_lola_select_actions_shape():
    values = np.random.default_rng(0).normal(size=(4, 5))
    actions, metrics = tabular_model_lola_select_actions(
        values,
        tau=0.05,
        epsilon=0.02,
        generator=torch.Generator().manual_seed(13),
        device=torch.device("cpu"),
    )
    assert actions.shape == (4,)
    assert torch.all((0 <= actions) & (actions < 5))
    assert {"model_lola_entropy", "model_lola_value_mean", "model_lola_value_std"}.issubset(metrics)
    assert all(np.isfinite(v) for v in metrics.values())


def test_tabular_rollout_lola_values_shape():
    B, K = 3, 4
    _, price_grid, _, profit_matrix = make_calvano_vec_env(B, H=4, K=K, seed=0)
    victim = init_victim_state(B, K, profit_matrix, delta=0.95, seed=1)
    values, metrics = tabular_rollout_lola_values(
        victim,
        profit_matrix,
        K,
        alpha=0.15,
        delta=0.95,
        beta=4e-6,
        horizon=3,
        num_particles=2,
        victim_policy_mode="epsilon_greedy",
        oracle_rollout_policy="greedy_best_response",
        discount=0.95,
        include_immediate=True,
        rng=np.random.default_rng(14),
        price_grid=price_grid,
    )
    assert values.shape == (B, K)
    assert {
        "rollout_lola_first_step_profit",
        "rollout_lola_future_profit",
        "rollout_lola_victim_price_simulated",
        "rollout_lola_oracle_price_simulated",
    }.issubset(metrics)


def test_tabular_rollout_lola_values_finite():
    B, K = 3, 4
    _, price_grid, _, profit_matrix = make_calvano_vec_env(B, H=4, K=K, seed=0)
    victim = init_victim_state(B, K, profit_matrix, delta=0.95, seed=1)
    values, metrics = tabular_rollout_lola_values(
        victim,
        profit_matrix,
        K,
        alpha=0.15,
        delta=0.95,
        beta=4e-6,
        horizon=4,
        num_particles=3,
        victim_policy_mode="epsilon_greedy",
        oracle_rollout_policy="fixed_first_action",
        discount=0.95,
        include_immediate=False,
        rng=np.random.default_rng(15),
        price_grid=price_grid,
    )
    assert np.isfinite(values).all()
    assert all(np.isfinite(v) for v in metrics.values())


def test_tabular_rollout_lola_values_torch_shape_finite_cpu():
    B, K = 3, 4
    _, price_grid, _, profit_matrix = make_calvano_vec_env(B, H=4, K=K, seed=0)
    victim = init_victim_state(B, K, profit_matrix, delta=0.95, seed=1)
    values, metrics = tabular_rollout_lola_values_torch(
        victim,
        profit_matrix,
        K,
        alpha=0.15,
        delta=0.95,
        beta=4e-6,
        horizon=3,
        num_particles=2,
        victim_policy_mode="epsilon_greedy",
        oracle_rollout_policy="greedy_best_response",
        discount=0.95,
        include_immediate=True,
        generator=torch.Generator().manual_seed(17),
        device=torch.device("cpu"),
        price_grid=price_grid,
    )
    assert values.shape == (B, K)
    assert torch.isfinite(values).all()
    assert {
        "rollout_lola_first_step_profit",
        "rollout_lola_future_profit",
        "rollout_lola_victim_price_simulated",
        "rollout_lola_oracle_price_simulated",
    }.issubset(metrics)
    assert all(np.isfinite(v) for v in metrics.values())


def test_tabular_rollout_lola_values_torch_matches_numpy_greedy_tiny():
    B, K = 2, 4
    _, price_grid, _, profit_matrix = make_calvano_vec_env(B, H=4, K=K, seed=0)
    victim = init_victim_state(B, K, profit_matrix, delta=0.95, seed=1)
    numpy_values, numpy_metrics = tabular_rollout_lola_values(
        victim,
        profit_matrix,
        K,
        alpha=0.15,
        delta=0.95,
        beta=4e-6,
        horizon=3,
        num_particles=3,
        victim_policy_mode="greedy",
        oracle_rollout_policy="fixed_first_action",
        discount=0.95,
        include_immediate=True,
        rng=np.random.default_rng(18),
        price_grid=price_grid,
    )
    torch_values, torch_metrics = tabular_rollout_lola_values_torch(
        victim,
        profit_matrix,
        K,
        alpha=0.15,
        delta=0.95,
        beta=4e-6,
        horizon=3,
        num_particles=3,
        victim_policy_mode="greedy",
        oracle_rollout_policy="fixed_first_action",
        discount=0.95,
        include_immediate=True,
        generator=torch.Generator().manual_seed(18),
        device=torch.device("cpu"),
        price_grid=price_grid,
    )
    np.testing.assert_allclose(torch_values.detach().cpu().numpy(), numpy_values, rtol=1e-5, atol=1e-5)
    for key, numpy_value in numpy_metrics.items():
        np.testing.assert_allclose(torch_metrics[key], numpy_value, rtol=1e-5, atol=1e-5)


def test_tabular_rollout_lola_select_actions_shape():
    values = np.random.default_rng(0).normal(size=(4, 5))
    actions, metrics = tabular_rollout_lola_select_actions(
        values,
        tau=0.05,
        epsilon=0.02,
        generator=torch.Generator().manual_seed(16),
        device=torch.device("cpu"),
        price_grid=np.linspace(1.0, 2.0, 5),
    )
    assert actions.shape == (4,)
    assert torch.all((0 <= actions) & (actions < 5))
    assert {"rollout_lola_value_mean", "rollout_lola_value_std", "rollout_lola_entropy", "rollout_lola_best_action_price"}.issubset(metrics)
    assert all(np.isfinite(v) for v in metrics.values())


def test_oracle_counterfactual_profit_shape():
    K = 4
    profit_matrix = np.zeros((K, K, 2), dtype=np.float32)
    for a_o in range(K):
        for a_v in range(K):
            profit_matrix[a_o, a_v, 0] = 10 * a_o + a_v
    cf = oracle_counterfactual_profit(profit_matrix, np.array([0, 2, 3]), torch.device("cpu"))
    assert cf.shape == (3, K)
    torch.testing.assert_close(cf[1], torch.tensor([2.0, 12.0, 22.0, 32.0]))


def test_dqn_jepa_train_step_updates_params():
    B, Z, K = 8, 6, 5
    gen = torch.Generator().manual_seed(7)
    q_params = init_dqn_params(gen, Z, hidden_dim=7, K=K)
    target_q_params = clone_params(q_params)
    jepa_params = init_jepa_params(gen, Z, hidden_dim=7, latent_dim=4, K=K)
    target_jepa_params = clone_params(jepa_params)
    before_q = {k: v.detach().clone() for k, v in q_params.items()}
    before_jepa = {k: v.detach().clone() for k, v in jepa_params.items()}
    optimizer = torch.optim.Adam(list(q_params.values()) + list(jepa_params.values()), lr=1e-2)
    batch = {
        "obs": torch.randn(B, Z),
        "action": torch.randint(0, K, (B,)),
        "reward": torch.randn(B),
        "next_obs": torch.randn(B, Z),
        "done": torch.zeros(B, dtype=torch.bool),
    }
    metrics = dqn_jepa_train_step(
        q_params,
        target_q_params,
        jepa_params,
        target_jepa_params,
        batch,
        optimizer,
        gamma=0.95,
        jepa_coef=0.1,
        K=K,
    )
    assert set(metrics) == {"dqn_loss", "jepa_loss", "total_loss", "q_mean", "q_max"}
    assert all(np.isfinite(v) for v in metrics.values())
    assert any(not torch.allclose(before_q[k], q_params[k]) for k in q_params)
    assert any(not torch.allclose(before_jepa[k], jepa_params[k]) for k in jepa_params)


def test_dqn_regret_train_step_updates_params():
    B, Z, K = 8, 6, 5
    gen = torch.Generator().manual_seed(9)
    q_params = init_dqn_params(gen, Z, hidden_dim=7, K=K)
    target_q_params = clone_params(q_params)
    regret_params = init_regret_params(gen, Z, hidden_dim=7, K=K)
    before_q = {k: v.detach().clone() for k, v in q_params.items()}
    before_regret = {k: v.detach().clone() for k, v in regret_params.items()}
    optimizer = torch.optim.Adam(list(q_params.values()) + list(regret_params.values()), lr=1e-2)
    batch = {
        "obs": torch.randn(B, Z),
        "action": torch.randint(0, K, (B,)),
        "victim_action": torch.randint(0, K, (B,)),
        "reward": torch.randn(B),
        "next_obs": torch.randn(B, Z),
        "done": torch.zeros(B, dtype=torch.bool),
        "cf_profit": torch.randn(B, K),
    }
    metrics = dqn_regret_train_step(
        q_params,
        target_q_params,
        regret_params,
        batch,
        optimizer,
        gamma=0.95,
        regret_coef=0.1,
    )
    assert set(metrics) == {"dqn_loss", "regret_loss", "total_loss", "q_mean", "q_max"}
    assert all(np.isfinite(v) for v in metrics.values())
    assert any(not torch.allclose(before_q[k], q_params[k]) for k in q_params)
    assert any(not torch.allclose(before_regret[k], regret_params[k]) for k in regret_params)


def test_online_loop_smoke():
    config = DQNOracleConfig(
        seed=0,
        B=4,
        H=5,
        K=7,
        total_steps=20,
        rollout_steps=5,
        train_every=2,
        eval_every=10,
        eval_steps=20,
        batch_size=8,
        reservoir_dim=6,
        hidden_dim=8,
        replay_capacity=100,
    )
    result = run_dqn_oracle_vs_qvictim(config)
    train_df = result["train_metrics"]
    eval_df = result["eval_metrics"]
    assert len(train_df) > 0
    assert len(eval_df) > 0
    assert np.isfinite(
        train_df.drop(
            columns=[
                "jepa_loss",
                "regret_loss",
                "avg_positive_regret",
                "avg_regret_abs",
                "avg_strategy_entropy",
                "avg_value",
                "lola_immediate_value",
                "lola_future_value",
                "lola_total_value",
                "lola_entropy",
                "model_lola_value",
                "model_lola_value_std",
                "model_lola_entropy",
                "model_lola_immediate_value",
                "model_lola_future_value",
                "model_lola_total_value",
                "model_lola_current_victim_entropy",
                "model_lola_future_victim_entropy",
                "rollout_lola_value_mean",
                "rollout_lola_value_std",
                "rollout_lola_entropy",
                "rollout_lola_best_action_price",
                "rollout_lola_first_step_profit",
                "rollout_lola_future_profit",
                "rollout_lola_victim_price_simulated",
                "rollout_lola_oracle_price_simulated",
                "victim_pred_accuracy",
            ]
        )
        .select_dtypes(include=[float, int])
        .to_numpy()
    ).all()
    assert np.isfinite(eval_df.select_dtypes(include=[float, int]).to_numpy()).all()


def test_dqn_jepa_online_loop_smoke():
    config = DQNOracleConfig(
        oracle_kind="dqn_jepa",
        seed=0,
        B=4,
        H=5,
        K=7,
        total_steps=20,
        rollout_steps=5,
        train_every=2,
        eval_every=10,
        eval_steps=20,
        batch_size=8,
        reservoir_dim=6,
        hidden_dim=8,
        replay_capacity=100,
        jepa_latent_dim=4,
        jepa_coef=0.1,
    )
    result = run_dqn_oracle_vs_qvictim(config)
    train_df = result["train_metrics"]
    eval_df = result["eval_metrics"]
    assert len(train_df) > 0
    assert len(eval_df) > 0
    assert {"dqn_loss", "jepa_loss", "total_loss", "avg_price_oracle", "avg_price_victim"}.issubset(train_df.columns)
    assert np.isfinite(
        train_df.drop(
            columns=[
                "regret_loss",
                "avg_positive_regret",
                "avg_regret_abs",
                "avg_strategy_entropy",
                "avg_value",
                "lola_immediate_value",
                "lola_future_value",
                "lola_total_value",
                "lola_entropy",
                "model_lola_value",
                "model_lola_value_std",
                "model_lola_entropy",
                "model_lola_immediate_value",
                "model_lola_future_value",
                "model_lola_total_value",
                "model_lola_current_victim_entropy",
                "model_lola_future_victim_entropy",
                "rollout_lola_value_mean",
                "rollout_lola_value_std",
                "rollout_lola_entropy",
                "rollout_lola_best_action_price",
                "rollout_lola_first_step_profit",
                "rollout_lola_future_profit",
                "rollout_lola_victim_price_simulated",
                "rollout_lola_oracle_price_simulated",
                "victim_pred_accuracy",
            ]
        )
        .select_dtypes(include=[float, int])
        .to_numpy()
    ).all()
    assert np.isfinite(eval_df.select_dtypes(include=[float, int]).to_numpy()).all()


def test_dqn_regret_online_loop_smoke():
    config = DQNOracleConfig(
        oracle_kind="dqn_regret",
        seed=0,
        B=4,
        H=5,
        K=7,
        total_steps=20,
        rollout_steps=5,
        train_every=2,
        eval_every=10,
        eval_steps=20,
        batch_size=8,
        reservoir_dim=6,
        hidden_dim=8,
        replay_capacity=100,
        regret_coef=0.1,
    )
    result = run_dqn_oracle_vs_qvictim(config)
    train_df = result["train_metrics"]
    eval_df = result["eval_metrics"]
    assert len(train_df) > 0
    assert len(eval_df) > 0
    assert {"dqn_loss", "regret_loss", "total_loss", "avg_price_oracle", "avg_price_victim"}.issubset(train_df.columns)
    assert np.isfinite(
        train_df.drop(
            columns=[
                "jepa_loss",
                "avg_positive_regret",
                "avg_regret_abs",
                "avg_strategy_entropy",
                "avg_value",
                "lola_immediate_value",
                "lola_future_value",
                "lola_total_value",
                "lola_entropy",
                "model_lola_value",
                "model_lola_value_std",
                "model_lola_entropy",
                "model_lola_immediate_value",
                "model_lola_future_value",
                "model_lola_total_value",
                "model_lola_current_victim_entropy",
                "model_lola_future_victim_entropy",
                "rollout_lola_value_mean",
                "rollout_lola_value_std",
                "rollout_lola_entropy",
                "rollout_lola_best_action_price",
                "rollout_lola_first_step_profit",
                "rollout_lola_future_profit",
                "rollout_lola_victim_price_simulated",
                "rollout_lola_oracle_price_simulated",
                "victim_pred_accuracy",
            ]
        )
        .select_dtypes(include=[float, int])
        .to_numpy()
    ).all()
    assert np.isfinite(eval_df.select_dtypes(include=[float, int]).to_numpy()).all()


def test_tabular_cfr_online_loop_smoke():
    config = DQNOracleConfig(
        oracle_kind="tabular_cfr",
        seed=0,
        B=4,
        H=5,
        K=7,
        total_steps=20,
        eval_every=10,
        eval_steps=20,
        cfr_state_mode="joint_last_action",
    )
    result = run_dqn_oracle_vs_qvictim(config)
    train_df = result["train_metrics"]
    eval_df = result["eval_metrics"]
    assert len(train_df) > 0
    assert len(eval_df) > 0
    assert {"avg_positive_regret", "avg_strategy_entropy", "avg_price_oracle", "avg_price_victim"}.issubset(train_df.columns)
    numeric = train_df.drop(
        columns=[
            "dqn_loss",
            "jepa_loss",
            "regret_loss",
            "q_mean",
            "q_max",
            "lola_immediate_value",
            "lola_future_value",
            "lola_total_value",
            "lola_entropy",
            "model_lola_value",
            "model_lola_value_std",
            "model_lola_entropy",
            "model_lola_immediate_value",
            "model_lola_future_value",
            "model_lola_total_value",
            "model_lola_current_victim_entropy",
            "model_lola_future_victim_entropy",
            "rollout_lola_value_mean",
            "rollout_lola_value_std",
            "rollout_lola_entropy",
            "rollout_lola_best_action_price",
            "rollout_lola_first_step_profit",
            "rollout_lola_future_profit",
            "rollout_lola_victim_price_simulated",
            "rollout_lola_oracle_price_simulated",
            "victim_pred_accuracy",
        ]
    ).select_dtypes(include=[float, int]).to_numpy()
    assert np.isfinite(numeric).all()
    assert np.isfinite(eval_df.select_dtypes(include=[float, int]).to_numpy()).all()


def test_tabular_multi_cfr_loop_smoke():
    config = DQNOracleConfig(
        oracle_kind="tabular_multi_cfr",
        seed=0,
        B=4,
        H=5,
        K=7,
        total_steps=20,
        eval_every=10,
        eval_steps=20,
        cfr_state_mode="joint_last_action",
        cfr_value_lr=0.1,
        cfr_gamma=0.95,
    )
    result = run_dqn_oracle_vs_qvictim(config)
    train_df = result["train_metrics"]
    eval_df = result["eval_metrics"]
    assert len(train_df) > 0
    assert len(eval_df) > 0
    assert {"avg_positive_regret", "avg_strategy_entropy", "avg_value", "avg_price_oracle", "avg_price_victim"}.issubset(train_df.columns)
    numeric = train_df.drop(
        columns=[
            "dqn_loss",
            "jepa_loss",
            "regret_loss",
            "q_mean",
            "q_max",
            "lola_immediate_value",
            "lola_future_value",
            "lola_total_value",
            "lola_entropy",
            "model_lola_value",
            "model_lola_value_std",
            "model_lola_entropy",
            "model_lola_immediate_value",
            "model_lola_future_value",
            "model_lola_total_value",
            "model_lola_current_victim_entropy",
            "model_lola_future_victim_entropy",
            "rollout_lola_value_mean",
            "rollout_lola_value_std",
            "rollout_lola_entropy",
            "rollout_lola_best_action_price",
            "rollout_lola_first_step_profit",
            "rollout_lola_future_profit",
            "rollout_lola_victim_price_simulated",
            "rollout_lola_oracle_price_simulated",
            "victim_pred_accuracy",
        ]
    ).select_dtypes(include=[float, int]).to_numpy()
    assert np.isfinite(numeric).all()
    assert np.isfinite(eval_df.select_dtypes(include=[float, int]).to_numpy()).all()


def test_tabular_lola_loop_smoke():
    config = DQNOracleConfig(
        oracle_kind="tabular_lola",
        seed=0,
        B=4,
        H=5,
        K=7,
        total_steps=20,
        eval_every=10,
        eval_steps=20,
        lola_gamma=0.95,
        lola_tau=0.05,
        lola_epsilon=0.05,
    )
    result = run_dqn_oracle_vs_qvictim(config)
    train_df = result["train_metrics"]
    eval_df = result["eval_metrics"]
    assert len(train_df) > 0
    assert len(eval_df) > 0
    assert {"lola_immediate_value", "lola_future_value", "lola_total_value", "lola_entropy", "victim_pred_accuracy"}.issubset(train_df.columns)
    numeric = train_df.drop(
        columns=[
            "dqn_loss",
            "jepa_loss",
            "regret_loss",
            "q_mean",
            "q_max",
            "avg_positive_regret",
            "avg_regret_abs",
            "avg_strategy_entropy",
            "avg_value",
            "model_lola_value",
            "model_lola_value_std",
            "model_lola_entropy",
            "model_lola_immediate_value",
            "model_lola_future_value",
            "model_lola_total_value",
            "model_lola_current_victim_entropy",
            "model_lola_future_victim_entropy",
            "rollout_lola_value_mean",
            "rollout_lola_value_std",
            "rollout_lola_entropy",
            "rollout_lola_best_action_price",
            "rollout_lola_first_step_profit",
            "rollout_lola_future_profit",
            "rollout_lola_victim_price_simulated",
            "rollout_lola_oracle_price_simulated",
        ]
    ).select_dtypes(include=[float, int]).to_numpy()
    assert np.isfinite(numeric).all()
    assert np.isfinite(eval_df.select_dtypes(include=[float, int]).to_numpy()).all()


def test_tabular_model_lola_loop_smoke():
    config = DQNOracleConfig(
        oracle_kind="tabular_model_lola",
        seed=0,
        B=4,
        H=5,
        K=7,
        total_steps=20,
        eval_every=10,
        eval_steps=20,
        model_lola_gamma=0.95,
        model_lola_tau=0.05,
        model_lola_epsilon=0.02,
        model_lola_victim_policy="epsilon_greedy",
        model_lola_future_policy="epsilon_greedy",
    )
    result = run_dqn_oracle_vs_qvictim(config)
    train_df = result["train_metrics"]
    eval_df = result["eval_metrics"]
    assert len(train_df) > 0
    assert len(eval_df) > 0
    assert {
        "model_lola_value",
        "model_lola_entropy",
        "model_lola_immediate_value",
        "model_lola_future_value",
        "model_lola_current_victim_entropy",
        "model_lola_future_victim_entropy",
        "victim_pred_accuracy",
    }.issubset(train_df.columns)
    numeric = train_df.drop(
        columns=[
            "dqn_loss",
            "jepa_loss",
            "regret_loss",
            "q_mean",
            "q_max",
            "avg_positive_regret",
            "avg_regret_abs",
            "avg_strategy_entropy",
            "avg_value",
            "lola_immediate_value",
            "lola_future_value",
            "lola_total_value",
            "lola_entropy",
            "rollout_lola_value_mean",
            "rollout_lola_value_std",
            "rollout_lola_entropy",
            "rollout_lola_best_action_price",
            "rollout_lola_first_step_profit",
            "rollout_lola_future_profit",
            "rollout_lola_victim_price_simulated",
            "rollout_lola_oracle_price_simulated",
        ]
    ).select_dtypes(include=[float, int]).to_numpy()
    assert np.isfinite(numeric).all()
    assert np.isfinite(eval_df.select_dtypes(include=[float, int]).to_numpy()).all()


def test_tabular_rollout_lola_loop_smoke():
    config = DQNOracleConfig(
        oracle_kind="tabular_rollout_lola",
        seed=0,
        B=4,
        H=5,
        K=7,
        total_steps=20,
        eval_every=10,
        eval_steps=20,
        rollout_lola_horizon=3,
        rollout_lola_num_particles=2,
        rollout_lola_tau=0.05,
        rollout_lola_epsilon=0.02,
        rollout_lola_backend="torch",
    )
    result = run_dqn_oracle_vs_qvictim(config)
    train_df = result["train_metrics"]
    eval_df = result["eval_metrics"]
    assert len(train_df) > 0
    assert len(eval_df) > 0
    assert {
        "rollout_lola_value_mean",
        "rollout_lola_value_std",
        "rollout_lola_entropy",
        "rollout_lola_best_action_price",
        "rollout_lola_first_step_profit",
        "rollout_lola_future_profit",
        "rollout_lola_victim_price_simulated",
        "rollout_lola_oracle_price_simulated",
    }.issubset(train_df.columns)
    numeric = train_df.drop(
        columns=[
            "dqn_loss",
            "jepa_loss",
            "regret_loss",
            "q_mean",
            "q_max",
            "avg_positive_regret",
            "avg_regret_abs",
            "avg_strategy_entropy",
            "avg_value",
            "lola_immediate_value",
            "lola_future_value",
            "lola_total_value",
            "lola_entropy",
            "model_lola_value",
            "model_lola_value_std",
            "model_lola_entropy",
            "model_lola_immediate_value",
            "model_lola_future_value",
            "model_lola_total_value",
            "model_lola_current_victim_entropy",
            "model_lola_future_victim_entropy",
        ]
    ).select_dtypes(include=[float, int]).to_numpy()
    assert np.isfinite(numeric).all()
    assert np.isfinite(eval_df.select_dtypes(include=[float, int]).to_numpy()).all()


def test_tabular_rollout_lola_progress_jsonl(tmp_path):
    out_dir = tmp_path / "rollout_progress"
    config = DQNOracleConfig(
        oracle_kind="tabular_rollout_lola",
        seed=0,
        B=3,
        H=4,
        K=5,
        total_steps=6,
        log_every=3,
        eval_every=3,
        eval_steps=4,
        rollout_lola_horizon=2,
        rollout_lola_num_particles=2,
        rollout_lola_backend="torch",
        out_dir=str(out_dir),
    )
    run_dqn_oracle_vs_qvictim(config)
    progress_path = out_dir / "progress.jsonl"
    assert progress_path.exists()
    rows = [json.loads(line) for line in progress_path.read_text(encoding="utf-8").splitlines()]
    assert len(rows) >= 2
    assert rows[-1]["step"] == 6
    assert rows[-1]["rollout_lola_backend"] == "torch"
    assert rows[-1]["device"] == "cpu"
    assert "steps_per_second" in rows[-1]


def test_cli_smoke(tmp_path):
    out_dir = tmp_path / "dqn_cli"
    cmd = [
        sys.executable,
        "-m",
        "experiments.dqn_oracle_vs_qvictim",
        "--total-steps",
        "20",
        "--B",
        "4",
        "--H",
        "5",
        "--K",
        "7",
        "--train-every",
        "2",
        "--eval-every",
        "10",
        "--eval-steps",
        "20",
        "--batch-size",
        "8",
        "--reservoir-dim",
        "6",
        "--hidden-dim",
        "8",
        "--out-dir",
        str(out_dir),
    ]
    subprocess.run(cmd, check=True)
    assert (out_dir / "train_metrics.csv").exists()
    assert (out_dir / "eval_metrics.csv").exists()
    assert (out_dir / "summary.json").exists()
    summary = json.loads((out_dir / "summary.json").read_text())
    assert "final_eval_avg_profit_oracle" in summary


def test_static_victim_run_writes_summary_kind(tmp_path):
    out_dir = tmp_path / "static_victim"
    config = DQNOracleConfig(
        oracle_kind="tabular_cfr",
        victim_kind="static_cooperative",
        seed=0,
        B=4,
        H=5,
        K=7,
        total_steps=20,
        eval_every=10,
        eval_steps=20,
        out_dir=str(out_dir),
    )
    result = run_dqn_oracle_vs_qvictim(config)
    summary = json.loads((out_dir / "summary.json").read_text())
    assert result["summary"]["victim_kind"] == "static_cooperative"
    assert summary["victim_kind"] == "static_cooperative"


def test_dqn_jepa_cli_smoke(tmp_path):
    out_dir = tmp_path / "dqn_jepa_cli"
    cmd = [
        sys.executable,
        "-m",
        "experiments.dqn_oracle_vs_qvictim",
        "--oracle-kind",
        "dqn_jepa",
        "--total-steps",
        "20",
        "--B",
        "4",
        "--H",
        "5",
        "--K",
        "7",
        "--train-every",
        "2",
        "--eval-every",
        "10",
        "--eval-steps",
        "20",
        "--batch-size",
        "8",
        "--reservoir-dim",
        "6",
        "--hidden-dim",
        "8",
        "--jepa-latent-dim",
        "4",
        "--jepa-coef",
        "0.1",
        "--out-dir",
        str(out_dir),
    ]
    subprocess.run(cmd, check=True)
    assert (out_dir / "train_metrics.csv").exists()
    assert (out_dir / "eval_metrics.csv").exists()
    assert (out_dir / "summary.json").exists()
    summary = json.loads((out_dir / "summary.json").read_text())
    assert summary["oracle_kind"] == "dqn_jepa"
    assert "final_eval_avg_price_oracle" in summary


def test_dqn_regret_cli_smoke(tmp_path):
    out_dir = tmp_path / "dqn_regret_cli"
    cmd = [
        sys.executable,
        "-m",
        "experiments.dqn_oracle_vs_qvictim",
        "--oracle-kind",
        "dqn_regret",
        "--total-steps",
        "20",
        "--B",
        "4",
        "--H",
        "5",
        "--K",
        "7",
        "--train-every",
        "2",
        "--eval-every",
        "10",
        "--eval-steps",
        "20",
        "--batch-size",
        "8",
        "--reservoir-dim",
        "6",
        "--hidden-dim",
        "8",
        "--regret-coef",
        "0.1",
        "--out-dir",
        str(out_dir),
    ]
    subprocess.run(cmd, check=True)
    assert (out_dir / "train_metrics.csv").exists()
    assert (out_dir / "eval_metrics.csv").exists()
    assert (out_dir / "summary.json").exists()
    summary = json.loads((out_dir / "summary.json").read_text())
    assert summary["oracle_kind"] == "dqn_regret"
    assert "final_eval_avg_price_oracle" in summary


def test_tabular_cfr_cli_smoke(tmp_path):
    out_dir = tmp_path / "tabular_cfr_cli"
    cmd = [
        sys.executable,
        "-m",
        "experiments.dqn_oracle_vs_qvictim",
        "--oracle-kind",
        "tabular_cfr",
        "--total-steps",
        "20",
        "--B",
        "4",
        "--H",
        "5",
        "--K",
        "7",
        "--eval-every",
        "10",
        "--eval-steps",
        "20",
        "--out-dir",
        str(out_dir),
    ]
    subprocess.run(cmd, check=True)
    assert (out_dir / "train_metrics.csv").exists()
    assert (out_dir / "eval_metrics.csv").exists()
    assert (out_dir / "summary.json").exists()
    summary = json.loads((out_dir / "summary.json").read_text())
    assert summary["oracle_kind"] == "tabular_cfr"
    assert "final_eval_avg_price_oracle" in summary


def test_cli_smoke_tabular_multi_cfr(tmp_path):
    out_dir = tmp_path / "tabular_multi_cfr_cli"
    cmd = [
        sys.executable,
        "-m",
        "experiments.dqn_oracle_vs_qvictim",
        "--oracle-kind",
        "tabular_multi_cfr",
        "--total-steps",
        "20",
        "--B",
        "4",
        "--H",
        "5",
        "--K",
        "7",
        "--eval-every",
        "10",
        "--eval-steps",
        "20",
        "--cfr-state-mode",
        "joint_last_action",
        "--cfr-value-lr",
        "0.1",
        "--cfr-gamma",
        "0.95",
        "--out-dir",
        str(out_dir),
    ]
    subprocess.run(cmd, check=True)
    assert (out_dir / "train_metrics.csv").exists()
    assert (out_dir / "eval_metrics.csv").exists()
    assert (out_dir / "summary.json").exists()
    summary = json.loads((out_dir / "summary.json").read_text())
    assert summary["oracle_kind"] == "tabular_multi_cfr"
    assert "final_eval_avg_price_oracle" in summary


def test_cli_smoke_tabular_lola(tmp_path):
    out_dir = tmp_path / "tabular_lola_cli"
    cmd = [
        sys.executable,
        "-m",
        "experiments.dqn_oracle_vs_qvictim",
        "--oracle-kind",
        "tabular_lola",
        "--total-steps",
        "20",
        "--B",
        "4",
        "--H",
        "5",
        "--K",
        "7",
        "--eval-every",
        "10",
        "--eval-steps",
        "20",
        "--lola-gamma",
        "0.95",
        "--lola-tau",
        "0.05",
        "--lola-epsilon",
        "0.05",
        "--out-dir",
        str(out_dir),
    ]
    subprocess.run(cmd, check=True)
    assert (out_dir / "train_metrics.csv").exists()
    assert (out_dir / "eval_metrics.csv").exists()
    assert (out_dir / "summary.json").exists()
    summary = json.loads((out_dir / "summary.json").read_text())
    assert summary["oracle_kind"] == "tabular_lola"
    assert "final_eval_avg_price_oracle" in summary


def test_cli_smoke_tabular_model_lola(tmp_path):
    out_dir = tmp_path / "tabular_model_lola_cli"
    cmd = [
        sys.executable,
        "-m",
        "experiments.dqn_oracle_vs_qvictim",
        "--oracle-kind",
        "tabular_model_lola",
        "--total-steps",
        "20",
        "--B",
        "4",
        "--H",
        "5",
        "--K",
        "7",
        "--eval-every",
        "10",
        "--eval-steps",
        "20",
        "--model-lola-gamma",
        "0.95",
        "--model-lola-tau",
        "0.05",
        "--model-lola-epsilon",
        "0.02",
        "--model-lola-victim-policy",
        "epsilon_greedy",
        "--model-lola-future-policy",
        "epsilon_greedy",
        "--out-dir",
        str(out_dir),
    ]
    subprocess.run(cmd, check=True)
    assert (out_dir / "train_metrics.csv").exists()
    assert (out_dir / "eval_metrics.csv").exists()
    assert (out_dir / "summary.json").exists()
    summary = json.loads((out_dir / "summary.json").read_text())
    assert summary["oracle_kind"] == "tabular_model_lola"
    assert "final_eval_avg_price_oracle" in summary


def test_cli_smoke_tabular_rollout_lola(tmp_path):
    out_dir = tmp_path / "tabular_rollout_lola_cli"
    cmd = [
        sys.executable,
        "-m",
        "experiments.dqn_oracle_vs_qvictim",
        "--oracle-kind",
        "tabular_rollout_lola",
        "--total-steps",
        "20",
        "--B",
        "4",
        "--H",
        "5",
        "--K",
        "7",
        "--eval-every",
        "10",
        "--eval-steps",
        "20",
        "--rollout-lola-horizon",
        "3",
        "--rollout-lola-num-particles",
        "2",
        "--rollout-lola-tau",
        "0.05",
        "--rollout-lola-epsilon",
        "0.02",
        "--rollout-lola-backend",
        "torch",
        "--out-dir",
        str(out_dir),
    ]
    subprocess.run(cmd, check=True)
    assert (out_dir / "train_metrics.csv").exists()
    assert (out_dir / "eval_metrics.csv").exists()
    assert (out_dir / "summary.json").exists()
    assert (out_dir / "progress.jsonl").exists()
    summary = json.loads((out_dir / "summary.json").read_text())
    assert summary["oracle_kind"] == "tabular_rollout_lola"
    assert "final_eval_avg_price_oracle" in summary
