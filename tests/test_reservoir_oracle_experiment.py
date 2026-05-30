from __future__ import annotations

import json
import subprocess
import sys

import numpy as np
import torch

from experiments.reservoir_oracle import (
    ReservoirExperimentConfig,
    build_depth_sweep_tasks,
    build_scenario,
    entropy_coef_at_update,
    evaluate_neural_policies,
    make_calvano_vec_env,
    run_reservoir_depth_sweep,
    run_reservoir_experiment,
)
from neural.functional_policies import init_linear_value, init_mlp_value, linear_value_forward, mlp_value_forward
from neural.losses import actor_critic_loss, train_ac_step_dual_lr, train_pg_step_dual_lr
from neural.observations import ObservationConfig, observation_dim
from neural.rollout import collect_duopoly_rollout


def tiny_config(scenario: str, **overrides) -> ReservoirExperimentConfig:
    values = {
        "scenario": scenario,
        "seed": 0,
        "B": 4,
        "T": 4,
        "H": 5,
        "K": 7,
        "updates": 2,
        "eval_every": 1,
        "eval_episodes": 1,
        "hidden_dim": 8,
        "value_hidden_dim": 8,
        "reservoir_dim_oracle": 6,
        "reservoir_dim_victim": 6,
        "entropy_coef": 0.01,
    }
    values.update(overrides)
    return ReservoirExperimentConfig(**values)


def unpack_scenario(scenario):
    return {
        "oracle_policy_fn": scenario[0],
        "oracle_params": scenario[1],
        "oracle_buffers": scenario[2],
        "victim_policy_fn": scenario[3],
        "victim_params": scenario[4],
        "victim_buffers": scenario[5],
        "oracle_value_fn": scenario[6],
        "oracle_value_params": scenario[7],
        "victim_value_fn": scenario[8],
        "victim_value_params": scenario[9],
        "rollout_config": scenario[10],
        "obs_config": scenario[11],
    }


def test_scenario_build_shapes():
    B, H, K = 3, 5, 7
    obs_dim = observation_dim(H)
    for scenario_name in ["mlp_vs_mlp", "reservoir_oracle_vs_mlp", "reservoir_vs_reservoir", "reservoir_oracle_vs_linear"]:
        config = tiny_config(scenario_name, B=B, H=H, K=K)
        gen = torch.Generator().manual_seed(10)
        s = unpack_scenario(build_scenario(config, obs_dim, K, gen))
        oracle_dim = obs_dim + config.reservoir_dim_oracle if s["rollout_config"].use_oracle_reservoir else obs_dim
        victim_dim = obs_dim + config.reservoir_dim_victim if s["rollout_config"].use_victim_reservoir else obs_dim
        logits_o, _ = s["oracle_policy_fn"](s["oracle_params"], s["oracle_buffers"], torch.zeros(B, oracle_dim), None)
        logits_v, _ = s["victim_policy_fn"](s["victim_params"], s["victim_buffers"], torch.zeros(B, victim_dim), None)
        assert logits_o.shape == (B, K)
        assert logits_v.shape == (B, K)


def test_rollout_per_agent_reservoir_flags():
    config = tiny_config("reservoir_oracle_vs_mlp")
    env, price_grid, _ = make_calvano_vec_env(config.B, config.H, config.K, config.seed)
    gen = torch.Generator().manual_seed(11)
    s = unpack_scenario(build_scenario(config, observation_dim(config.H), config.K, gen))
    rollout = collect_duopoly_rollout(
        env,
        s["oracle_policy_fn"],
        s["oracle_params"],
        s["oracle_buffers"],
        s["victim_policy_fn"],
        s["victim_params"],
        s["victim_buffers"],
        ObservationConfig(price_min=float(price_grid[0]), price_max=float(price_grid[-1])),
        s["rollout_config"],
        gen,
    )
    assert rollout["obs_oracle"].shape[-1] != rollout["obs_victim"].shape[-1]
    assert rollout["logp"].shape == (config.T, config.B, 2)


def test_dual_lr_update():
    config = tiny_config("mlp_vs_mlp")
    env, price_grid, _ = make_calvano_vec_env(config.B, config.H, config.K, config.seed)
    gen = torch.Generator().manual_seed(12)
    s = unpack_scenario(build_scenario(config, observation_dim(config.H), config.K, gen))
    oracle_before = {k: v.detach().clone() for k, v in s["oracle_params"].items()}
    victim_before = {k: v.detach().clone() for k, v in s["victim_params"].items()}
    rollout = collect_duopoly_rollout(
        env,
        s["oracle_policy_fn"],
        s["oracle_params"],
        s["oracle_buffers"],
        s["victim_policy_fn"],
        s["victim_params"],
        s["victim_buffers"],
        ObservationConfig(price_min=float(price_grid[0]), price_max=float(price_grid[-1])),
        s["rollout_config"],
        gen,
    )
    train_pg_step_dual_lr(s["oracle_params"], s["victim_params"], rollout, gamma=0.95, lr_oracle=0.05, lr_victim=0.0, entropy_coef=0.0)
    assert any(not torch.allclose(oracle_before[k], s["oracle_params"][k]) for k in s["oracle_params"])
    assert all(torch.allclose(victim_before[k], s["victim_params"][k]) for k in s["victim_params"])


def test_eval_metrics():
    config = tiny_config("reservoir_oracle_vs_mlp")
    _, price_grid, benchmarks = make_calvano_vec_env(config.B, config.H, config.K, config.seed)
    gen = torch.Generator().manual_seed(13)
    s = unpack_scenario(build_scenario(config, observation_dim(config.H), config.K, gen))
    metrics = evaluate_neural_policies(
        config,
        s["oracle_policy_fn"],
        s["oracle_params"],
        s["oracle_buffers"],
        s["victim_policy_fn"],
        s["victim_params"],
        s["victim_buffers"],
        ObservationConfig(price_min=float(price_grid[0]), price_max=float(price_grid[-1])),
        s["rollout_config"],
        benchmarks,
    )
    expected = {
        "eval_avg_profit_oracle",
        "eval_avg_profit_victim",
        "eval_profit_asymmetry",
        "eval_avg_price_oracle",
        "eval_avg_price_victim",
        "eval_market_price_mean",
        "eval_distance_to_nash_price",
        "eval_distance_to_monopoly_price",
        "eval_oracle_profit_gain",
        "eval_victim_profit_gain",
        "eval_asymmetry_index",
    }
    assert expected.issubset(metrics)
    assert all(np.isfinite(metrics[k]) for k in expected)


def test_run_reservoir_experiment_smoke():
    result = run_reservoir_experiment(tiny_config("reservoir_oracle_vs_mlp"))
    train_df = result["train_metrics"]
    eval_df = result["eval_metrics"]
    assert len(train_df) == 2
    assert len(eval_df) == 2
    assert np.isfinite(train_df.select_dtypes(include=[float, int]).to_numpy()).all()
    assert np.isfinite(eval_df.select_dtypes(include=[float, int]).to_numpy()).all()


def test_cli_smoke(tmp_path):
    out_dir = tmp_path / "reservoir_cli"
    cmd = [
        sys.executable,
        "-m",
        "experiments.reservoir_oracle",
        "--scenario",
        "reservoir_oracle_vs_mlp",
        "--updates",
        "1",
        "--B",
        "4",
        "--T",
        "4",
        "--H",
        "5",
        "--K",
        "7",
        "--eval-every",
        "1",
        "--eval-episodes",
        "1",
        "--hidden-dim",
        "8",
        "--reservoir-dim-oracle",
        "6",
        "--out-dir",
        str(out_dir),
    ]
    subprocess.run(cmd, check=True)
    assert (out_dir / "train_metrics.csv").exists()
    assert (out_dir / "eval_metrics.csv").exists()
    assert (out_dir / "summary.json").exists()
    summary = json.loads((out_dir / "summary.json").read_text())
    assert summary["scenario"] == "reservoir_oracle_vs_mlp"


def test_entropy_schedule():
    constant = tiny_config("mlp_vs_mlp", entropy_coef=0.03)
    assert entropy_coef_at_update(constant, 100) == 0.03
    annealed = tiny_config(
        "mlp_vs_mlp",
        entropy_coef=0.03,
        entropy_coef_start=0.02,
        entropy_coef_end=0.0,
        entropy_anneal_steps=10,
    )
    assert entropy_coef_at_update(annealed, 0) == 0.02
    np.testing.assert_allclose(entropy_coef_at_update(annealed, 5), 0.01)
    assert entropy_coef_at_update(annealed, 20) == 0.0


def test_value_head_shapes():
    B, Z = 5, 11
    gen = torch.Generator().manual_seed(21)
    obs = torch.randn(B, Z)
    linear = init_linear_value(gen, Z)
    mlp = init_mlp_value(gen, Z, hidden_dim=7)
    assert linear_value_forward(linear, obs).shape == (B,)
    assert mlp_value_forward(mlp, obs).shape == (B,)


def test_actor_critic_loss_backward():
    B, T, Z, K = 3, 4, 5, 6
    gen = torch.Generator().manual_seed(22)
    Wp = torch.randn(Z, K, generator=gen, requires_grad=True)
    Wv = torch.randn(Z, 1, generator=gen, requires_grad=True)
    obs = torch.randn(T, B, Z)
    logits = obs @ Wp
    dist = torch.distributions.Categorical(logits=logits)
    logp = dist.log_prob(dist.sample())
    values = (obs @ Wv).squeeze(-1)
    returns = torch.randn(T, B)
    loss_parts = actor_critic_loss(logp, values, returns)
    loss_parts["loss"].backward()
    assert Wp.grad is not None and torch.any(torch.abs(Wp.grad) > 0)
    assert Wv.grad is not None and torch.any(torch.abs(Wv.grad) > 0)


def test_rollout_with_values():
    config = tiny_config("reservoir_oracle_vs_mlp")
    env, price_grid, _ = make_calvano_vec_env(config.B, config.H, config.K, config.seed)
    gen = torch.Generator().manual_seed(23)
    s = unpack_scenario(build_scenario(config, observation_dim(config.H), config.K, gen))
    rollout = collect_duopoly_rollout(
        env,
        s["oracle_policy_fn"],
        s["oracle_params"],
        s["oracle_buffers"],
        s["victim_policy_fn"],
        s["victim_params"],
        s["victim_buffers"],
        ObservationConfig(price_min=float(price_grid[0]), price_max=float(price_grid[-1])),
        s["rollout_config"],
        gen,
        oracle_value_fn=s["oracle_value_fn"],
        oracle_value_params=s["oracle_value_params"],
        victim_value_fn=s["victim_value_fn"],
        victim_value_params=s["victim_value_params"],
    )
    assert rollout["values"].shape == (config.T, config.B, 2)
    assert rollout["values"].requires_grad


def test_train_ac_step_changes_policy_and_value_params():
    config = tiny_config("mlp_vs_mlp")
    env, price_grid, _ = make_calvano_vec_env(config.B, config.H, config.K, config.seed)
    gen = torch.Generator().manual_seed(24)
    s = unpack_scenario(build_scenario(config, observation_dim(config.H), config.K, gen))
    before_policy = {k: v.detach().clone() for k, v in s["oracle_params"].items()}
    before_value = {k: v.detach().clone() for k, v in s["oracle_value_params"].items()}
    rollout = collect_duopoly_rollout(
        env,
        s["oracle_policy_fn"],
        s["oracle_params"],
        s["oracle_buffers"],
        s["victim_policy_fn"],
        s["victim_params"],
        s["victim_buffers"],
        ObservationConfig(price_min=float(price_grid[0]), price_max=float(price_grid[-1])),
        s["rollout_config"],
        gen,
        oracle_value_fn=s["oracle_value_fn"],
        oracle_value_params=s["oracle_value_params"],
        victim_value_fn=s["victim_value_fn"],
        victim_value_params=s["victim_value_params"],
    )
    train_ac_step_dual_lr(
        s["oracle_params"],
        s["victim_params"],
        s["oracle_value_params"],
        s["victim_value_params"],
        rollout,
        gamma=0.95,
        lr_policy_oracle=0.05,
        lr_policy_victim=0.05,
        lr_value_oracle=0.05,
        lr_value_victim=0.05,
    )
    assert any(not torch.allclose(before_policy[k], s["oracle_params"][k]) for k in s["oracle_params"])
    assert any(not torch.allclose(before_value[k], s["oracle_value_params"][k]) for k in s["oracle_value_params"])


def test_depth_sweep_config(tmp_path):
    config = tiny_config("reservoir_oracle_vs_mlp")
    tasks = build_depth_sweep_tasks(config, [0, 16], [0, 1], tmp_path)
    assert len(tasks) == 4
    assert tasks[0]["out_dir"].parts[-2:] == ("depth_0", "seed_0")
    assert tasks[-1]["out_dir"].parts[-2:] == ("depth_16", "seed_1")


def test_depth_sweep_smoke(tmp_path):
    config = tiny_config("reservoir_oracle_vs_mlp", updates=1, B=4, T=4, eval_every=1, eval_episodes=1)
    result = run_reservoir_depth_sweep(config, [0, 4], [0], tmp_path)
    assert (tmp_path / "depth_sweep_raw.csv").exists()
    assert (tmp_path / "depth_sweep_aggregate.csv").exists()
    assert len(result["raw"]) == 2
    assert len(result["aggregate"]) == 2
