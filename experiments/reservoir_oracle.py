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
from neural.functional_policies import (
    init_linear_policy,
    init_linear_value,
    init_mlp_policy,
    init_mlp_value,
    linear_policy_forward,
    linear_value_forward,
    mlp_policy_forward,
    mlp_value_forward,
)
from neural.losses import train_ac_step_dual_lr, train_pg_step_dual_lr
from neural.observations import ObservationConfig, build_observation, observation_dim
from neural.reservoir import ReservoirConfig, init_reservoir_buffers, reservoir_observation, reservoir_update
from neural.rollout import RolloutConfig, collect_duopoly_rollout, sample_actions


@dataclass(frozen=True)
class ReservoirExperimentConfig:
    scenario: str
    seed: int = 0
    B: int = 64
    T: int = 32
    H: int = 8
    K: int = 15
    updates: int = 1000
    eval_every: int = 50
    eval_episodes: int = 4
    gamma: float = 0.95
    lr_oracle: float = 1e-2
    lr_victim: float = 1e-2
    entropy_coef: float = 0.01
    entropy_coef_start: float | None = None
    entropy_coef_end: float | None = None
    entropy_anneal_steps: int = 0
    training_mode: str = "reinforce"
    value_hidden_dim: int = 64
    value_coef: float = 0.5
    lr_value_oracle: float = 1e-2
    lr_value_victim: float = 1e-2
    hidden_dim: int = 64
    reservoir_dim_oracle: int = 128
    reservoir_dim_victim: int = 128
    reservoir_spectral_radius: float = 0.9
    reservoir_leak_rate: float = 0.5
    device: str = "cpu"
    out_dir: str | None = None


def entropy_coef_at_update(config: ReservoirExperimentConfig, update: int) -> float:
    if config.entropy_coef_start is None or config.entropy_coef_end is None or config.entropy_anneal_steps <= 0:
        return float(config.entropy_coef)
    progress = min(float(update) / float(config.entropy_anneal_steps), 1.0)
    return float(config.entropy_coef_start + progress * (config.entropy_coef_end - config.entropy_coef_start))


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
    return env, price_grid, benchmarks


def build_scenario(config: ReservoirExperimentConfig, obs_dim: int, K: int, generator: torch.Generator):
    device = torch.device(config.device)
    oracle_buffers: dict[str, Any] = {}
    victim_buffers: dict[str, Any] = {}
    use_oracle_reservoir = False
    use_victim_reservoir = False
    oracle_policy_fn = mlp_policy_forward
    victim_policy_fn = mlp_policy_forward
    oracle_value_fn = mlp_value_forward
    victim_value_fn = mlp_value_forward
    oracle_input_dim = obs_dim
    victim_input_dim = obs_dim

    if config.scenario == "mlp_vs_mlp":
        pass
    elif config.scenario == "reservoir_oracle_vs_mlp":
        use_oracle_reservoir = True
        oracle_input_dim = obs_dim + config.reservoir_dim_oracle
    elif config.scenario == "reservoir_vs_reservoir":
        use_oracle_reservoir = True
        use_victim_reservoir = True
        oracle_input_dim = obs_dim + config.reservoir_dim_oracle
        victim_input_dim = obs_dim + config.reservoir_dim_victim
    elif config.scenario == "reservoir_oracle_vs_linear":
        use_oracle_reservoir = True
        oracle_input_dim = obs_dim + config.reservoir_dim_oracle
        victim_policy_fn = linear_policy_forward
        victim_value_fn = linear_value_forward
    else:
        raise ValueError(f"unknown scenario: {config.scenario}")

    if use_oracle_reservoir:
        oracle_buffers["reservoir"] = init_reservoir_buffers(
            generator,
            ReservoirConfig(
                input_dim=obs_dim,
                reservoir_dim=config.reservoir_dim_oracle,
                spectral_radius=config.reservoir_spectral_radius,
                leak_rate=config.reservoir_leak_rate,
                device=device,
            ),
        )
    if use_victim_reservoir:
        victim_buffers["reservoir"] = init_reservoir_buffers(
            generator,
            ReservoirConfig(
                input_dim=obs_dim,
                reservoir_dim=config.reservoir_dim_victim,
                spectral_radius=config.reservoir_spectral_radius,
                leak_rate=config.reservoir_leak_rate,
                device=device,
            ),
        )

    oracle_params = init_mlp_policy(generator, oracle_input_dim, config.hidden_dim, K, device=device)
    oracle_value_params = init_mlp_value(generator, oracle_input_dim, config.value_hidden_dim, device=device)
    if victim_policy_fn is linear_policy_forward:
        victim_params = init_linear_policy(generator, victim_input_dim, K, device=device)
        victim_value_params = init_linear_value(generator, victim_input_dim, device=device)
    else:
        victim_params = init_mlp_policy(generator, victim_input_dim, config.hidden_dim, K, device=device)
        victim_value_params = init_mlp_value(generator, victim_input_dim, config.value_hidden_dim, device=device)

    rollout_config = RolloutConfig(
        T=config.T,
        B=config.B,
        H=config.H,
        K=K,
        use_oracle_reservoir=use_oracle_reservoir,
        use_victim_reservoir=use_victim_reservoir,
        device=device,
    )
    obs_config = ObservationConfig(device=device)
    return (
        oracle_policy_fn,
        oracle_params,
        oracle_buffers,
        victim_policy_fn,
        victim_params,
        victim_buffers,
        oracle_value_fn,
        oracle_value_params,
        victim_value_fn,
        victim_value_params,
        rollout_config,
        obs_config,
    )


def _market_features(env, obs_config: ObservationConfig, t: int) -> torch.Tensor:
    return build_observation(
        cm.get_price_history_view(env),
        cm.get_current_prices(env),
        cm.get_rewards(env),
        cm.get_market_share(env),
        cm.get_outside_share(env),
        cm.get_margins(env),
        obs_config,
        time_step=t,
    )


def evaluate_neural_policies(
    config: ReservoirExperimentConfig,
    oracle_policy_fn,
    oracle_params,
    oracle_buffers,
    victim_policy_fn,
    victim_params,
    victim_buffers,
    obs_config: ObservationConfig,
    rollout_config: RolloutConfig,
    benchmarks: StaticBenchmarks,
    greedy: bool = True,
    seed_offset: int = 10_000,
) -> dict[str, float]:
    device = torch.device(config.device)
    reward_sum = torch.zeros(2, dtype=torch.float64)
    price_sum = torch.zeros(2, dtype=torch.float64)
    count = 0
    generator = torch.Generator(device=device).manual_seed(config.seed + seed_offset)

    use_oracle_reservoir = rollout_config.use_reservoir or rollout_config.use_oracle_reservoir
    use_victim_reservoir = rollout_config.use_reservoir or rollout_config.use_victim_reservoir

    with torch.no_grad():
        for ep in range(config.eval_episodes):
            env, price_grid, _ = make_calvano_vec_env(config.B, config.H, config.K, config.seed + seed_offset + ep)
            obs_cfg = ObservationConfig(
                price_min=float(price_grid[0]),
                price_max=float(price_grid[-1]),
                device=device,
            )
            oracle_state = None
            victim_state = None
            oracle_h = None
            victim_h = None

            for t in range(config.T):
                features = _market_features(env, obs_cfg, t).to(device)
                oracle_obs = features
                victim_obs = features

                if use_oracle_reservoir:
                    if oracle_h is None:
                        R = oracle_buffers["reservoir"]["W_res"].shape[0]
                        oracle_h = torch.zeros(config.B, R, dtype=torch.float32, device=device)
                    oracle_h = reservoir_update(features, oracle_h, oracle_buffers["reservoir"])
                    oracle_obs = reservoir_observation(features, oracle_h)
                if use_victim_reservoir:
                    if victim_h is None:
                        R = victim_buffers["reservoir"]["W_res"].shape[0]
                        victim_h = torch.zeros(config.B, R, dtype=torch.float32, device=device)
                    victim_h = reservoir_update(features, victim_h, victim_buffers["reservoir"])
                    victim_obs = reservoir_observation(features, victim_h)

                logits_o, oracle_state = oracle_policy_fn(oracle_params, oracle_buffers, oracle_obs, oracle_state)
                logits_v, victim_state = victim_policy_fn(victim_params, victim_buffers, victim_obs, victim_state)
                if greedy:
                    action_o = torch.argmax(logits_o, dim=1)
                    action_v = torch.argmax(logits_v, dim=1)
                else:
                    action_o, _, _ = sample_actions(logits_o, generator)
                    action_v, _, _ = sample_actions(logits_v, generator)
                actions = torch.stack([action_o.detach(), action_v.detach()], dim=1)
                cm.step(env, actions.cpu().numpy().astype(np.int64, copy=False))
                rewards = torch.as_tensor(np.asarray(cm.get_rewards(env)), dtype=torch.float64)
                prices = torch.as_tensor(np.asarray(cm.get_current_prices(env)), dtype=torch.float64)
                reward_sum += rewards.sum(dim=0)
                price_sum += prices.sum(dim=0)
                count += config.B

    avg_profit = reward_sum / max(count, 1)
    avg_price = price_sum / max(count, 1)
    pi_n = torch.as_tensor(benchmarks.pi_n, dtype=torch.float64)
    pi_m = torch.as_tensor(benchmarks.pi_m, dtype=torch.float64)
    denom = torch.clamp(torch.abs(pi_m - pi_n), min=1e-12)
    gains = (avg_profit - pi_n) / denom
    market_price_mean = float(avg_price.mean().item())

    return {
        "eval_avg_profit_oracle": float(avg_profit[0].item()),
        "eval_avg_profit_victim": float(avg_profit[1].item()),
        "eval_profit_asymmetry": float((avg_profit[0] - avg_profit[1]).item()),
        "eval_avg_price_oracle": float(avg_price[0].item()),
        "eval_avg_price_victim": float(avg_price[1].item()),
        "eval_market_price_mean": market_price_mean,
        "eval_distance_to_nash_price": float(abs(market_price_mean - benchmarks.p_n)),
        "eval_distance_to_monopoly_price": float(abs(market_price_mean - benchmarks.p_m)),
        "eval_oracle_profit_gain": float(gains[0].item()),
        "eval_victim_profit_gain": float(gains[1].item()),
        "eval_asymmetry_index": float((avg_profit[0] - avg_profit[1]).item()),
    }


def _plot_outputs(train_df: pd.DataFrame, eval_df: pd.DataFrame, out_dir: Path) -> list[str]:
    warnings: list[str] = []
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        return [f"matplotlib unavailable: {exc}"]

    specs = [
        ("profit_timeseries.png", train_df, ["avg_reward_oracle", "avg_reward_victim"], "Average Training Profit"),
        ("price_timeseries.png", train_df, ["avg_price_oracle", "avg_price_victim"], "Average Training Price"),
        ("asymmetry_timeseries.png", train_df, ["profit_asymmetry", "price_asymmetry"], "Training Asymmetry"),
        ("eval_profit_gain.png", eval_df, ["eval_oracle_profit_gain", "eval_victim_profit_gain"], "Evaluation Profit Gain"),
    ]
    for filename, df, cols, title in specs:
        try:
            if df.empty:
                continue
            fig, ax = plt.subplots(figsize=(7, 4))
            x = df["update"] if "update" in df else np.arange(len(df))
            for col in cols:
                if col in df:
                    ax.plot(x, df[col], label=col)
            ax.set_title(title)
            ax.set_xlabel("update")
            ax.legend()
            fig.tight_layout()
            fig.savefig(plots_dir / filename, dpi=150)
            plt.close(fig)
        except Exception as exc:
            warnings.append(f"{filename} failed: {exc}")
    return warnings


def _summary(config: ReservoirExperimentConfig, eval_df: pd.DataFrame, benchmarks: StaticBenchmarks, plot_warnings: list[str]) -> dict[str, Any]:
    final = eval_df.iloc[-1] if not eval_df.empty else {}
    asym = eval_df["eval_profit_asymmetry"] if "eval_profit_asymmetry" in eval_df else pd.Series(dtype=float)
    return {
        "scenario": config.scenario,
        "seed": config.seed,
        "training_mode": config.training_mode,
        "final_eval_avg_profit_oracle": None if eval_df.empty else float(final["eval_avg_profit_oracle"]),
        "final_eval_avg_profit_victim": None if eval_df.empty else float(final["eval_avg_profit_victim"]),
        "final_eval_profit_asymmetry": None if eval_df.empty else float(final["eval_profit_asymmetry"]),
        "final_eval_oracle_profit_gain": None if eval_df.empty else float(final["eval_oracle_profit_gain"]),
        "final_eval_victim_profit_gain": None if eval_df.empty else float(final["eval_victim_profit_gain"]),
        "max_eval_profit_asymmetry": None if eval_df.empty else float(asym.max()),
        "mean_last_10_eval_profit_asymmetry": None if eval_df.empty else float(asym.tail(10).mean()),
        "benchmarks": {
            "p_n": float(benchmarks.p_n),
            "p_m": float(benchmarks.p_m),
            "pi_n": [float(x) for x in benchmarks.pi_n],
            "pi_m": [float(x) for x in benchmarks.pi_m],
        },
        "plot_warnings": plot_warnings,
    }


def _config_for_depth(base_config: ReservoirExperimentConfig, depth: int, seed: int, out_dir: Path) -> ReservoirExperimentConfig:
    values = asdict(base_config)
    values["seed"] = int(seed)
    values["out_dir"] = str(out_dir)
    if depth <= 0:
        values["scenario"] = "mlp_vs_mlp"
        values["reservoir_dim_oracle"] = max(1, int(base_config.reservoir_dim_oracle))
    else:
        values["scenario"] = "reservoir_oracle_vs_mlp"
        values["reservoir_dim_oracle"] = int(depth)
    return ReservoirExperimentConfig(**values)


def build_depth_sweep_tasks(base_config: ReservoirExperimentConfig, depths: list[int], seeds: list[int], base_out_dir: str | Path) -> list[dict[str, Any]]:
    root = Path(base_out_dir)
    tasks = []
    for depth in depths:
        for seed in seeds:
            out_dir = root / f"depth_{int(depth)}" / f"seed_{int(seed)}"
            tasks.append({"depth": int(depth), "seed": int(seed), "out_dir": out_dir})
    return tasks


def _plot_depth_sweep(raw_df: pd.DataFrame, base_out_dir: Path) -> list[str]:
    warnings: list[str] = []
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        return [f"matplotlib unavailable: {exc}"]

    specs = [
        ("depth_vs_oracle_profit.png", "final_eval_avg_profit_oracle", "Oracle Profit"),
        ("depth_vs_asymmetry.png", "final_eval_profit_asymmetry", "Profit Asymmetry"),
        ("depth_vs_profit_gain.png", "final_eval_oracle_profit_gain", "Oracle Profit Gain"),
    ]
    for filename, col, title in specs:
        try:
            fig, ax = plt.subplots(figsize=(7, 4))
            grouped = raw_df.groupby("depth", sort=True)[col]
            means = grouped.mean()
            stds = grouped.std(ddof=0).fillna(0.0)
            ax.errorbar(means.index.to_numpy(), means.to_numpy(), yerr=stds.to_numpy(), marker="o")
            ax.set_title(title)
            ax.set_xlabel("reservoir depth")
            fig.tight_layout()
            fig.savefig(base_out_dir / filename, dpi=150)
            plt.close(fig)
        except Exception as exc:
            warnings.append(f"{filename} failed: {exc}")
    return warnings


def aggregate_depth_sweep(raw_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for depth, group in raw_df.groupby("depth", sort=True):
        asym = group["final_eval_profit_asymmetry"].astype(float)
        rows.append(
            {
                "depth": int(depth),
                "seeds": int(group["seed"].nunique()),
                "oracle_profit_mean": float(group["final_eval_avg_profit_oracle"].mean()),
                "oracle_profit_std": float(group["final_eval_avg_profit_oracle"].std(ddof=0)),
                "victim_profit_mean": float(group["final_eval_avg_profit_victim"].mean()),
                "victim_profit_std": float(group["final_eval_avg_profit_victim"].std(ddof=0)),
                "asymmetry_mean": float(asym.mean()),
                "asymmetry_std": float(asym.std(ddof=0)),
                "oracle_gain_mean": float(group["final_eval_oracle_profit_gain"].mean()),
                "oracle_gain_std": float(group["final_eval_oracle_profit_gain"].std(ddof=0)),
                "positive_asymmetry_share": float((asym > 0.0).mean()),
                "max_asymmetry_mean": float(group["max_eval_profit_asymmetry"].mean()),
                "final_entropy_oracle_mean": float(group["final_entropy_oracle"].mean()),
                "final_entropy_victim_mean": float(group["final_entropy_victim"].mean()),
            }
        )
    return pd.DataFrame.from_records(rows)


def run_reservoir_depth_sweep(base_config: ReservoirExperimentConfig, depths: list[int], seeds: list[int], base_out_dir: str | Path) -> dict[str, Any]:
    root = Path(base_out_dir)
    root.mkdir(parents=True, exist_ok=True)
    raw_rows = []
    for task in build_depth_sweep_tasks(base_config, depths, seeds, root):
        cfg = _config_for_depth(base_config, task["depth"], task["seed"], task["out_dir"])
        result = run_reservoir_experiment(cfg)
        summary = result["summary"]
        train_df = result["train_metrics"]
        final_train = train_df.iloc[-1] if not train_df.empty else {}
        raw_rows.append(
            {
                "depth": task["depth"],
                "seed": task["seed"],
                "scenario": summary["scenario"],
                "training_mode": summary["training_mode"],
                "final_eval_avg_profit_oracle": summary["final_eval_avg_profit_oracle"],
                "final_eval_avg_profit_victim": summary["final_eval_avg_profit_victim"],
                "final_eval_profit_asymmetry": summary["final_eval_profit_asymmetry"],
                "final_eval_oracle_profit_gain": summary["final_eval_oracle_profit_gain"],
                "final_eval_victim_profit_gain": summary["final_eval_victim_profit_gain"],
                "max_eval_profit_asymmetry": summary["max_eval_profit_asymmetry"],
                "final_entropy_oracle": float(final_train.get("entropy_oracle", np.nan)),
                "final_entropy_victim": float(final_train.get("entropy_victim", np.nan)),
                "out_dir": str(task["out_dir"]),
            }
        )
    raw_df = pd.DataFrame.from_records(raw_rows).sort_values(["depth", "seed"]).reset_index(drop=True)
    agg_df = aggregate_depth_sweep(raw_df)
    raw_df.to_csv(root / "depth_sweep_raw.csv", index=False)
    agg_df.to_csv(root / "depth_sweep_aggregate.csv", index=False)
    warnings = _plot_depth_sweep(raw_df, root)
    (root / "depth_sweep_summary.json").write_text(
        json.dumps(
            {
                "depths": [int(x) for x in depths],
                "seeds": [int(x) for x in seeds],
                "runs": int(len(raw_df)),
                "plot_warnings": warnings,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return {"raw": raw_df, "aggregate": agg_df, "plot_warnings": warnings}


def run_reservoir_experiment(config: ReservoirExperimentConfig) -> dict[str, Any]:
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    device = torch.device(config.device)
    generator = torch.Generator(device=device).manual_seed(config.seed)

    env, price_grid, benchmarks = make_calvano_vec_env(config.B, config.H, config.K, config.seed)
    base_obs_dim = observation_dim(config.H)
    (
        oracle_policy_fn,
        oracle_params,
        oracle_buffers,
        victim_policy_fn,
        victim_params,
        victim_buffers,
        oracle_value_fn,
        oracle_value_params,
        victim_value_fn,
        victim_value_params,
        rollout_config,
        obs_config,
    ) = build_scenario(config, base_obs_dim, config.K, generator)
    obs_config = ObservationConfig(price_min=float(price_grid[0]), price_max=float(price_grid[-1]), device=device)

    train_rows = []
    eval_rows = []
    for update in range(1, config.updates + 1):
        rollout = collect_duopoly_rollout(
            env,
            oracle_policy_fn,
            oracle_params,
            oracle_buffers,
            victim_policy_fn,
            victim_params,
            victim_buffers,
            obs_config,
            rollout_config,
            generator,
            oracle_value_fn=oracle_value_fn if config.training_mode == "actor_critic" else None,
            oracle_value_params=oracle_value_params if config.training_mode == "actor_critic" else None,
            victim_value_fn=victim_value_fn if config.training_mode == "actor_critic" else None,
            victim_value_params=victim_value_params if config.training_mode == "actor_critic" else None,
        )
        entropy_coef = entropy_coef_at_update(config, update)
        if config.training_mode == "actor_critic":
            metrics = train_ac_step_dual_lr(
                oracle_params,
                victim_params,
                oracle_value_params,
                victim_value_params,
                rollout,
                gamma=config.gamma,
                lr_policy_oracle=config.lr_oracle,
                lr_policy_victim=config.lr_victim,
                lr_value_oracle=config.lr_value_oracle,
                lr_value_victim=config.lr_value_victim,
                entropy_coef=entropy_coef,
                value_coef=config.value_coef,
            )
        elif config.training_mode == "reinforce":
            metrics = train_pg_step_dual_lr(
                oracle_params,
                victim_params,
                rollout,
                gamma=config.gamma,
                lr_oracle=config.lr_oracle,
                lr_victim=config.lr_victim,
                entropy_coef=entropy_coef,
            )
        else:
            raise ValueError(f"unknown training_mode: {config.training_mode}")
        avg_rewards = rollout["rewards"].mean(dim=(0, 1))
        avg_prices = rollout["prices"].mean(dim=(0, 1))
        avg_entropy = rollout["entropy"].detach().mean(dim=(0, 1))
        train_rows.append(
            {
                "update": update,
                "avg_reward_oracle": float(avg_rewards[0].item()),
                "avg_reward_victim": float(avg_rewards[1].item()),
                "avg_price_oracle": float(avg_prices[0].item()),
                "avg_price_victim": float(avg_prices[1].item()),
                "profit_asymmetry": float((avg_rewards[0] - avg_rewards[1]).item()),
                "price_asymmetry": float((avg_prices[0] - avg_prices[1]).item()),
                "loss_oracle": metrics["loss_oracle"],
                "loss_victim": metrics["loss_victim"],
                "entropy_oracle": float(avg_entropy[0].item()),
                "entropy_victim": float(avg_entropy[1].item()),
                "entropy_coef": entropy_coef,
                "value_loss_oracle": metrics.get("value_loss_oracle", 0.0),
                "value_loss_victim": metrics.get("value_loss_victim", 0.0),
                "explained_variance_oracle": metrics.get("explained_variance_oracle", 0.0),
                "explained_variance_victim": metrics.get("explained_variance_victim", 0.0),
            }
        )
        if update % max(config.eval_every, 1) == 0 or update == config.updates:
            row = {"update": update}
            row.update(
                evaluate_neural_policies(
                    config,
                    oracle_policy_fn,
                    oracle_params,
                    oracle_buffers,
                    victim_policy_fn,
                    victim_params,
                    victim_buffers,
                    obs_config,
                    rollout_config,
                    benchmarks,
                    greedy=True,
                )
            )
            eval_rows.append(row)

    train_df = pd.DataFrame.from_records(train_rows)
    eval_df = pd.DataFrame.from_records(eval_rows)
    plot_warnings: list[str] = []
    summary = _summary(config, eval_df, benchmarks, plot_warnings)

    if config.out_dir is not None:
        out_dir = Path(config.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "config.json").write_text(json.dumps(asdict(config), indent=2, sort_keys=True), encoding="utf-8")
        train_df.to_csv(out_dir / "train_metrics.csv", index=False)
        eval_df.to_csv(out_dir / "eval_metrics.csv", index=False)
        plot_warnings = _plot_outputs(train_df, eval_df, out_dir)
        summary = _summary(config, eval_df, benchmarks, plot_warnings)
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    return {
        "config": config,
        "train_metrics": train_df,
        "eval_metrics": eval_df,
        "summary": summary,
        "benchmarks": benchmarks,
        "price_grid": price_grid,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Reservoir Oracle vs Victim neural duopoly experiment.")
    parser.add_argument("--scenario", default="reservoir_oracle_vs_mlp", choices=["mlp_vs_mlp", "reservoir_oracle_vs_mlp", "reservoir_vs_reservoir", "reservoir_oracle_vs_linear"])
    parser.add_argument("--depth-sweep", action="store_true")
    parser.add_argument("--reservoir-depths", type=str, default="0,16,32,64,128,256")
    parser.add_argument("--seeds", type=str, default="0")
    parser.add_argument("--base-out-dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--B", type=int, default=64)
    parser.add_argument("--T", type=int, default=32)
    parser.add_argument("--H", type=int, default=8)
    parser.add_argument("--K", type=int, default=15)
    parser.add_argument("--updates", type=int, default=1000)
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--eval-episodes", type=int, default=4)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--lr-oracle", type=float, default=1e-2)
    parser.add_argument("--lr-victim", type=float, default=1e-2)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--entropy-coef-start", type=float, default=None)
    parser.add_argument("--entropy-coef-end", type=float, default=None)
    parser.add_argument("--entropy-anneal-steps", type=int, default=0)
    parser.add_argument("--training-mode", choices=["reinforce", "actor_critic"], default="reinforce")
    parser.add_argument("--value-hidden-dim", type=int, default=64)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--lr-value-oracle", type=float, default=1e-2)
    parser.add_argument("--lr-value-victim", type=float, default=1e-2)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--reservoir-dim-oracle", type=int, default=128)
    parser.add_argument("--reservoir-dim-victim", type=int, default=128)
    parser.add_argument("--reservoir-spectral-radius", type=float, default=0.9)
    parser.add_argument("--reservoir-leak-rate", type=float, default=0.5)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--out-dir", type=str, default=None)
    return parser.parse_args()


def _parse_int_list(text: str) -> list[int]:
    return [int(part.strip()) for part in text.split(",") if part.strip()]


def main() -> None:
    args = parse_args()
    config = ReservoirExperimentConfig(
        scenario=args.scenario,
        seed=args.seed,
        B=args.B,
        T=args.T,
        H=args.H,
        K=args.K,
        updates=args.updates,
        eval_every=args.eval_every,
        eval_episodes=args.eval_episodes,
        gamma=args.gamma,
        lr_oracle=args.lr_oracle,
        lr_victim=args.lr_victim,
        entropy_coef=args.entropy_coef,
        entropy_coef_start=args.entropy_coef_start,
        entropy_coef_end=args.entropy_coef_end,
        entropy_anneal_steps=args.entropy_anneal_steps,
        training_mode=args.training_mode,
        value_hidden_dim=args.value_hidden_dim,
        value_coef=args.value_coef,
        lr_value_oracle=args.lr_value_oracle,
        lr_value_victim=args.lr_value_victim,
        hidden_dim=args.hidden_dim,
        reservoir_dim_oracle=args.reservoir_dim_oracle,
        reservoir_dim_victim=args.reservoir_dim_victim,
        reservoir_spectral_radius=args.reservoir_spectral_radius,
        reservoir_leak_rate=args.reservoir_leak_rate,
        device=args.device,
        out_dir=args.out_dir,
    )
    if args.depth_sweep:
        if args.base_out_dir is None:
            raise ValueError("--base-out-dir is required with --depth-sweep")
        result = run_reservoir_depth_sweep(config, _parse_int_list(args.reservoir_depths), _parse_int_list(args.seeds), args.base_out_dir)
        print(result["aggregate"].to_string(index=False))
        return

    result = run_reservoir_experiment(config)
    summary = result["summary"]
    print(f"scenario={summary['scenario']} seed={summary['seed']}")
    print(f"final_eval_avg_profit_oracle={summary['final_eval_avg_profit_oracle']}")
    print(f"final_eval_avg_profit_victim={summary['final_eval_avg_profit_victim']}")
    print(f"final_eval_profit_asymmetry={summary['final_eval_profit_asymmetry']}")


if __name__ == "__main__":
    main()
