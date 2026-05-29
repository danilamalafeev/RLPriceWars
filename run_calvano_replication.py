from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from calvano_market import CalvanoMarketConfig, build_static_benchmarks
from calvano_qlearning import QLearningConfig, parameter_grid, run_sessions, summarize_sessions


def session_record(
    result,
    cell_id: int,
    session_id: int,
    session_seed: int,
    q_config: QLearningConfig,
    market_config: CalvanoMarketConfig,
    benchmarks,
) -> dict[str, float | int | bool]:
    record: dict[str, float | int | bool] = {
        "cell_id": cell_id,
        "session": session_id,
        "n": q_config.n,
        "k": q_config.k,
        "m": q_config.m,
        "alpha": q_config.alpha,
        "beta": q_config.beta,
        "delta": q_config.delta,
        "mu": market_config.mu,
        "xi": market_config.xi,
        "seed": session_seed,
        "convergence_window": q_config.convergence_window,
        "max_periods": q_config.max_periods,
        "eval_periods": q_config.eval_periods,
        "converged": result.converged,
        "periods_to_convergence": result.periods_to_convergence,
        "detected_cycle_length": result.detected_cycle_length,
        "profit_gain_mean": float(np.mean(result.profit_gain_delta)),
        "p_n": float(benchmarks.p_n),
        "p_m": float(benchmarks.p_m),
        "discrete_nash_action_0": int(benchmarks.nash_actions[0]),
        "discrete_nash_action_1": int(benchmarks.nash_actions[1]),
        "discrete_monopoly_action_0": int(benchmarks.monopoly_actions[0]),
        "discrete_monopoly_action_1": int(benchmarks.monopoly_actions[1]),
        "discrete_nash_price_0": float(benchmarks.price_grid[benchmarks.nash_actions[0]]),
        "discrete_nash_price_1": float(benchmarks.price_grid[benchmarks.nash_actions[1]]),
        "discrete_monopoly_price_0": float(benchmarks.price_grid[benchmarks.monopoly_actions[0]]),
        "discrete_monopoly_price_1": float(benchmarks.price_grid[benchmarks.monopoly_actions[1]]),
        "pi_n_0": float(benchmarks.pi_n[0]),
        "pi_n_1": float(benchmarks.pi_n[1]),
        "pi_m_0": float(benchmarks.pi_m[0]),
        "pi_m_1": float(benchmarks.pi_m[1]),
    }
    for i, value in enumerate(result.long_run_avg_price):
        record[f"long_run_avg_price_{i}"] = float(value)
    for i, value in enumerate(result.long_run_avg_profit):
        record[f"long_run_avg_profit_{i}"] = float(value)
    for i, value in enumerate(result.profit_gain_delta):
        record[f"profit_gain_delta_{i}"] = float(value)
    for i, value in enumerate(result.last_prices):
        record[f"last_price_{i}"] = float(value)
    return record


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Calvano et al. tabular independent Q-learning replication.")
    parser.add_argument("--sessions", type=int, default=10)
    parser.add_argument("--alpha", type=float, default=0.15)
    parser.add_argument("--beta", type=float, default=4e-6)
    parser.add_argument("--grid-debug", action="store_true", help="Run a small 5x5 alpha/beta grid instead of one cell.")
    parser.add_argument("--representative-check", action="store_true", help="Run the representative alpha=0.15, beta=4e-6, m=15 cell and print full summary.")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=str, default="results/calvano_replication.csv")
    parser.add_argument("--max-periods", type=int, default=1_000_000_000)
    parser.add_argument("--convergence-window", type=int, default=100_000)
    parser.add_argument("--eval-periods", type=int, default=10_000)
    parser.add_argument("--delta", type=float, default=0.95)
    parser.add_argument("--m", type=int, default=15)
    parser.add_argument("--k", type=int, default=1)
    parser.add_argument("--mu", type=float, default=0.25)
    parser.add_argument("--xi", type=float, default=0.1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.representative_check:
        args.alpha = 0.15
        args.beta = 4e-6
        args.m = 15

    market_config = CalvanoMarketConfig(m=args.m, xi=args.xi, mu=args.mu)
    benchmarks = build_static_benchmarks(market_config)
    cells = parameter_grid(debug=True) if args.grid_debug and not args.representative_check else [(args.alpha, args.beta)]

    records = []
    for cell_id, (alpha, beta) in enumerate(cells):
        cell_seed = args.seed + cell_id * args.sessions
        q_config = QLearningConfig(
            alpha=alpha,
            beta=beta,
            delta=args.delta,
            k=args.k,
            m=args.m,
            convergence_window=args.convergence_window,
            max_periods=args.max_periods,
            eval_periods=args.eval_periods,
            seed=cell_seed,
        )
        results = run_sessions(args.sessions, q_config, market_config, benchmarks, workers=args.workers)
        summary = summarize_sessions(results, benchmarks)
        print(
            f"cell={cell_id} alpha={alpha:.6g} beta={beta:.6g} "
            f"gain={summary['average_profit_gain']:.4f} "
            f"conv_rate={summary['convergence_rate']:.3f}"
        )
        if args.representative_check:
            for key in [
                "average_profit_gain",
                "std_profit_gain",
                "convergence_rate",
                "average_convergence_periods",
                "average_long_run_price",
                "distance_to_nash_price",
                "distance_to_monopoly_price",
                "constant_price_frequency",
                "cycle_length_2_frequency",
            ]:
                print(f"{key}={summary[key]:.6g}")
        for session_id, result in enumerate(results):
            session_seed = cell_seed + session_id
            records.append(session_record(result, cell_id, session_id, session_seed, q_config, market_config, benchmarks))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame.from_records(records).sort_values(["cell_id", "session"]).reset_index(drop=True)
    if out_path.suffix.lower() == ".parquet":
        df.to_parquet(out_path, index=False)
    elif out_path.suffix.lower() == ".csv":
        df.to_csv(out_path, index=False)
    else:
        raise ValueError("--out must end with .csv or .parquet")
    print(f"wrote {len(df)} rows to {out_path}")


if __name__ == "__main__":
    main()
