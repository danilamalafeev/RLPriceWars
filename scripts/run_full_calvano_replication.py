from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from calvano_market import CalvanoMarketConfig, StaticBenchmarks, build_static_benchmarks
from calvano_qlearning import QLearningConfig, parameter_grid, run_session


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def build_experiment_cells(mode: str) -> list[dict[str, Any]]:
    if mode == "representative":
        pairs = [(0.15, 4e-6)]
    elif mode == "midpoint":
        pairs = [(0.125, 1e-5)]
    elif mode == "debug-grid":
        pairs = parameter_grid(debug=True)
    elif mode == "full-grid":
        pairs = parameter_grid(debug=False)
    else:
        raise ValueError(f"unknown mode: {mode}")

    return [
        {
            "cell_id": idx,
            "alpha": float(alpha),
            "beta": float(beta),
            "label": f"{mode}:alpha={alpha:.8g}:beta={beta:.8g}",
        }
        for idx, (alpha, beta) in enumerate(pairs)
    ]


def get_git_metadata(repo_dir: Path | None = None) -> tuple[str, bool | None]:
    cwd = repo_dir or Path.cwd()
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        commit = "unknown"

    try:
        status = subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=cwd,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        dirty: bool | None = bool(status.strip())
    except Exception:
        dirty = None
    return commit, dirty


def benchmark_metadata(benchmarks: StaticBenchmarks) -> dict[str, Any]:
    return {
        "p_n": float(benchmarks.p_n),
        "p_m": float(benchmarks.p_m),
        "pi_n_mean": float(np.mean(benchmarks.pi_n)),
        "pi_m_mean": float(np.mean(benchmarks.pi_m)),
        "discrete_nash_actions": [int(x) for x in benchmarks.nash_actions],
        "discrete_monopoly_actions": [int(x) for x in benchmarks.monopoly_actions],
        "discrete_nash_prices": [float(benchmarks.price_grid[a]) for a in benchmarks.nash_actions],
        "discrete_monopoly_prices": [float(benchmarks.price_grid[a]) for a in benchmarks.monopoly_actions],
    }


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [json_safe(v) for v in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if pd.isna(value) if not isinstance(value, (dict, list, tuple, str, bytes)) else False:
        return None
    return value


def raw_record(
    *,
    mode: str,
    cell: dict[str, Any],
    session_id: int,
    session_seed: int,
    result,
    q_config: QLearningConfig,
    market_config: CalvanoMarketConfig,
    benchmarks: StaticBenchmarks,
    started_at: str,
    completed_at: str,
    wall_time_seconds: float,
    code_version: str,
    dirty_worktree: bool | None,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "mode": mode,
        "cell_id": int(cell["cell_id"]),
        "cell_label": str(cell["label"]),
        "session": int(session_id),
        "n": int(q_config.n),
        "k": int(q_config.k),
        "m": int(q_config.m),
        "alpha": float(q_config.alpha),
        "beta": float(q_config.beta),
        "delta": float(q_config.delta),
        "mu": float(market_config.mu),
        "xi": float(market_config.xi),
        "seed": int(session_seed),
        "convergence_window": int(q_config.convergence_window),
        "max_periods": int(q_config.max_periods),
        "eval_periods": int(q_config.eval_periods),
        "converged": bool(result.converged),
        "periods_to_convergence": int(result.periods_to_convergence),
        "detected_cycle_length": int(result.detected_cycle_length),
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
        "wall_time_seconds": float(wall_time_seconds),
        "run_started_at": started_at,
        "run_completed_at": completed_at,
        "code_version": code_version,
        "dirty_worktree": dirty_worktree,
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


def _run_one_session(task: dict[str, Any]) -> dict[str, Any]:
    q_config = QLearningConfig(
        alpha=task["alpha"],
        beta=task["beta"],
        delta=task["delta"],
        n=task["n"],
        k=task["k"],
        m=task["m"],
        convergence_window=task["convergence_window"],
        max_periods=task["max_periods"],
        eval_periods=task["eval_periods"],
        seed=task["session_seed"],
    )
    market_config = task["market_config"]
    benchmarks = task["benchmarks"]
    started = utc_now_iso()
    t0 = time.perf_counter()
    result = run_session(q_config, market_config, benchmarks, seed=task["session_seed"])
    wall = time.perf_counter() - t0
    completed = utc_now_iso()
    return raw_record(
        mode=task["mode"],
        cell=task["cell"],
        session_id=task["session_id"],
        session_seed=task["session_seed"],
        result=result,
        q_config=q_config,
        market_config=market_config,
        benchmarks=benchmarks,
        started_at=started,
        completed_at=completed,
        wall_time_seconds=wall,
        code_version=task["code_version"],
        dirty_worktree=task["dirty_worktree"],
    )


def raw_filename(out_dir: Path, fmt: str) -> Path:
    return out_dir / f"raw_sessions.{fmt}"


def read_raw(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def write_raw(df: pd.DataFrame, path: Path) -> None:
    df = df.sort_values(["cell_id", "session"]).reset_index(drop=True)
    if path.suffix == ".parquet":
        df.to_parquet(path, index=False)
    else:
        df.to_csv(path, index=False)


def aggregate_results(raw_df: pd.DataFrame) -> pd.DataFrame:
    if raw_df.empty:
        return pd.DataFrame(
            columns=[
                "mode",
                "cell_id",
                "alpha",
                "beta",
                "sessions",
                "convergence_rate",
                "average_profit_gain",
                "std_profit_gain",
                "median_profit_gain",
                "q05_profit_gain",
                "q95_profit_gain",
                "average_periods_to_convergence",
                "median_periods_to_convergence",
                "average_long_run_price",
                "std_long_run_price",
                "average_long_run_profit",
                "std_long_run_profit",
                "constant_price_frequency",
                "cycle_length_2_frequency",
                "average_cycle_length",
                "share_profit_gain_positive",
                "share_profit_gain_above_0_5",
                "share_profit_gain_above_0_8",
            ]
        )

    df = raw_df.copy()
    price_cols = [c for c in df.columns if c.startswith("long_run_avg_price_")]
    profit_cols = [c for c in df.columns if c.startswith("long_run_avg_profit_")]
    df["_avg_price"] = df[price_cols].mean(axis=1) if price_cols else np.nan
    df["_avg_profit"] = df[profit_cols].mean(axis=1) if profit_cols else np.nan

    rows = []
    for keys, group in df.groupby(["mode", "cell_id", "alpha", "beta"], sort=True):
        mode, cell_id, alpha, beta = keys
        gains = group["profit_gain_mean"].astype(float)
        cycles = group["detected_cycle_length"].astype(float)
        rows.append(
            {
                "mode": mode,
                "cell_id": int(cell_id),
                "alpha": float(alpha),
                "beta": float(beta),
                "sessions": int(len(group)),
                "convergence_rate": float(group["converged"].astype(bool).mean()),
                "average_profit_gain": float(gains.mean()),
                "std_profit_gain": float(gains.std(ddof=0)),
                "median_profit_gain": float(gains.median()),
                "q05_profit_gain": float(gains.quantile(0.05)),
                "q95_profit_gain": float(gains.quantile(0.95)),
                "average_periods_to_convergence": float(group["periods_to_convergence"].mean()),
                "median_periods_to_convergence": float(group["periods_to_convergence"].median()),
                "average_long_run_price": float(group["_avg_price"].mean()),
                "std_long_run_price": float(group["_avg_price"].std(ddof=0)),
                "average_long_run_profit": float(group["_avg_profit"].mean()),
                "std_long_run_profit": float(group["_avg_profit"].std(ddof=0)),
                "constant_price_frequency": float((cycles == 1).mean()),
                "cycle_length_2_frequency": float((cycles == 2).mean()),
                "average_cycle_length": float(cycles.mean()),
                "share_profit_gain_positive": float((gains > 0.0).mean()),
                "share_profit_gain_above_0_5": float((gains > 0.5).mean()),
                "share_profit_gain_above_0_8": float((gains > 0.8).mean()),
            }
        )
    return pd.DataFrame.from_records(rows).sort_values(["cell_id"]).reset_index(drop=True)


def make_summary(
    *,
    run_config: dict[str, Any],
    raw_df: pd.DataFrame,
    aggregate_df: pd.DataFrame,
    benchmarks: StaticBenchmarks,
    total_wall_time_seconds: float,
    plot_warnings: list[str],
) -> dict[str, Any]:
    best_gain = None
    best_conv = None
    if not aggregate_df.empty:
        best_gain = aggregate_df.loc[aggregate_df["average_profit_gain"].idxmax()].to_dict()
        with_sessions = aggregate_df[aggregate_df["sessions"] > 0]
        if not with_sessions.empty:
            best_conv = with_sessions.loc[with_sessions["convergence_rate"].idxmax()].to_dict()

    return json_safe({
        "run_config": run_config,
        "number_of_cells": int(len(build_experiment_cells(run_config["mode"]))),
        "number_of_sessions_completed": int(len(raw_df)),
        "total_wall_time_seconds": float(total_wall_time_seconds),
        "overall_convergence_rate": float(raw_df["converged"].astype(bool).mean()) if len(raw_df) else None,
        "overall_average_profit_gain": float(raw_df["profit_gain_mean"].mean()) if len(raw_df) else None,
        "best_cell_by_average_profit_gain": best_gain,
        "best_cell_by_convergence_rate": best_conv,
        "benchmark_values": benchmark_metadata(benchmarks),
        "plot_warnings": plot_warnings,
    })


def _plot_heatmap(aggregate_df: pd.DataFrame, value: str, title: str, out_path: Path) -> None:
    import matplotlib.pyplot as plt

    pivot = aggregate_df.pivot(index="alpha", columns="beta", values=value).sort_index(ascending=True)
    fig, ax = plt.subplots(figsize=(7, 5))
    try:
        import seaborn as sns

        sns.heatmap(pivot, ax=ax, cmap="viridis")
    except Exception:
        image = ax.imshow(pivot.to_numpy(), aspect="auto", origin="lower", cmap="viridis")
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels([f"{x:.2g}" for x in pivot.columns], rotation=45, ha="right")
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels([f"{x:.3g}" for x in pivot.index])
        fig.colorbar(image, ax=ax)
    ax.set_title(title)
    ax.set_xlabel("beta")
    ax.set_ylabel("alpha")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def generate_plots(raw_df: pd.DataFrame, aggregate_df: pd.DataFrame, plots_dir: Path, mode: str) -> list[str]:
    warnings: list[str] = []
    plots_dir.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        return [f"matplotlib unavailable: {exc}"]

    try:
        if mode in {"debug-grid", "full-grid"} and not aggregate_df.empty:
            _plot_heatmap(aggregate_df, "average_profit_gain", "Average Profit Gain", plots_dir / "profit_gain_heatmap.png")
            _plot_heatmap(aggregate_df, "convergence_rate", "Convergence Rate", plots_dir / "convergence_rate_heatmap.png")
            _plot_heatmap(aggregate_df, "average_long_run_price", "Average Long-Run Price", plots_dir / "avg_price_heatmap.png")
    except Exception as exc:
        warnings.append(f"heatmap plotting failed: {exc}")

    try:
        price_cols = [c for c in raw_df.columns if c.startswith("long_run_avg_price_")]
        if price_cols:
            fig, ax = plt.subplots(figsize=(7, 4))
            for col in price_cols:
                ax.hist(raw_df[col].astype(float), bins=20, alpha=0.55, label=col)
            ax.set_title("Long-Run Price Distribution")
            ax.set_xlabel("price")
            ax.set_ylabel("count")
            ax.legend()
            fig.tight_layout()
            fig.savefig(plots_dir / "representative_price_distribution.png", dpi=150)
            plt.close(fig)
    except Exception as exc:
        warnings.append(f"price distribution plotting failed: {exc}")

    try:
        if "detected_cycle_length" in raw_df:
            counts = raw_df["detected_cycle_length"].value_counts().sort_index()
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.bar([str(x) for x in counts.index], counts.values)
            ax.set_title("Cycle Length Distribution")
            ax.set_xlabel("cycle length")
            ax.set_ylabel("sessions")
            fig.tight_layout()
            fig.savefig(plots_dir / "cycle_length_distribution.png", dpi=150)
            plt.close(fig)
    except Exception as exc:
        warnings.append(f"cycle distribution plotting failed: {exc}")

    try:
        if "profit_gain_mean" in raw_df:
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.hist(raw_df["profit_gain_mean"].astype(float), bins=20, alpha=0.75)
            ax.set_title("Profit Gain Distribution")
            ax.set_xlabel("profit gain")
            ax.set_ylabel("sessions")
            fig.tight_layout()
            fig.savefig(plots_dir / "profit_gain_distribution.png", dpi=150)
            plt.close(fig)
    except Exception as exc:
        warnings.append(f"profit gain distribution plotting failed: {exc}")

    return warnings


def prepare_out_dir(out_dir: Path, overwrite: bool, resume: bool) -> None:
    if out_dir.exists():
        if overwrite:
            shutil.rmtree(out_dir)
        elif not resume:
            raise FileExistsError(f"{out_dir} already exists; pass --overwrite or --resume")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "plots").mkdir(exist_ok=True)


def completed_pairs(existing: pd.DataFrame) -> set[tuple[int, int]]:
    if existing.empty:
        return set()
    return {(int(row.cell_id), int(row.session)) for row in existing[["cell_id", "session"]].itertuples(index=False)}


def run_tasks(tasks: list[dict[str, Any]], workers: int) -> list[dict[str, Any]]:
    if workers <= 1:
        return [_run_one_session(task) for task in tasks]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(_run_one_session, tasks))


def save_outputs(
    *,
    out_dir: Path,
    fmt: str,
    raw_df: pd.DataFrame,
    run_config: dict[str, Any],
    benchmarks: StaticBenchmarks,
    total_wall_time_seconds: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    aggregate_df = aggregate_results(raw_df)
    aggregate_df.to_csv(out_dir / "aggregate_by_cell.csv", index=False)
    plot_warnings = generate_plots(raw_df, aggregate_df, out_dir / "plots", run_config["mode"])
    summary = make_summary(
        run_config=run_config,
        raw_df=raw_df,
        aggregate_df=aggregate_df,
        benchmarks=benchmarks,
        total_wall_time_seconds=total_wall_time_seconds,
        plot_warnings=plot_warnings,
    )
    (out_dir / "config.json").write_text(json.dumps(run_config, indent=2, sort_keys=True), encoding="utf-8")
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    write_raw(raw_df, raw_filename(out_dir, fmt))
    return aggregate_df, summary


def run_full_replication(args: argparse.Namespace) -> dict[str, Any]:
    run_started = time.perf_counter()
    out_dir = Path(args.out_dir)
    prepare_out_dir(out_dir, overwrite=args.overwrite, resume=args.resume)
    raw_path = raw_filename(out_dir, args.format)
    existing = read_raw(raw_path) if args.resume else pd.DataFrame()
    done = completed_pairs(existing)

    cells = build_experiment_cells(args.mode)
    market_config = CalvanoMarketConfig(m=args.m, xi=args.xi, mu=args.mu)
    benchmarks = build_static_benchmarks(market_config)
    code_version, dirty_worktree = get_git_metadata(Path.cwd())
    run_config = {
        "mode": args.mode,
        "sessions": int(args.sessions),
        "workers": int(args.workers),
        "seed": int(args.seed),
        "max_periods": int(args.max_periods),
        "convergence_window": int(args.convergence_window),
        "eval_periods": int(args.eval_periods),
        "format": args.format,
        "m": int(args.m),
        "k": int(args.k),
        "delta": float(args.delta),
        "mu": float(args.mu),
        "xi": float(args.xi),
        "code_version": code_version,
        "dirty_worktree": dirty_worktree,
        "created_at": utc_now_iso(),
    }
    (out_dir / "config.json").write_text(json.dumps(run_config, indent=2, sort_keys=True), encoding="utf-8")

    raw_df = existing.copy()
    for cell in cells:
        tasks = []
        for session_id in range(args.sessions):
            pair = (int(cell["cell_id"]), session_id)
            if pair in done:
                continue
            session_seed = args.seed + int(cell["cell_id"]) * args.sessions + session_id
            tasks.append(
                {
                    "mode": args.mode,
                    "cell": cell,
                    "session_id": session_id,
                    "session_seed": session_seed,
                    "alpha": float(cell["alpha"]),
                    "beta": float(cell["beta"]),
                    "delta": float(args.delta),
                    "n": 2,
                    "k": int(args.k),
                    "m": int(args.m),
                    "convergence_window": int(args.convergence_window),
                    "max_periods": int(args.max_periods),
                    "eval_periods": int(args.eval_periods),
                    "market_config": market_config,
                    "benchmarks": benchmarks,
                    "code_version": code_version,
                    "dirty_worktree": dirty_worktree,
                }
            )

        if tasks:
            records = run_tasks(tasks, args.workers)
            batch_df = pd.DataFrame.from_records(records)
            raw_df = pd.concat([raw_df, batch_df], ignore_index=True)
            raw_df = raw_df.drop_duplicates(["cell_id", "session"], keep="last")
            write_raw(raw_df, raw_path)
            done.update((int(r["cell_id"]), int(r["session"])) for r in records)
        print(f"cell={cell['cell_id']} label={cell['label']} completed={args.sessions - sum((int(cell['cell_id']), s) not in done for s in range(args.sessions))}/{args.sessions}")

    raw_df = raw_df.sort_values(["cell_id", "session"]).reset_index(drop=True)
    total_wall = time.perf_counter() - run_started
    aggregate_df, summary = save_outputs(
        out_dir=out_dir,
        fmt=args.format,
        raw_df=raw_df,
        run_config=run_config,
        benchmarks=benchmarks,
        total_wall_time_seconds=total_wall,
    )
    print(f"wrote raw sessions: {raw_path}")
    print(f"wrote aggregate: {out_dir / 'aggregate_by_cell.csv'}")
    print(f"wrote summary: {out_dir / 'summary.json'}")
    print(f"sessions_completed={len(raw_df)} cells={len(aggregate_df)}")
    print(f"overall_average_profit_gain={summary['overall_average_profit_gain']}")
    print(f"overall_convergence_rate={summary['overall_convergence_rate']}")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run and report Calvano et al. full baseline replication experiments.")
    parser.add_argument("--mode", choices=["representative", "midpoint", "debug-grid", "full-grid"], required=True)
    parser.add_argument("--sessions", type=int, default=None)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-periods", type=int, default=1_000_000_000)
    parser.add_argument("--convergence-window", type=int, default=100_000)
    parser.add_argument("--eval-periods", type=int, default=10_000)
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--format", choices=["csv", "parquet"], default="csv")
    parser.add_argument("--m", type=int, default=15)
    parser.add_argument("--k", type=int, default=1)
    parser.add_argument("--delta", type=float, default=0.95)
    parser.add_argument("--mu", type=float, default=0.25)
    parser.add_argument("--xi", type=float, default=0.1)
    args = parser.parse_args()
    if args.overwrite and args.resume:
        raise ValueError("--overwrite and --resume are mutually exclusive")
    if args.sessions is None:
        args.sessions = 8 if args.mode == "debug-grid" else 100
    return args


def main() -> None:
    run_full_replication(parse_args())


if __name__ == "__main__":
    main()
