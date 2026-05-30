from __future__ import annotations

import argparse
import json
import platform
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from calvano_market import CalvanoMarketConfig, build_static_benchmarks
from calvano_qlearning import QLearningConfig, run_session


DEFAULT_ROOT = Path("results/long_matrix_100k_plus/block2_tabular_heterogeneity_100k")
DEFAULT_SEEDS = range(10)
DEFAULT_MAX_PERIODS = 100_000
DEFAULT_EVAL_PERIODS = 10_000


@dataclass(frozen=True)
class HeterogeneityCondition:
    name: str
    alpha_0: float = 0.15
    alpha_1: float = 0.15
    delta_0: float = 0.95
    delta_1: float = 0.95
    beta_0: float = 4e-6
    beta_1: float = 4e-6


@dataclass(frozen=True)
class HeterogeneityTask:
    condition: HeterogeneityCondition
    seed: int
    out_dir: Path


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_int_list(value: str) -> list[int]:
    values: list[int] = []
    for part in [p.strip() for p in value.split(",") if p.strip()]:
        if "-" in part:
            start_text, end_text = part.split("-", 1)
            start = int(start_text)
            end = int(end_text)
            step = 1 if end >= start else -1
            values.extend(range(start, end + step, step))
        else:
            values.append(int(part))
    return list(dict.fromkeys(values))


def planned_conditions() -> list[HeterogeneityCondition]:
    return [
        HeterogeneityCondition(name="alpha_o0.15_v0.03", alpha_0=0.15, alpha_1=0.03),
        HeterogeneityCondition(name="alpha_o0.03_v0.15", alpha_0=0.03, alpha_1=0.15),
        HeterogeneityCondition(name="delta_o0.95_v0.70", delta_0=0.95, delta_1=0.70),
        HeterogeneityCondition(name="delta_o0.70_v0.95", delta_0=0.70, delta_1=0.95),
    ]


def build_tasks(root: Path, seeds: Iterable[int]) -> list[HeterogeneityTask]:
    return [
        HeterogeneityTask(
            condition=condition,
            seed=int(seed),
            out_dir=root / condition.name / f"seed_{int(seed)}",
        )
        for condition in planned_conditions()
        for seed in seeds
    ]


def is_task_complete(task: HeterogeneityTask) -> bool:
    return (task.out_dir / "summary.json").exists()


def q_config_for_task(task: HeterogeneityTask, max_periods: int, eval_periods: int, m: int) -> QLearningConfig:
    c = task.condition
    return QLearningConfig(
        alpha_0=c.alpha_0,
        alpha_1=c.alpha_1,
        beta_0=c.beta_0,
        beta_1=c.beta_1,
        delta_0=c.delta_0,
        delta_1=c.delta_1,
        m=m,
        max_periods=max_periods,
        convergence_window=max_periods,
        eval_periods=eval_periods,
        seed=task.seed,
    )


def summary_from_result(
    task: HeterogeneityTask,
    q_config: QLearningConfig,
    market_config: CalvanoMarketConfig,
    benchmarks,
    result,
) -> dict[str, Any]:
    profit = result.long_run_avg_profit
    price = result.long_run_avg_price
    return {
        "condition": task.condition.name,
        "seed": task.seed,
        "alpha_0": float(q_config.alpha_0 if q_config.alpha_0 is not None else q_config.alpha),
        "alpha_1": float(q_config.alpha_1 if q_config.alpha_1 is not None else q_config.alpha),
        "beta_0": float(q_config.beta_0 if q_config.beta_0 is not None else q_config.beta),
        "beta_1": float(q_config.beta_1 if q_config.beta_1 is not None else q_config.beta),
        "delta_0": float(q_config.delta_0 if q_config.delta_0 is not None else q_config.delta),
        "delta_1": float(q_config.delta_1 if q_config.delta_1 is not None else q_config.delta),
        "m": int(q_config.m),
        "max_periods": int(q_config.max_periods),
        "eval_periods": int(q_config.eval_periods),
        "converged": bool(result.converged),
        "periods_to_convergence": int(result.periods_to_convergence),
        "detected_cycle_length": int(result.detected_cycle_length),
        "final_avg_profit_oracle": float(profit[0]),
        "final_avg_profit_victim": float(profit[1]),
        "final_profit_asymmetry": float(profit[0] - profit[1]),
        "final_avg_price_oracle": float(price[0]),
        "final_avg_price_victim": float(price[1]),
        "final_market_price_mean": float(price.mean()),
        "final_profit_gain_oracle": float(result.profit_gain_delta[0]),
        "final_profit_gain_victim": float(result.profit_gain_delta[1]),
        "distance_to_nash_price": float(abs(price.mean() - benchmarks.p_n)),
        "distance_to_monopoly_price": float(abs(price.mean() - benchmarks.p_m)),
        "benchmarks": {
            "p_n": float(benchmarks.p_n),
            "p_m": float(benchmarks.p_m),
            "pi_n": [float(x) for x in benchmarks.pi_n],
            "pi_m": [float(x) for x in benchmarks.pi_m],
        },
        "market_config": asdict(market_config),
    }


def run_task(task: HeterogeneityTask, max_periods: int, eval_periods: int, m: int) -> dict[str, Any]:
    market_config = CalvanoMarketConfig(m=m)
    benchmarks = build_static_benchmarks(market_config)
    q_config = q_config_for_task(task, max_periods=max_periods, eval_periods=eval_periods, m=m)
    result = run_session(q_config, market_config, benchmarks=benchmarks, seed=task.seed)
    summary = summary_from_result(task, q_config, market_config, benchmarks, result)

    task.out_dir.mkdir(parents=True, exist_ok=True)
    (task.out_dir / "config.json").write_text(
        json.dumps(
            {
                "condition": asdict(task.condition),
                "q_config": asdict(q_config),
                "market_config": asdict(market_config),
                "seed": task.seed,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    (task.out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def read_summary(task: HeterogeneityTask) -> dict[str, Any] | None:
    path = task.out_dir / "summary.json"
    if not path.exists():
        return None
    try:
        row = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    row["out_dir"] = str(task.out_dir)
    return row


def aggregate_outputs(tasks: Iterable[HeterogeneityTask], root: Path) -> dict[str, Any]:
    rows = [row for task in tasks if (row := read_summary(task)) is not None]
    root.mkdir(parents=True, exist_ok=True)
    if not rows:
        return {"completed_summaries": 0}

    summary_df = pd.DataFrame.from_records(rows)
    summary_df.sort_values(["condition", "seed"]).to_csv(root / "summary_by_seed.csv", index=False)
    value_cols = [
        col
        for col in summary_df.select_dtypes(include="number").columns
        if col not in {"seed"} and col.startswith(("final_", "distance_", "periods_", "detected_"))
    ]
    aggregate = summary_df.groupby("condition", dropna=False)[value_cols].agg(["mean", "std", "min", "max"])
    aggregate.columns = ["_".join(str(part) for part in col if part) for col in aggregate.columns]
    aggregate = aggregate.reset_index()
    aggregate["completed_seeds"] = summary_df.groupby("condition", dropna=False)["seed"].count().to_numpy()
    aggregate.to_csv(root / "aggregate_by_condition.csv", index=False)
    return {"completed_summaries": int(len(summary_df))}


def write_manifest(root: Path, records: list[dict[str, Any]], aggregate_info: dict[str, Any]) -> dict[str, Any]:
    manifest = {
        "timestamp": utc_now_iso(),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "aggregate_info": aggregate_info,
        "tasks": records,
    }
    root.mkdir(parents=True, exist_ok=True)
    (root / "run_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest


def run_block2(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.root)
    seeds = parse_int_list(args.seeds)
    tasks = build_tasks(root, seeds)
    records: list[dict[str, Any]] = []

    for task in tasks:
        base = {
            "condition": task.condition.name,
            "seed": task.seed,
            "out_dir": str(task.out_dir),
            "summary_path": str(task.out_dir / "summary.json"),
        }
        if args.resume and is_task_complete(task):
            records.append({**base, "status": "skipped_completed"})
            continue
        if args.dry_run:
            records.append({**base, "status": "pending"})
            print(f"{task.condition.name} seed={task.seed} out_dir={task.out_dir}")
            continue

        started = datetime.now(timezone.utc)
        try:
            run_task(task, max_periods=args.max_periods, eval_periods=args.eval_periods, m=args.m)
            status = "success"
        except Exception as exc:
            status = "failed"
            records.append({**base, "status": status, "error": repr(exc)})
            raise
        finally:
            completed = datetime.now(timezone.utc)
        records.append(
            {
                **base,
                "status": status,
                "started_at": started.isoformat(),
                "completed_at": completed.isoformat(),
                "elapsed_seconds": (completed - started).total_seconds(),
            }
        )

    aggregate_info = aggregate_outputs(tasks, root) if not args.dry_run else {"completed_summaries": 0}
    manifest = write_manifest(root, records, aggregate_info) if not args.dry_run else {"tasks": records, "dry_run": True}
    if not args.dry_run:
        print(f"block2 tabular heterogeneity completed; see {root / 'run_manifest.json'}")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Block 2 tabular Q-learning heterogeneity controls.")
    parser.add_argument("--root", type=str, default=str(DEFAULT_ROOT))
    parser.add_argument("--seeds", type=str, default="0-9")
    parser.add_argument("--max-periods", type=int, default=DEFAULT_MAX_PERIODS)
    parser.add_argument("--eval-periods", type=int, default=DEFAULT_EVAL_PERIODS)
    parser.add_argument("--m", type=int, default=15)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    run_block2(parse_args())


if __name__ == "__main__":
    main()
