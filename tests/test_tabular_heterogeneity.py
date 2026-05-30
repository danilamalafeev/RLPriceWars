from __future__ import annotations

import json
from argparse import Namespace

import numpy as np
import pandas as pd

from calvano_market import CalvanoMarketConfig, build_static_benchmarks
from calvano_qlearning import QLearningConfig, agent_learning_parameters, initialize_q_tables
from scripts.run_tabular_heterogeneity import (
    aggregate_outputs,
    build_tasks,
    is_task_complete,
    planned_conditions,
    run_block2,
)


def test_agent_learning_parameters_are_per_agent():
    config = QLearningConfig(alpha=0.1, alpha_0=0.15, alpha_1=0.03, delta=0.9, delta_0=0.95, delta_1=0.70)
    alpha, beta, delta = agent_learning_parameters(config)
    np.testing.assert_allclose(alpha, np.array([0.15, 0.03]))
    np.testing.assert_allclose(beta, np.array([4e-6, 4e-6]))
    np.testing.assert_allclose(delta, np.array([0.95, 0.70]))


def test_heterogeneous_delta_initializes_each_agent_separately():
    market_config = CalvanoMarketConfig(m=4)
    benchmarks = build_static_benchmarks(market_config)
    q = initialize_q_tables(benchmarks.profit_matrix, delta=np.array([0.95, 0.70]), n=2, k=1, m=4)
    for a0 in range(4):
        expected0 = np.mean(benchmarks.profit_matrix[a0, :, 0]) / (1.0 - 0.95)
        np.testing.assert_allclose(q[0, :, a0], expected0)
    for a1 in range(4):
        expected1 = np.mean(benchmarks.profit_matrix[:, a1, 1]) / (1.0 - 0.70)
        np.testing.assert_allclose(q[1, :, a1], expected1)


def test_block2_task_construction_four_conditions_times_seeds(tmp_path):
    tasks = build_tasks(tmp_path, [0, 1, 2])
    assert len(planned_conditions()) == 4
    assert len(tasks) == 12
    assert {task.condition.name for task in tasks} == {
        "alpha_o0.15_v0.03",
        "alpha_o0.03_v0.15",
        "delta_o0.95_v0.70",
        "delta_o0.70_v0.95",
    }
    assert tasks[0].out_dir == tmp_path / "alpha_o0.15_v0.03" / "seed_0"


def test_block2_resume_skip_behavior(tmp_path):
    tasks = build_tasks(tmp_path, [0])
    task = tasks[0]
    task.out_dir.mkdir(parents=True)
    (task.out_dir / "summary.json").write_text("{}", encoding="utf-8")
    assert is_task_complete(task)

    args = Namespace(
        root=str(tmp_path),
        seeds="0",
        max_periods=5,
        eval_periods=5,
        m=4,
        resume=True,
        dry_run=True,
    )
    manifest = run_block2(args)
    first = [record for record in manifest["tasks"] if record["condition"] == task.condition.name][0]
    assert first["status"] == "skipped_completed"


def write_fake_summary(path, condition: str, seed: int, profit: float) -> None:
    path.mkdir(parents=True)
    payload = {
        "condition": condition,
        "seed": seed,
        "final_avg_profit_oracle": profit,
        "final_avg_profit_victim": profit / 2,
        "final_market_price_mean": 1.5 + profit,
        "final_profit_asymmetry": profit / 2,
        "distance_to_nash_price": 0.2,
        "periods_to_convergence": 100,
        "detected_cycle_length": 1,
    }
    (path / "summary.json").write_text(json.dumps(payload), encoding="utf-8")


def test_block2_aggregation_on_fake_outputs(tmp_path):
    tasks = build_tasks(tmp_path, [0, 1])
    condition = tasks[0].condition.name
    same_condition_tasks = [task for task in tasks if task.condition.name == condition]
    write_fake_summary(same_condition_tasks[0].out_dir, condition, 0, 0.3)
    write_fake_summary(same_condition_tasks[1].out_dir, condition, 1, 0.5)

    info = aggregate_outputs(tasks, tmp_path)

    summary = pd.read_csv(tmp_path / "summary_by_seed.csv")
    aggregate = pd.read_csv(tmp_path / "aggregate_by_condition.csv")
    assert info["completed_summaries"] == 2
    assert len(summary) == 2
    assert len(aggregate) == 1
    assert aggregate.loc[0, "condition"] == condition
    assert aggregate.loc[0, "completed_seeds"] == 2
    assert aggregate.loc[0, "final_avg_profit_oracle_mean"] == 0.4
