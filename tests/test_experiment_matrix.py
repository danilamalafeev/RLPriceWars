from __future__ import annotations

import json
import subprocess
import sys
from argparse import Namespace

import pandas as pd

from scripts.run_experiment_matrix import (
    aggregate_completed,
    build_matrix_tasks,
    parse_int_list,
    plan_task_records,
    run_matrix,
)


def test_parse_int_list_ranges_and_csv():
    assert parse_int_list("0-2,5,7-6") == [0, 1, 2, 5, 7, 6]


def test_build_block3_tasks_default_shape(tmp_path):
    tasks = build_matrix_tasks(tmp_path, ["block3"], [0, 1])
    assert len(tasks) == 10
    modes = [task.mode for task in tasks]
    assert modes[:2] == ["actor_critic", "actor_critic"]
    actor = tasks[0]
    assert actor.block == "block3_architectures_150k"
    assert actor.task_class == "neural"
    assert actor.out_dir == tmp_path / "block3_architectures_150k" / "actor_critic" / "seed_0"
    assert "--reservoir-dim" in actor.command
    assert "512" in actor.command
    assert "--total-steps" in actor.command
    assert "150000" in actor.command

    tabular = [task for task in tasks if task.mode == "tabular_cfr"][0]
    assert tabular.task_class == "tabular"
    assert "--reservoir-dim" not in tabular.command


def test_build_rollout_tasks_horizons(tmp_path):
    tasks = build_matrix_tasks(tmp_path, ["block4_rollout"], [0], rollout_device="cuda", rollout_backend="torch")
    assert len(tasks) == 3
    assert [task.horizon for task in tasks] == [5, 12, 25]
    assert all(task.mode == "tabular_rollout_lola" for task in tasks)
    assert all(task.task_class == "rollout" for task in tasks)
    assert tasks[1].out_dir == tmp_path / "block4_rollout_lola_150k" / "horizon_12" / "seed_0"
    assert "--rollout-lola-num-particles" in tasks[1].command
    assert "32" in tasks[1].command
    assert "--device" in tasks[1].command
    assert "cuda" in tasks[1].command
    assert "--rollout-lola-backend" in tasks[1].command
    assert "torch" in tasks[1].command


def test_build_static_victim_tasks(tmp_path):
    tasks = build_matrix_tasks(tmp_path, ["block1_static_victim"], [0, 1])
    assert len(tasks) == 4
    assert {task.mode for task in tasks} == {"dqn", "tabular_cfr"}
    assert all(task.block == "block1_static_victim_100k" for task in tasks)
    assert all("--victim-kind" in task.command for task in tasks)
    assert all("static_cooperative" in task.command for task in tasks)
    assert all("100000" in task.command for task in tasks)
    assert tasks[0].out_dir == tmp_path / "block1_static_victim_100k" / "dqn" / "seed_0"


def test_resume_skips_completed_summary(tmp_path):
    tasks = build_matrix_tasks(tmp_path, ["block3"], [0])
    completed = tasks[0]
    completed.out_dir.mkdir(parents=True)
    (completed.out_dir / "summary.json").write_text("{}", encoding="utf-8")

    pending, records = plan_task_records(tasks, resume=True, running_commands=[])

    assert completed not in pending
    completed_record = [record for record in records if record["mode"] == completed.mode][0]
    assert completed_record["status"] == "skipped_completed"


def write_fake_summary(path, mode: str, seed: int, profit: float, horizon: int | None = None) -> None:
    path.mkdir(parents=True)
    payload = {
        "oracle_kind": mode,
        "seed": seed,
        "final_eval_avg_profit_oracle": profit,
        "final_eval_avg_profit_victim": profit / 2,
        "final_eval_market_price_mean": 1.5 + profit,
        "max_eval_profit_asymmetry": profit / 3,
        "mean_last_5_eval_profit_asymmetry": profit / 4,
    }
    if horizon is not None:
        payload["rollout_lola_horizon"] = horizon
    (path / "summary.json").write_text(json.dumps(payload), encoding="utf-8")


def test_aggregate_completed_writes_block_outputs(tmp_path):
    tasks = build_matrix_tasks(tmp_path, ["block3"], [0, 1])
    write_fake_summary(tasks[0].out_dir, tasks[0].mode, 0, 0.3)
    write_fake_summary(tasks[1].out_dir, tasks[1].mode, 1, 0.5)

    info = aggregate_completed(tasks, tmp_path)

    block_dir = tmp_path / "block3_architectures_150k"
    summary = pd.read_csv(block_dir / "summary_by_seed.csv")
    aggregate = pd.read_csv(block_dir / "aggregate_by_mode.csv")
    assert info["completed_summaries"] == 2
    assert len(summary) == 2
    assert len(aggregate) == 1
    assert aggregate.loc[0, "mode"] == "actor_critic"
    assert aggregate.loc[0, "completed_seeds"] == 2
    assert aggregate.loc[0, "final_eval_avg_profit_oracle_mean"] == 0.4


def test_dry_run_run_matrix_does_not_create_outputs(tmp_path, capsys):
    args = Namespace(
        root=str(tmp_path),
        blocks="block3",
        seeds="0-1",
        resume=True,
        dry_run=True,
        max_neural=2,
        max_tabular=6,
        max_rollout=1,
        rollout_device="cpu",
        rollout_backend="numpy",
    )

    result = run_matrix(args)
    output = capsys.readouterr().out

    assert result["dry_run"] is True
    assert result["pending_count"] == 10
    assert "experiments.dqn_oracle_vs_qvictim" in output
    assert not (tmp_path / "run_manifest.json").exists()


def test_dry_run_cli_smoke(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.run_experiment_matrix",
            "--root",
            str(tmp_path),
            "--blocks",
            "block3",
            "--seeds",
            "0-1",
            "--dry-run",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "experiments.dqn_oracle_vs_qvictim" in result.stdout
    assert "--total-steps 150000" in result.stdout
    assert not (tmp_path / "run_manifest.json").exists()


def test_dry_run_cli_rollout_backend_device_flags(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.run_experiment_matrix",
            "--root",
            str(tmp_path),
            "--blocks",
            "block4_rollout",
            "--seeds",
            "0",
            "--rollout-device",
            "cuda",
            "--rollout-backend",
            "torch",
            "--dry-run",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "--device cuda" in result.stdout
    assert "--rollout-lola-backend torch" in result.stdout
    assert not (tmp_path / "run_manifest.json").exists()
