from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


BLOCK3_MODES = ["actor_critic", "dqn", "dqn_jepa", "dqn_regret", "tabular_cfr"]
STATIC_VICTIM_MODES = ["dqn", "tabular_cfr"]
ROLLOUT_HORIZONS = [5, 12, 25]
VALID_BLOCKS = {"block1_static_victim", "block3", "block4_rollout"}
NEURAL_MODES = {"actor_critic", "dqn", "dqn_jepa", "dqn_regret"}
TABULAR_MODES = {"tabular_cfr"}
ROLLOUT_MODES = {"tabular_rollout_lola"}

DEFAULT_ROOT = Path("results/long_matrix_100k_plus")
TOTAL_STEPS = 150_000
STATIC_VICTIM_STEPS = 100_000
EVAL_EVERY = 5_000
EVAL_STEPS = 2_000
ROLLOUT_PARTICLES = 32


@dataclass(frozen=True)
class MatrixTask:
    block: str
    task_class: str
    mode: str
    seed: int
    out_dir: Path
    command: list[str]
    horizon: int | None = None


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_csv(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def parse_int_list(value: str) -> list[int]:
    values: list[int] = []
    for part in parse_csv(value):
        if "-" in part:
            start_text, end_text = part.split("-", 1)
            start = int(start_text)
            end = int(end_text)
            step = 1 if end >= start else -1
            values.extend(range(start, end + step, step))
        else:
            values.append(int(part))
    return list(dict.fromkeys(values))


def validate_blocks(blocks: Iterable[str]) -> list[str]:
    selected = list(blocks)
    unknown = sorted(set(selected) - VALID_BLOCKS)
    if unknown:
        raise ValueError(f"unknown matrix block(s): {', '.join(unknown)}")
    return selected


def task_class_for_mode(mode: str) -> str:
    if mode in NEURAL_MODES:
        return "neural"
    if mode in TABULAR_MODES:
        return "tabular"
    if mode in ROLLOUT_MODES:
        return "rollout"
    raise ValueError(f"unknown oracle mode: {mode}")


def command_for_task(
    mode: str,
    seed: int,
    out_dir: Path,
    horizon: int | None = None,
    rollout_device: str = "cpu",
    rollout_backend: str = "numpy",
) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "experiments.dqn_oracle_vs_qvictim",
        "--oracle-kind",
        mode,
        "--seed",
        str(seed),
        "--total-steps",
        str(TOTAL_STEPS),
        "--eval-every",
        str(EVAL_EVERY),
        "--eval-steps",
        str(EVAL_STEPS),
        "--out-dir",
        str(out_dir),
    ]
    if mode == "actor_critic":
        command.extend(["--reservoir-dim", "512"])
    if mode == "tabular_rollout_lola":
        if horizon is None:
            raise ValueError("tabular_rollout_lola requires a horizon")
        command.extend(
            [
                "--rollout-lola-horizon",
                str(horizon),
                "--rollout-lola-num-particles",
                str(ROLLOUT_PARTICLES),
                "--device",
                rollout_device,
                "--rollout-lola-backend",
                rollout_backend,
            ]
        )
    return command


def static_victim_command_for_task(mode: str, seed: int, out_dir: Path) -> list[str]:
    command = command_for_task(mode, seed, out_dir)
    total_idx = command.index("--total-steps") + 1
    command[total_idx] = str(STATIC_VICTIM_STEPS)
    command.extend(["--victim-kind", "static_cooperative"])
    return command


def build_matrix_tasks(
    root: Path,
    blocks: Iterable[str],
    seeds: Iterable[int],
    rollout_device: str = "cpu",
    rollout_backend: str = "numpy",
) -> list[MatrixTask]:
    selected_blocks = validate_blocks(blocks)
    selected_seeds = list(seeds)
    tasks: list[MatrixTask] = []

    if "block1_static_victim" in selected_blocks:
        block_dir = root / "block1_static_victim_100k"
        for mode in STATIC_VICTIM_MODES:
            for seed in selected_seeds:
                out_dir = block_dir / mode / f"seed_{seed}"
                tasks.append(
                    MatrixTask(
                        block="block1_static_victim_100k",
                        task_class=task_class_for_mode(mode),
                        mode=mode,
                        seed=seed,
                        out_dir=out_dir,
                        command=static_victim_command_for_task(mode, seed, out_dir),
                    )
                )

    if "block3" in selected_blocks:
        block_dir = root / "block3_architectures_150k"
        for mode in BLOCK3_MODES:
            for seed in selected_seeds:
                out_dir = block_dir / mode / f"seed_{seed}"
                tasks.append(
                    MatrixTask(
                        block="block3_architectures_150k",
                        task_class=task_class_for_mode(mode),
                        mode=mode,
                        seed=seed,
                        out_dir=out_dir,
                        command=command_for_task(mode, seed, out_dir),
                    )
                )

    if "block4_rollout" in selected_blocks:
        block_dir = root / "block4_rollout_lola_150k"
        mode = "tabular_rollout_lola"
        for horizon in ROLLOUT_HORIZONS:
            for seed in selected_seeds:
                out_dir = block_dir / f"horizon_{horizon}" / f"seed_{seed}"
                tasks.append(
                    MatrixTask(
                        block="block4_rollout_lola_150k",
                        task_class=task_class_for_mode(mode),
                        mode=mode,
                        seed=seed,
                        horizon=horizon,
                        out_dir=out_dir,
                        command=command_for_task(
                            mode,
                            seed,
                            out_dir,
                            horizon=horizon,
                            rollout_device=rollout_device,
                            rollout_backend=rollout_backend,
                        ),
                    )
                )

    return tasks


def task_env(task: MatrixTask) -> dict[str, str]:
    env = os.environ.copy()
    if task.task_class == "neural":
        threads = "2"
    else:
        threads = "1"
    env["OMP_NUM_THREADS"] = threads
    env["MKL_NUM_THREADS"] = threads
    return env


def get_git_commit(repo_dir: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_dir,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return "unknown"


def get_torch_version() -> str | None:
    try:
        import torch

        return str(torch.__version__)
    except Exception:
        return None


def active_command_lines() -> list[str]:
    if os.name == "nt":
        commands = [
            ["powershell", "-NoProfile", "-Command", "Get-CimInstance Win32_Process -Filter \"name = 'python.exe'\" | ForEach-Object { $_.CommandLine }"],
            ["wmic", "process", "where", "name='python.exe'", "get", "CommandLine", "/value"],
        ]
    else:
        commands = [["ps", "-eo", "args="]]

    for command in commands:
        try:
            output = subprocess.check_output(command, stderr=subprocess.DEVNULL, text=True, timeout=10)
        except Exception:
            continue
        return [line.strip() for line in output.splitlines() if line.strip()]
    return []


def is_task_running(task: MatrixTask, command_lines: Iterable[str]) -> bool:
    out_dir_candidates = {
        str(task.out_dir),
        str(task.out_dir.resolve()),
        str(task.out_dir).replace("/", "\\"),
        str(task.out_dir).replace("\\", "/"),
    }
    for command_line in command_lines:
        if "experiments.dqn_oracle_vs_qvictim" not in command_line:
            continue
        if any(candidate and candidate in command_line for candidate in out_dir_candidates):
            return True
    return False


def base_manifest_record(task: MatrixTask, status: str) -> dict[str, Any]:
    return {
        "block": task.block,
        "task_class": task.task_class,
        "mode": task.mode,
        "seed": task.seed,
        "horizon": task.horizon,
        "out_dir": str(task.out_dir),
        "summary_path": str(task.out_dir / "summary.json"),
        "log_path": str(task.out_dir / "logs" / "run.log"),
        "command": task.command,
        "status": status,
        "started_at": None,
        "completed_at": None,
        "elapsed_seconds": None,
        "returncode": None,
    }


def plan_task_records(tasks: Iterable[MatrixTask], resume: bool, running_commands: Iterable[str]) -> tuple[list[MatrixTask], list[dict[str, Any]]]:
    pending: list[MatrixTask] = []
    records: list[dict[str, Any]] = []
    for task in tasks:
        if resume and (task.out_dir / "summary.json").exists():
            records.append(base_manifest_record(task, "skipped_completed"))
        elif is_task_running(task, running_commands):
            records.append(base_manifest_record(task, "running_external"))
        else:
            pending.append(task)
            records.append(base_manifest_record(task, "pending"))
    return pending, records


def run_one_task(task: MatrixTask) -> dict[str, Any]:
    record = base_manifest_record(task, "running")
    started = time.perf_counter()
    record["started_at"] = utc_now_iso()
    log_dir = task.out_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    task.out_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "run.log"
    with log_path.open("w", encoding="utf-8") as log_file:
        log_file.write(f"$ {' '.join(task.command)}\n\n")
        log_file.flush()
        proc = subprocess.run(
            task.command,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            env=task_env(task),
            text=True,
        )
    record["returncode"] = int(proc.returncode)
    record["completed_at"] = utc_now_iso()
    record["elapsed_seconds"] = time.perf_counter() - started
    if proc.returncode == 0 and (task.out_dir / "summary.json").exists():
        record["status"] = "success"
    elif proc.returncode == 0:
        record["status"] = "failed_missing_summary"
    else:
        record["status"] = "failed"
    return record


def run_task_class(tasks: list[MatrixTask], max_workers: int) -> list[dict[str, Any]]:
    if not tasks:
        return []
    if max_workers <= 1:
        return [run_one_task(task) for task in tasks]
    records: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(run_one_task, task) for task in tasks]
        for future in as_completed(futures):
            records.append(future.result())
    return records


def merge_manifest_records(planned: list[dict[str, Any]], completed: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key = {
        (record["block"], record["mode"], record["seed"], record["horizon"]): record
        for record in planned
    }
    for record in completed:
        key = (record["block"], record["mode"], record["seed"], record["horizon"])
        by_key[key] = record
    return sorted(by_key.values(), key=lambda r: (r["block"], r["mode"], -1 if r["horizon"] is None else r["horizon"], r["seed"]))


def read_summary(task: MatrixTask) -> dict[str, Any] | None:
    path = task.out_dir / "summary.json"
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    row: dict[str, Any] = {
        "block": task.block,
        "mode": task.mode,
        "seed": task.seed,
        "horizon": task.horizon,
        "out_dir": str(task.out_dir),
    }
    row.update(data)
    return row


def aggregate_completed(tasks: Iterable[MatrixTask], root: Path) -> dict[str, Any]:
    rows = [row for task in tasks if (row := read_summary(task)) is not None]
    if not rows:
        return {"completed_summaries": 0, "blocks": []}

    summary_df = pd.DataFrame.from_records(rows)
    blocks_written: list[str] = []
    numeric_cols = summary_df.select_dtypes(include="number").columns.tolist()
    value_cols = [
        col
        for col in numeric_cols
        if col not in {"seed", "horizon"} and col.startswith(("final_", "max_", "mean_"))
    ]

    for block, block_df in summary_df.groupby("block", sort=True):
        block_dir = root / str(block)
        block_dir.mkdir(parents=True, exist_ok=True)
        block_df.sort_values(["mode", "horizon", "seed"], na_position="first").to_csv(
            block_dir / "summary_by_seed.csv",
            index=False,
        )
        group_cols = ["mode"] + (["horizon"] if block_df["horizon"].notna().any() else [])
        aggregate = block_df.groupby(group_cols, dropna=False)[value_cols].agg(["mean", "std", "min", "max"])
        aggregate.columns = ["_".join(str(part) for part in col if part) for col in aggregate.columns]
        aggregate = aggregate.reset_index()
        aggregate["completed_seeds"] = block_df.groupby(group_cols, dropna=False)["seed"].count().to_numpy()
        aggregate.to_csv(block_dir / "aggregate_by_mode.csv", index=False)
        blocks_written.append(str(block))

    return {"completed_summaries": int(len(summary_df)), "blocks": blocks_written}


def write_manifest(root: Path, records: list[dict[str, Any]], aggregate_info: dict[str, Any], dry_run: bool) -> dict[str, Any]:
    manifest = {
        "timestamp": utc_now_iso(),
        "dry_run": dry_run,
        "git_commit": get_git_commit(Path.cwd()),
        "python_version": platform.python_version(),
        "torch_version": get_torch_version(),
        "platform": platform.platform(),
        "aggregate_info": aggregate_info,
        "tasks": records,
    }
    root.mkdir(parents=True, exist_ok=True)
    (root / "run_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest


def run_matrix(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.root)
    blocks = validate_blocks(parse_csv(args.blocks))
    seeds = parse_int_list(args.seeds)
    tasks = build_matrix_tasks(
        root,
        blocks,
        seeds,
        rollout_device=args.rollout_device,
        rollout_backend=args.rollout_backend,
    )
    running_commands = active_command_lines()
    pending, planned_records = plan_task_records(tasks, resume=args.resume, running_commands=running_commands)

    if args.dry_run:
        for task in pending:
            print(" ".join(task.command))
        return {
            "dry_run": True,
            "tasks": planned_records,
            "pending_count": len(pending),
        }

    completed: list[dict[str, Any]] = []
    for task_class, max_workers in [
        ("neural", args.max_neural),
        ("tabular", args.max_tabular),
        ("rollout", args.max_rollout),
    ]:
        class_tasks = [task for task in pending if task.task_class == task_class]
        completed.extend(run_task_class(class_tasks, max_workers=max_workers))

    records = merge_manifest_records(planned_records, completed)
    aggregate_info = aggregate_completed(tasks, root)
    manifest = write_manifest(root, records, aggregate_info, dry_run=False)

    failures = [record for record in records if record["status"].startswith("failed")]
    if failures:
        print(f"completed with {len(failures)} failed task(s); see {root / 'run_manifest.json'}")
    else:
        print(f"matrix scheduler completed; see {root / 'run_manifest.json'}")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run implemented blocks from EXPERIMENT_MATRIX_100K_PLAN.md.")
    parser.add_argument("--root", type=str, default=str(DEFAULT_ROOT))
    parser.add_argument("--blocks", type=str, default="block3,block4_rollout")
    parser.add_argument("--seeds", type=str, default="0-9")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-neural", type=int, default=2)
    parser.add_argument("--max-tabular", type=int, default=6)
    parser.add_argument("--max-rollout", type=int, default=1)
    parser.add_argument("--rollout-device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--rollout-backend", choices=["numpy", "torch"], default="numpy")
    return parser.parse_args()


def main() -> None:
    run_matrix(parse_args())


if __name__ == "__main__":
    main()
