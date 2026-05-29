from __future__ import annotations

from argparse import Namespace
from pathlib import Path

import numpy as np
import pandas as pd

from calvano_market import CalvanoMarketConfig, build_static_benchmarks
from scripts.run_full_calvano_replication import (
    _run_one_session,
    aggregate_results,
    build_experiment_cells,
    generate_plots,
    make_summary,
    run_full_replication,
)


def tiny_args(out_dir: Path, mode: str = "representative", sessions: int = 1, resume: bool = False) -> Namespace:
    return Namespace(
        mode=mode,
        sessions=sessions,
        workers=1,
        seed=5,
        max_periods=20,
        convergence_window=3,
        eval_periods=5,
        out_dir=str(out_dir),
        overwrite=not resume,
        resume=resume,
        format="csv",
        m=3,
        k=1,
        delta=0.95,
        mu=0.25,
        xi=0.1,
    )


def synthetic_raw() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "mode": "debug-grid",
                "cell_id": 0,
                "alpha": 0.1,
                "beta": 0.0,
                "session": 0,
                "converged": True,
                "profit_gain_mean": 0.2,
                "periods_to_convergence": 10,
                "long_run_avg_price_0": 1.4,
                "long_run_avg_price_1": 1.6,
                "long_run_avg_profit_0": 0.2,
                "long_run_avg_profit_1": 0.3,
                "detected_cycle_length": 1,
            },
            {
                "mode": "debug-grid",
                "cell_id": 0,
                "alpha": 0.1,
                "beta": 0.0,
                "session": 1,
                "converged": False,
                "profit_gain_mean": 0.8,
                "periods_to_convergence": 20,
                "long_run_avg_price_0": 1.8,
                "long_run_avg_price_1": 2.0,
                "long_run_avg_profit_0": 0.4,
                "long_run_avg_profit_1": 0.5,
                "detected_cycle_length": 2,
            },
        ]
    )


def test_build_experiment_cells_counts():
    assert len(build_experiment_cells("representative")) == 1
    assert len(build_experiment_cells("midpoint")) == 1
    assert len(build_experiment_cells("debug-grid")) == 25
    assert len(build_experiment_cells("full-grid")) == 10_000
    assert build_experiment_cells("representative")[0]["cell_id"] == 0


def test_aggregate_results_synthetic_values():
    agg = aggregate_results(synthetic_raw())
    assert len(agg) == 1
    row = agg.iloc[0]
    assert row["sessions"] == 2
    np.testing.assert_allclose(row["convergence_rate"], 0.5)
    np.testing.assert_allclose(row["average_profit_gain"], 0.5)
    np.testing.assert_allclose(row["median_profit_gain"], 0.5)
    np.testing.assert_allclose(row["constant_price_frequency"], 0.5)
    np.testing.assert_allclose(row["cycle_length_2_frequency"], 0.5)
    assert "share_profit_gain_above_0_5" in agg.columns


def test_resume_logic_skips_completed_pairs(tmp_path):
    out_dir = tmp_path / "resume_run"
    out_dir.mkdir()
    (out_dir / "plots").mkdir()
    market_config = CalvanoMarketConfig(m=3)
    benchmarks = build_static_benchmarks(market_config)
    cell = build_experiment_cells("debug-grid")[0]
    partial = _run_one_session(
        {
            "mode": "debug-grid",
            "cell": cell,
            "session_id": 0,
            "session_seed": 5,
            "alpha": cell["alpha"],
            "beta": cell["beta"],
            "delta": 0.95,
            "n": 2,
            "k": 1,
            "m": 3,
            "convergence_window": 3,
            "max_periods": 20,
            "eval_periods": 5,
            "market_config": market_config,
            "benchmarks": benchmarks,
            "code_version": "test",
            "dirty_worktree": None,
        }
    )
    pd.DataFrame([partial]).to_csv(out_dir / "raw_sessions.csv", index=False)

    run_full_replication(tiny_args(out_dir, mode="debug-grid", sessions=2, resume=True))

    raw = pd.read_csv(out_dir / "raw_sessions.csv")
    assert len(raw) == 50
    assert raw[["cell_id", "session"]].duplicated().sum() == 0


def test_summary_json_creation(tmp_path):
    out_dir = tmp_path / "summary_run"
    summary = run_full_replication(tiny_args(out_dir, mode="representative", sessions=1))
    assert (out_dir / "summary.json").exists()
    assert summary["run_config"]["mode"] == "representative"
    assert summary["number_of_cells"] == 1
    assert summary["number_of_sessions_completed"] == 1
    assert "benchmark_values" in summary
    assert "best_cell_by_average_profit_gain" in summary


def test_plotting_functions_create_pngs_or_warnings(tmp_path):
    raw = synthetic_raw()
    agg = aggregate_results(raw)
    warnings = generate_plots(raw, agg, tmp_path / "plots", mode="representative")
    created = {p.name for p in (tmp_path / "plots").glob("*.png")}
    assert warnings or {
        "representative_price_distribution.png",
        "cycle_length_distribution.png",
        "profit_gain_distribution.png",
    }.issubset(created)


def test_make_summary_required_keys():
    raw = synthetic_raw()
    agg = aggregate_results(raw)
    benchmarks = build_static_benchmarks(CalvanoMarketConfig(m=3))
    summary = make_summary(
        run_config={"mode": "debug-grid"},
        raw_df=raw,
        aggregate_df=agg,
        benchmarks=benchmarks,
        total_wall_time_seconds=1.25,
        plot_warnings=[],
    )
    for key in [
        "run_config",
        "number_of_cells",
        "number_of_sessions_completed",
        "total_wall_time_seconds",
        "overall_convergence_rate",
        "overall_average_profit_gain",
        "best_cell_by_average_profit_gain",
        "best_cell_by_convergence_rate",
        "benchmark_values",
    ]:
        assert key in summary
