import numpy as np

from calvano_market import CalvanoMarketConfig, build_static_benchmarks
from calvano_qlearning import QLearningConfig, run_sessions, summarize_sessions


def test_calvano_regression_ranges():
    market_cfg = CalvanoMarketConfig(m=5)
    benchmarks = build_static_benchmarks(market_cfg)
    q_cfg = QLearningConfig(
        alpha=0.15,
        beta=4e-6,
        m=5,
        convergence_window=50,
        max_periods=5000,
        eval_periods=500,
        seed=101,
    )

    results = run_sessions(4, q_cfg, market_cfg, benchmarks)
    summary = summarize_sessions(results, benchmarks)

    assert 0.0 <= summary["convergence_rate"] <= 1.0
    assert np.isfinite(summary["average_profit_gain"])
    assert np.isfinite(summary["average_long_run_price"])
    assert benchmarks.price_grid[0] <= summary["average_long_run_price"] <= benchmarks.price_grid[-1]

    for result in results:
        assert result.detected_cycle_length > 0
        assert np.isfinite(result.long_run_avg_price).all()
        assert np.isfinite(result.long_run_avg_profit).all()
        assert np.isfinite(result.profit_gain_delta).all()
        assert -2.0 <= float(np.mean(result.profit_gain_delta)) <= 2.0


def test_parallel_sessions_match_serial_results():
    market_cfg = CalvanoMarketConfig(m=3)
    benchmarks = build_static_benchmarks(market_cfg)
    q_cfg = QLearningConfig(
        alpha=0.15,
        beta=4e-6,
        m=3,
        convergence_window=10,
        max_periods=300,
        eval_periods=40,
        seed=222,
    )

    serial = run_sessions(3, q_cfg, market_cfg, benchmarks, workers=1)
    parallel = run_sessions(3, q_cfg, market_cfg, benchmarks, workers=2)

    for left, right in zip(serial, parallel):
        assert left.converged == right.converged
        assert left.periods_to_convergence == right.periods_to_convergence
        np.testing.assert_array_equal(left.final_greedy_policy, right.final_greedy_policy)
        np.testing.assert_allclose(left.long_run_avg_price, right.long_run_avg_price)
        np.testing.assert_allclose(left.long_run_avg_profit, right.long_run_avg_profit)
