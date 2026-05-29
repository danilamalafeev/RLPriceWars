import numpy as np

from calvano_market import (
    CalvanoMarketConfig,
    build_price_grid,
    build_static_benchmarks,
    logit_demand,
    market_arrays,
    profit_vector,
)
from calvano_qlearning import (
    QLearningConfig,
    bellman_update,
    decode_state,
    encode_state,
    epsilon_schedule,
    greedy_action,
    initialize_q_tables,
    run_sessions,
    run_session,
    state_space_size,
)


def test_logit_demand_normalization():
    qualities = np.array([2.0, 2.0])
    prices = np.array([1.5, 1.7])
    shares, outside = logit_demand(prices, qualities, 0.0, 0.25)
    np.testing.assert_allclose(np.sum(shares) + outside, 1.0)


def test_profit_matrix():
    cfg = CalvanoMarketConfig(m=5)
    bench = build_static_benchmarks(cfg)
    qualities, costs = market_arrays(cfg)
    prices = bench.price_grid[[1, 3]]
    rewards, demand, _ = profit_vector(prices, qualities, costs, cfg.outside_quality, cfg.mu)
    np.testing.assert_allclose(rewards, (prices - costs) * demand)


def test_price_grid():
    p_n, p_m, m, xi = 1.5, 2.0, 15, 0.1
    grid = build_price_grid(p_n, p_m, m, xi)
    assert len(grid) == m
    np.testing.assert_allclose(grid[0], p_n - xi * (p_m - p_n))
    np.testing.assert_allclose(grid[-1], p_m + xi * (p_m - p_n))


def test_state_encoding():
    h1 = np.array([[2, 1]], dtype=np.int64)
    state = encode_state(h1, m=3)
    assert state == 2 * 3 + 1
    np.testing.assert_array_equal(decode_state(state, n=2, k=1, m=3), h1)

    h2 = np.array([[0, 2], [1, 2]], dtype=np.int64)
    state2 = encode_state(h2, m=3)
    np.testing.assert_array_equal(decode_state(state2, n=2, k=2, m=3), h2)


def test_q_initialization():
    cfg = CalvanoMarketConfig(m=4)
    bench = build_static_benchmarks(cfg)
    Q = initialize_q_tables(bench.profit_matrix, delta=0.95, n=2, k=1, m=4)
    assert Q.shape == (2, state_space_size(2, 1, 4), 4)
    for a in range(4):
        expected0 = np.mean(bench.profit_matrix[a, :, 0]) / (1 - 0.95)
        expected1 = np.mean(bench.profit_matrix[:, a, 1]) / (1 - 0.95)
        np.testing.assert_allclose(Q[0, :, a], expected0)
        np.testing.assert_allclose(Q[1, :, a], expected1)


def test_epsilon_schedule():
    assert epsilon_schedule(0, beta=4e-6) == 1.0
    assert epsilon_schedule(100, beta=4e-6) < epsilon_schedule(10, beta=4e-6)


def test_tie_break():
    assert greedy_action(np.array([1.0, 2.0, 2.0, 0.5])) == 1


def test_single_q_update():
    updated = bellman_update(old_value=10.0, reward=2.0, next_max=12.0, alpha=0.25, delta=0.95)
    expected = 0.75 * 10.0 + 0.25 * (2.0 + 0.95 * 12.0)
    np.testing.assert_allclose(updated, expected)


def test_convergence_smoke():
    market_cfg = CalvanoMarketConfig(m=3)
    bench = build_static_benchmarks(market_cfg)
    q_cfg = QLearningConfig(
        alpha=0.15,
        beta=1e-3,
        m=3,
        convergence_window=10,
        max_periods=200,
        eval_periods=20,
        seed=7,
    )
    result = run_session(q_cfg, market_cfg, bench)
    assert isinstance(result.converged, bool)
    assert result.final_greedy_policy.shape == (2, 9)
    assert np.isfinite(result.long_run_avg_profit).all()
    assert np.isfinite(result.profit_gain_delta).all()


def test_representative_short_run():
    market_cfg = CalvanoMarketConfig(m=3)
    bench = build_static_benchmarks(market_cfg)
    q_cfg = QLearningConfig(
        alpha=0.15,
        beta=4e-6,
        m=3,
        convergence_window=10,
        max_periods=200,
        eval_periods=20,
        seed=11,
    )
    results = run_sessions(2, q_cfg, market_cfg, bench)
    assert len(results) == 2
    for result in results:
        assert result.final_greedy_policy.shape == (2, 9)
        assert result.last_prices.shape == (2,)
        assert np.isfinite(result.long_run_avg_price).all()
