import numpy as np
import pytest

import calvano_market_cpp as cm


def make_config(B=4, A=2, H=8, K=5, **overrides):
    config = {
        "B": B,
        "A": A,
        "K": K,
        "H": H,
        "price_grid": np.linspace(1.0, 5.0, K, dtype=np.float32),
        "qualities": np.array([2.0, 2.2], dtype=np.float32)[:A],
        "costs": np.array([0.5, 0.6], dtype=np.float32)[:A],
        "outside_quality": 0.0,
        "mu": 0.25,
        "demand_scale": 1.0,
        "random_seed": 123,
    }
    config.update(overrides)
    return config


def test_basic_step_shapes():
    env = cm.create_env(make_config())
    cm.reset(env)
    actions = np.array([[0, 1], [2, 3], [4, 0], [1, 2]], dtype=np.int64)
    cm.step(env, actions)

    for getter in [
        cm.get_current_prices,
        cm.get_demand,
        cm.get_rewards,
        cm.get_market_share,
        cm.get_margins,
    ]:
        assert getter(env).shape == (4, 2)

    for getter in [
        cm.get_outside_share,
        cm.get_price_gap,
        cm.get_mean_price,
        cm.get_min_price,
        cm.get_max_price,
    ]:
        assert getter(env).shape == (4,)

    assert cm.get_price_history_view(env).shape == (4, 8, 2)


def test_logit_accounting():
    costs = np.array([0.75, 1.25], dtype=np.float32)
    env = cm.create_env(make_config(costs=costs))
    actions = np.array([[0, 1], [2, 3], [4, 0], [1, 2]], dtype=np.int64)
    cm.step(env, actions)

    prices = cm.get_current_prices(env)
    demand = cm.get_demand(env)
    rewards = cm.get_rewards(env)
    market_share = cm.get_market_share(env)
    outside_share = cm.get_outside_share(env)
    margins = cm.get_margins(env)

    np.testing.assert_allclose(market_share.sum(axis=1) + outside_share, 1.0, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(margins, prices - costs.reshape(1, 2), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(rewards, margins * demand, rtol=1e-6, atol=1e-6)


def test_history_returns_last_h_prices_in_order():
    B, A, H, K = 3, 2, 5, 7
    price_grid = np.linspace(10.0, 16.0, K, dtype=np.float32)
    env = cm.create_env(make_config(B=B, A=A, H=H, K=K, price_grid=price_grid))

    recorded = []
    for t in range(H + 4):
        actions = np.zeros((B, A), dtype=np.int64)
        for b in range(B):
            for a in range(A):
                actions[b, a] = (t + b + 2 * a) % K
        cm.step(env, actions)
        recorded.append(price_grid[actions])

    history = cm.get_price_history_view(env)
    expected = np.stack(recorded[-H:], axis=1)
    np.testing.assert_allclose(history, expected, rtol=0.0, atol=0.0)


def test_zero_copy_history_view_and_torch_from_numpy():
    env = cm.create_env(make_config(B=2, H=4))
    cm.step(env, np.array([[0, 1], [2, 3]], dtype=np.int64))

    view = cm.get_price_history_view(env)
    assert view.shape == (2, 4, 2)
    assert not view.flags["OWNDATA"]

    torch = pytest.importorskip("torch")
    tensor = torch.from_numpy(view)
    assert tuple(tensor.shape) == (2, 4, 2)

    cm.step(env, np.array([[4, 4], [1, 1]], dtype=np.int64))
    fresh_view = cm.get_price_history_view(env)
    assert fresh_view.shape == (2, 4, 2)
    np.testing.assert_allclose(fresh_view[:, -1, :], np.array([[5.0, 5.0], [2.0, 2.0]], dtype=np.float32))


def test_numerical_stability_extreme_utilities():
    config = make_config(
        B=2,
        K=3,
        price_grid=np.array([-1.0e6, 0.0, 1.0e6], dtype=np.float32),
        qualities=np.array([[1.0e6, -1.0e6], [-1.0e6, 1.0e6]], dtype=np.float32),
        costs=np.array([0.0, 0.0], dtype=np.float32),
        outside_quality=0.0,
        mu=0.01,
    )
    env = cm.create_env(config)
    cm.step(env, np.array([[0, 2], [2, 0]], dtype=np.int64))

    for arr in [
        cm.get_market_share(env),
        cm.get_outside_share(env),
        cm.get_demand(env),
        cm.get_rewards(env),
    ]:
        assert np.isfinite(arr).all()

    np.testing.assert_allclose(
        cm.get_market_share(env).sum(axis=1) + cm.get_outside_share(env),
        1.0,
        rtol=1e-6,
        atol=1e-6,
    )


def test_invalid_actions_raise():
    env = cm.create_env(make_config())
    with pytest.raises(Exception):
        cm.step(env, np.array([[0, 1], [2, 3], [4, 0], [1, -1]], dtype=np.int64))

    with pytest.raises(Exception):
        cm.step(env, np.array([[0, 1], [2, 3], [5, 0], [1, 2]], dtype=np.int64))


def test_static_benchmarks_smoke():
    config = make_config(B=1, H=2)
    matrix = cm.compute_static_profit_matrix(config)
    assert matrix.shape == (5, 5, 2)

    nash = cm.find_discrete_nash_prices(config)
    monopoly = cm.find_joint_monopoly_prices(config)
    assert nash["actions"].shape == (2,)
    assert monopoly["prices"].shape == (2,)


def test_static_profit_matrix_does_not_mutate_existing_env_history():
    config = make_config(B=1, H=4)
    env = cm.create_env(config)
    cm.step(env, np.array([[1, 2]], dtype=np.int64))
    head_before = env.head
    history_before = cm.get_price_history_view(env).copy()

    matrix = cm.compute_static_profit_matrix(config)

    assert matrix.shape == (5, 5, 2)
    assert env.head == head_before
    np.testing.assert_allclose(cm.get_price_history_view(env), history_before)
