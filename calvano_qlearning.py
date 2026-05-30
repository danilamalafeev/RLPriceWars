from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass

import numpy as np
from numba import njit

from calvano_market import CalvanoMarketConfig, StaticBenchmarks, build_static_benchmarks, market_arrays, profit_vector


@dataclass(frozen=True)
class QLearningConfig:
    alpha: float = 0.15
    beta: float = 4e-6
    delta: float = 0.95
    alpha_0: float | None = None
    alpha_1: float | None = None
    beta_0: float | None = None
    beta_1: float | None = None
    delta_0: float | None = None
    delta_1: float | None = None
    n: int = 2
    k: int = 1
    m: int = 15
    convergence_window: int = 100_000
    max_periods: int = 1_000_000_000
    eval_periods: int = 10_000
    seed: int = 0
    keep_q: bool = False


@dataclass
class SessionResult:
    converged: bool
    periods_to_convergence: int
    final_greedy_policy: np.ndarray
    final_q: np.ndarray | None
    last_prices: np.ndarray
    detected_cycle_length: int
    long_run_avg_price: np.ndarray
    long_run_avg_profit: np.ndarray
    profit_gain_delta: np.ndarray


def state_space_size(n: int, k: int, m: int) -> int:
    return int(m ** (n * k))


def encode_state(history: np.ndarray, m: int) -> int:
    state = 0
    for action in np.asarray(history, dtype=np.int64).ravel():
        if action < 0 or action >= m:
            raise ValueError("history action outside [0, m)")
        state = state * m + int(action)
    return state


def decode_state(state_id: int, n: int, k: int, m: int) -> np.ndarray:
    if state_id < 0 or state_id >= state_space_size(n, k, m):
        raise ValueError("state_id outside valid state space")
    out = np.zeros((k, n), dtype=np.int64)
    value = int(state_id)
    for idx in range(n * k - 1, -1, -1):
        out.ravel()[idx] = value % m
        value //= m
    return out


def update_state_id(state_id: int, actions: np.ndarray, n: int, k: int, m: int) -> int:
    joint = encode_state(np.asarray(actions, dtype=np.int64).reshape(1, n), m)
    if k == 1:
        return joint
    keep_mod = m ** (n * (k - 1))
    return (state_id % keep_mod) * (m**n) + joint


def epsilon_schedule(t: int, beta: float) -> float:
    return float(np.exp(-beta * t))


def greedy_action(q_values: np.ndarray) -> int:
    return int(np.argmax(q_values))


def greedy_policy(Q: np.ndarray) -> np.ndarray:
    return np.argmax(Q, axis=2).astype(np.int64)


def agent_learning_parameters(q_config: QLearningConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if q_config.n != 2:
        raise ValueError("per-agent learning parameter overrides currently support n == 2")
    alpha = np.array(
        [
            q_config.alpha if q_config.alpha_0 is None else q_config.alpha_0,
            q_config.alpha if q_config.alpha_1 is None else q_config.alpha_1,
        ],
        dtype=np.float64,
    )
    beta = np.array(
        [
            q_config.beta if q_config.beta_0 is None else q_config.beta_0,
            q_config.beta if q_config.beta_1 is None else q_config.beta_1,
        ],
        dtype=np.float64,
    )
    delta = np.array(
        [
            q_config.delta if q_config.delta_0 is None else q_config.delta_0,
            q_config.delta if q_config.delta_1 is None else q_config.delta_1,
        ],
        dtype=np.float64,
    )
    return alpha, beta, delta


def initialize_q_tables(profit_matrix: np.ndarray, delta: float | np.ndarray, n: int, k: int, m: int) -> np.ndarray:
    if n != 2:
        raise ValueError("Q initialization currently supports n == 2")
    S = state_space_size(n, k, m)
    delta_arr = np.asarray(delta, dtype=np.float64)
    if delta_arr.ndim == 0:
        delta_arr = np.full(n, float(delta_arr), dtype=np.float64)
    if delta_arr.shape != (n,):
        raise ValueError(f"delta must be scalar or shape ({n},)")
    Q = np.zeros((n, S, m), dtype=np.float64)
    for a_i in range(m):
        Q[0, :, a_i] = float(np.mean(profit_matrix[a_i, :, 0])) / (1.0 - delta_arr[0])
        Q[1, :, a_i] = float(np.mean(profit_matrix[:, a_i, 1])) / (1.0 - delta_arr[1])
    return Q


def bellman_update(old_value: float, reward: float, next_max: float, alpha: float, delta: float) -> float:
    return (1.0 - alpha) * old_value + alpha * (reward + delta * next_max)


@njit(cache=True)
def _argmax_lowest(values):
    best_idx = 0
    best_val = values[0]
    for idx in range(1, values.shape[0]):
        if values[idx] > best_val:
            best_val = values[idx]
            best_idx = idx
    return best_idx


@njit(cache=True)
def _compute_policy(Q, policy):
    n = Q.shape[0]
    S = Q.shape[1]
    for i in range(n):
        for s in range(S):
            policy[i, s] = _argmax_lowest(Q[i, s])


@njit(cache=True)
def _policies_equal(a, b):
    for i in range(a.shape[0]):
        for s in range(a.shape[1]):
            if a[i, s] != b[i, s]:
                return False
    return True


@njit(cache=True)
def _joint_action_id(actions, m):
    out = 0
    for i in range(actions.shape[0]):
        out = out * m + actions[i]
    return out


@njit(cache=True)
def _next_state_id(state, actions, n, k, m):
    joint = _joint_action_id(actions, m)
    if k == 1:
        return joint
    keep_mod = m ** (n * (k - 1))
    return (state % keep_mod) * (m**n) + joint


@njit(cache=True)
def _market_rewards(actions, price_grid, qualities, costs, outside_quality, mu, demand_scale, rewards, prices):
    n = actions.shape[0]
    outside_utility = outside_quality / mu
    max_utility = outside_utility
    for i in range(n):
        prices[i] = price_grid[actions[i]]
        utility = (qualities[i] - prices[i]) / mu
        if utility > max_utility:
            max_utility = utility

    denom = np.exp(outside_utility - max_utility)
    for i in range(n):
        utility = (qualities[i] - prices[i]) / mu
        rewards[i] = np.exp(utility - max_utility)
        denom += rewards[i]

    for i in range(n):
        share = rewards[i] / denom
        rewards[i] = (prices[i] - costs[i]) * demand_scale * share


@njit(cache=True)
def _train_session_numba(
    Q,
    alpha,
    beta,
    delta,
    n,
    k,
    m,
    convergence_window,
    max_periods,
    price_grid,
    qualities,
    costs,
    outside_quality,
    mu,
    demand_scale,
    seed,
):
    np.random.seed(seed)
    S = Q.shape[1]
    policy = np.empty((n, S), dtype=np.int64)
    _compute_policy(Q, policy)

    state = 0
    for _ in range(k * n):
        state = state * m + np.random.randint(0, m)

    actions = np.empty(n, dtype=np.int64)
    rewards = np.empty(n, dtype=np.float64)
    prices = np.empty(n, dtype=np.float64)
    stable_counter = 0
    converged = False
    periods = max_periods

    for t in range(max_periods):
        for i in range(n):
            epsilon = np.exp(-beta[i] * t)
            if np.random.random() < epsilon:
                actions[i] = np.random.randint(0, m)
            else:
                actions[i] = policy[i, state]

        _market_rewards(actions, price_grid, qualities, costs, outside_quality, mu, demand_scale, rewards, prices)
        next_state = _next_state_id(state, actions, n, k, m)

        policy_changed = False
        for i in range(n):
            next_max = Q[i, next_state, 0]
            for a in range(1, m):
                if Q[i, next_state, a] > next_max:
                    next_max = Q[i, next_state, a]
            old = Q[i, state, actions[i]]
            Q[i, state, actions[i]] = (1.0 - alpha[i]) * old + alpha[i] * (rewards[i] + delta[i] * next_max)
            old_greedy = policy[i, state]
            new_greedy = _argmax_lowest(Q[i, state])
            if new_greedy != old_greedy:
                policy[i, state] = new_greedy
                policy_changed = True

        if policy_changed:
            stable_counter = 0
        else:
            stable_counter += 1

        if stable_counter >= convergence_window:
            converged = True
            periods = t + 1
            state = next_state
            break

        state = next_state

    return converged, periods, policy, Q, state


def evaluate_policy(
    policy: np.ndarray,
    initial_state: int,
    eval_periods: int,
    price_grid: np.ndarray,
    market_config: CalvanoMarketConfig,
    k: int,
) -> tuple[int, np.ndarray, np.ndarray, np.ndarray]:
    qualities, costs = market_arrays(market_config)
    n = market_config.n
    state = int(initial_state)
    seen: dict[tuple[int, tuple[int, ...]], int] = {}
    cycle_length = 0
    price_sum = np.zeros(n, dtype=np.float64)
    profit_sum = np.zeros(n, dtype=np.float64)
    last_prices = np.zeros(n, dtype=np.float64)

    for t in range(eval_periods):
        actions = policy[:, state].astype(np.int64)
        key = (state, tuple(int(x) for x in actions))
        if cycle_length == 0 and key in seen:
            cycle_length = t - seen[key]
        seen.setdefault(key, t)

        last_prices = price_grid[actions]
        profits, _, _ = profit_vector(
            last_prices,
            qualities,
            costs,
            market_config.outside_quality,
            market_config.mu,
            market_config.demand_scale,
        )
        price_sum += last_prices
        profit_sum += profits
        state = update_state_id(state, actions, n, k, len(price_grid))

    return cycle_length, last_prices, price_sum / eval_periods, profit_sum / eval_periods


def run_session(
    q_config: QLearningConfig,
    market_config: CalvanoMarketConfig,
    benchmarks: StaticBenchmarks | None = None,
    seed: int | None = None,
) -> SessionResult:
    if q_config.n != market_config.n or q_config.m != market_config.m:
        raise ValueError("q_config n/m must match market_config n/m")
    if benchmarks is None:
        benchmarks = build_static_benchmarks(market_config)
    qualities, costs = market_arrays(market_config)
    alpha, beta, delta = agent_learning_parameters(q_config)
    Q0 = initialize_q_tables(benchmarks.profit_matrix, delta, q_config.n, q_config.k, q_config.m)
    session_seed = q_config.seed if seed is None else int(seed)
    converged, periods, policy, Q, last_state = _train_session_numba(
        Q0.copy(),
        alpha,
        beta,
        delta,
        q_config.n,
        q_config.k,
        q_config.m,
        q_config.convergence_window,
        q_config.max_periods,
        benchmarks.price_grid.astype(np.float64),
        qualities.astype(np.float64),
        costs.astype(np.float64),
        market_config.outside_quality,
        market_config.mu,
        market_config.demand_scale,
        session_seed,
    )

    cycle_length, last_prices, avg_price, avg_profit = evaluate_policy(
        policy,
        int(last_state),
        q_config.eval_periods,
        benchmarks.price_grid,
        market_config,
        q_config.k,
    )
    denom = benchmarks.pi_m - benchmarks.pi_n
    profit_gain = np.divide(
        avg_profit - benchmarks.pi_n,
        denom,
        out=np.zeros_like(avg_profit),
        where=np.abs(denom) > 1e-12,
    )

    return SessionResult(
        converged=bool(converged),
        periods_to_convergence=int(periods),
        final_greedy_policy=policy.copy(),
        final_q=Q.copy() if q_config.keep_q else None,
        last_prices=last_prices.copy(),
        detected_cycle_length=int(cycle_length),
        long_run_avg_price=avg_price.copy(),
        long_run_avg_profit=avg_profit.copy(),
        profit_gain_delta=profit_gain.copy(),
    )


def run_sessions(
    sessions: int,
    q_config: QLearningConfig,
    market_config: CalvanoMarketConfig,
    benchmarks: StaticBenchmarks | None = None,
    workers: int = 1,
) -> list[SessionResult]:
    if benchmarks is None:
        benchmarks = build_static_benchmarks(market_config)
    if workers <= 1:
        return [
            run_session(q_config, market_config, benchmarks, seed=q_config.seed + idx)
            for idx in range(sessions)
        ]

    tasks = [(idx, q_config, market_config, benchmarks, q_config.seed + idx) for idx in range(sessions)]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        ordered = list(pool.map(_run_session_worker, tasks))
    ordered.sort(key=lambda item: item[0])
    return [result for _, result in ordered]


def _run_session_worker(args) -> tuple[int, SessionResult]:
    idx, q_config, market_config, benchmarks, seed = args
    return idx, run_session(q_config, market_config, benchmarks, seed=seed)


def summarize_sessions(results: list[SessionResult], benchmarks: StaticBenchmarks) -> dict[str, float]:
    gains = np.array([np.mean(r.profit_gain_delta) for r in results], dtype=np.float64)
    periods = np.array([r.periods_to_convergence for r in results], dtype=np.float64)
    prices = np.array([r.long_run_avg_price for r in results], dtype=np.float64)
    cycles = np.array([r.detected_cycle_length for r in results], dtype=np.int64)
    converged = np.array([r.converged for r in results], dtype=bool)
    return {
        "average_profit_gain": float(np.mean(gains)),
        "std_profit_gain": float(np.std(gains)),
        "average_convergence_periods": float(np.mean(periods)),
        "convergence_rate": float(np.mean(converged)),
        "constant_price_frequency": float(np.mean(cycles == 1)),
        "cycle_length_2_frequency": float(np.mean(cycles == 2)),
        "average_long_run_price": float(np.mean(prices)),
        "distance_to_nash_price": float(np.mean(np.abs(prices - benchmarks.p_n))),
        "distance_to_monopoly_price": float(np.mean(np.abs(prices - benchmarks.p_m))),
    }


def representative_config(**overrides) -> QLearningConfig:
    values = {"alpha": 0.15, "beta": 4e-6}
    values.update(overrides)
    return QLearningConfig(**values)


def midpoint_config(**overrides) -> QLearningConfig:
    values = {"alpha": 0.125, "beta": 1e-5}
    values.update(overrides)
    return QLearningConfig(**values)


def parameter_grid(debug: bool = False) -> list[tuple[float, float]]:
    if debug:
        alpha_grid = np.linspace(0.025, 0.25, 5)
        beta_grid = np.linspace(0.0, 2e-5, 5)
    else:
        alpha_grid = np.linspace(0.025, 0.25, 100)
        beta_grid = np.linspace(0.0, 2e-5, 100)
    return [(float(a), float(b)) for a in alpha_grid for b in beta_grid]
