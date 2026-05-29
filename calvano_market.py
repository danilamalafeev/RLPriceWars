from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import numpy as np


@dataclass(frozen=True)
class CalvanoMarketConfig:
    n: int = 2
    costs: tuple[float, ...] | None = None
    qualities: tuple[float, ...] | None = None
    outside_quality: float = 0.0
    mu: float = 0.25
    demand_scale: float = 1.0
    m: int = 15
    xi: float = 0.1


@dataclass(frozen=True)
class StaticBenchmarks:
    p_n: float
    p_m: float
    price_grid: np.ndarray
    profit_matrix: np.ndarray
    nash_actions: np.ndarray
    monopoly_actions: np.ndarray
    pi_n: np.ndarray
    pi_m: np.ndarray


def market_arrays(config: CalvanoMarketConfig) -> tuple[np.ndarray, np.ndarray]:
    costs = np.full(config.n, 1.0, dtype=np.float64)
    if config.costs is not None:
        costs = np.asarray(config.costs, dtype=np.float64)
    if costs.shape != (config.n,):
        raise ValueError("costs must have shape [n]")

    qualities = costs + 1.0
    if config.qualities is not None:
        qualities = np.asarray(config.qualities, dtype=np.float64)
    if qualities.shape != (config.n,):
        raise ValueError("qualities must have shape [n]")
    return qualities, costs


def logit_demand(
    prices: np.ndarray,
    qualities: np.ndarray,
    outside_quality: float,
    mu: float,
) -> tuple[np.ndarray, float]:
    if mu <= 0 or not np.isfinite(mu):
        raise ValueError("mu must be finite and > 0")
    prices = np.asarray(prices, dtype=np.float64)
    qualities = np.asarray(qualities, dtype=np.float64)
    utilities = (qualities - prices) / mu
    outside_utility = outside_quality / mu
    max_u = max(float(np.max(utilities)), float(outside_utility))
    exp_i = np.exp(utilities - max_u)
    exp_out = float(np.exp(outside_utility - max_u))
    denom = float(np.sum(exp_i) + exp_out)
    return exp_i / denom, exp_out / denom


def profit_vector(
    prices: np.ndarray,
    qualities: np.ndarray,
    costs: np.ndarray,
    outside_quality: float,
    mu: float,
    demand_scale: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, float]:
    shares, outside_share = logit_demand(prices, qualities, outside_quality, mu)
    demand = demand_scale * shares
    return (np.asarray(prices, dtype=np.float64) - costs) * demand, demand, outside_share


def _symmetric_share(price: float, n: int, quality: float, outside_quality: float, mu: float) -> tuple[float, float]:
    utility = (quality - price) / mu
    outside_utility = outside_quality / mu
    max_u = max(utility, outside_utility)
    exp_i = np.exp(utility - max_u)
    exp_out = np.exp(outside_utility - max_u)
    denom = n * exp_i + exp_out
    return float(exp_i / denom), float(exp_out / denom)


def _bisect_root(func, low: float, high: float, max_iter: int = 200, tol: float = 1e-12) -> float:
    f_low = func(low)
    f_high = func(high)
    expand = 0
    while f_low * f_high > 0 and expand < 100:
        high = high * 2.0 + 1.0
        f_high = func(high)
        expand += 1
    if f_low * f_high > 0:
        raise RuntimeError("failed to bracket continuous benchmark root")

    for _ in range(max_iter):
        mid = 0.5 * (low + high)
        f_mid = func(mid)
        if abs(f_mid) <= tol or (high - low) <= tol:
            return mid
        if f_low * f_mid <= 0:
            high = mid
            f_high = f_mid
        else:
            low = mid
            f_low = f_mid
    return 0.5 * (low + high)


def continuous_symmetric_benchmarks(config: CalvanoMarketConfig) -> tuple[float, float]:
    qualities, costs = market_arrays(config)
    if not np.allclose(qualities, qualities[0]) or not np.allclose(costs, costs[0]):
        raise ValueError("continuous benchmark solver currently expects symmetric qualities and costs")

    quality = float(qualities[0])
    cost = float(costs[0])

    def nash_foc(price: float) -> float:
        share, _ = _symmetric_share(price, config.n, quality, config.outside_quality, config.mu)
        return price - cost - config.mu / (1.0 - share)

    def monopoly_foc(price: float) -> float:
        _, outside_share = _symmetric_share(price, config.n, quality, config.outside_quality, config.mu)
        return price - cost - config.mu / outside_share

    low = cost + 1e-10
    high = max(quality + 10.0, cost + 10.0, config.outside_quality + 10.0)
    p_n = _bisect_root(nash_foc, low, high)
    p_m = _bisect_root(monopoly_foc, max(p_n, low), high)
    return p_n, p_m


def build_price_grid(p_n: float, p_m: float, m: int, xi: float) -> np.ndarray:
    if m <= 1:
        raise ValueError("m must be > 1")
    lower = p_n - xi * (p_m - p_n)
    upper = p_m + xi * (p_m - p_n)
    return np.linspace(lower, upper, m, dtype=np.float64)


def discrete_profit_matrix(
    price_grid: np.ndarray,
    qualities: np.ndarray,
    costs: np.ndarray,
    outside_quality: float,
    mu: float,
    demand_scale: float = 1.0,
) -> np.ndarray:
    if len(qualities) != 2:
        raise ValueError("discrete_profit_matrix currently supports n == 2")
    m = int(len(price_grid))
    out = np.zeros((m, m, 2), dtype=np.float64)
    for a0, a1 in product(range(m), repeat=2):
        prices = np.array([price_grid[a0], price_grid[a1]], dtype=np.float64)
        profits, _, _ = profit_vector(prices, qualities, costs, outside_quality, mu, demand_scale)
        out[a0, a1, :] = profits
    return out


def find_discrete_nash(profit_matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    m = profit_matrix.shape[0]
    for a0 in range(m):
        for a1 in range(m):
            p0 = profit_matrix[a0, a1, 0]
            p1 = profit_matrix[a0, a1, 1]
            if p0 >= np.max(profit_matrix[:, a1, 0]) - 1e-12 and p1 >= np.max(profit_matrix[a0, :, 1]) - 1e-12:
                actions = np.array([a0, a1], dtype=np.int64)
                return actions, profit_matrix[a0, a1, :].copy()
    raise RuntimeError("no pure discrete Nash equilibrium found")


def find_discrete_monopoly(profit_matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    total = profit_matrix[:, :, 0] + profit_matrix[:, :, 1]
    flat = int(np.argmax(total))
    a0, a1 = np.unravel_index(flat, total.shape)
    actions = np.array([a0, a1], dtype=np.int64)
    return actions, profit_matrix[a0, a1, :].copy()


def build_static_benchmarks(config: CalvanoMarketConfig) -> StaticBenchmarks:
    qualities, costs = market_arrays(config)
    p_n, p_m = continuous_symmetric_benchmarks(config)
    price_grid = build_price_grid(p_n, p_m, config.m, config.xi)
    profit_matrix = discrete_profit_matrix(
        price_grid,
        qualities,
        costs,
        config.outside_quality,
        config.mu,
        config.demand_scale,
    )
    nash_actions, pi_n = find_discrete_nash(profit_matrix)
    monopoly_actions, pi_m = find_discrete_monopoly(profit_matrix)
    return StaticBenchmarks(
        p_n=p_n,
        p_m=p_m,
        price_grid=price_grid,
        profit_matrix=profit_matrix,
        nash_actions=nash_actions,
        monopoly_actions=monopoly_actions,
        pi_n=pi_n,
        pi_m=pi_m,
    )
