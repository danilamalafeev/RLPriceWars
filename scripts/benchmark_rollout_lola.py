from __future__ import annotations

import argparse
import time

import numpy as np
import torch

from experiments.dqn_oracle_vs_qvictim import (
    init_victim_state,
    make_calvano_vec_env,
    resolve_torch_device,
    tabular_rollout_lola_values,
    tabular_rollout_lola_values_torch,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Micro-benchmark rollout-LOLA value backends.")
    parser.add_argument("--backend", choices=["numpy", "torch"], default="torch")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--B", type=int, default=64)
    parser.add_argument("--H", type=int, default=8)
    parser.add_argument("--K", type=int, default=15)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--particles", type=int, default=32)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = resolve_torch_device(args.device)
    rng = np.random.default_rng(args.seed)
    torch_gen = torch.Generator(device=device).manual_seed(args.seed)
    _, price_grid, benchmarks, profit_matrix = make_calvano_vec_env(args.B, args.H, args.K, args.seed)
    victim = init_victim_state(args.B, args.K, benchmarks, delta=0.95, rng=rng)

    def run_once():
        if args.backend == "torch":
            return tabular_rollout_lola_values_torch(
                victim,
                profit_matrix,
                args.K,
                alpha=0.15,
                delta=0.95,
                beta=4e-6,
                horizon=args.horizon,
                num_particles=args.particles,
                victim_policy_mode="epsilon_greedy",
                oracle_rollout_policy="greedy_best_response",
                discount=0.95,
                include_immediate=True,
                generator=torch_gen,
                device=device,
                price_grid=price_grid,
            )
        return tabular_rollout_lola_values(
            victim,
            profit_matrix,
            args.K,
            alpha=0.15,
            delta=0.95,
            beta=4e-6,
            horizon=args.horizon,
            num_particles=args.particles,
            victim_policy_mode="epsilon_greedy",
            oracle_rollout_policy="greedy_best_response",
            discount=0.95,
            include_immediate=True,
            rng=rng,
            price_grid=price_grid,
        )

    run_once()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    timings = []
    for _ in range(max(int(args.repeats), 1)):
        started = time.perf_counter()
        run_once()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        timings.append(time.perf_counter() - started)

    mean_ms = 1000.0 * sum(timings) / len(timings)
    print(
        f"backend={args.backend} device={device} B={args.B} K={args.K} "
        f"horizon={args.horizon} particles={args.particles} mean_ms={mean_ms:.2f} "
        f"min_ms={1000.0 * min(timings):.2f}"
    )


if __name__ == "__main__":
    main()
