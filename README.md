# Calvano Market C++ Kernel

This repository implements the first layer of the research system: a vectorized C++ market simulator with a `pybind11` Python binding. It is only a fast market kernel. It does not implement Buy Box logic, RL training, PyTorch integration, LOLA, DiCE, agents, replay buffers, multiprocessing, or distributed training.

## Model

The environment follows the Calvano et al. style repeated differentiated Bertrand oligopoly with a finite discrete price grid. At each step Python passes integer price action indices:

```text
actions_idx: int64 [B, A]
```

For market `b` and firm `i`:

```text
prices[b, i] = price_grid[actions_idx[b, i]]
utility[b, i] = (qualities[b, i] - prices[b, i]) / mu
outside_utility = outside_quality / mu
```

Demand is multinomial logit with a stable softmax:

```text
m[b] = max(max_i utility[b, i], outside_utility)
exp_i[b, i] = exp(utility[b, i] - m[b])
exp_out[b] = exp(outside_utility - m[b])
denom[b] = sum_i exp_i[b, i] + exp_out[b]
market_share[b, i] = exp_i[b, i] / denom[b]
outside_share[b] = exp_out[b] / denom[b]
demand[b, i] = demand_scale * market_share[b, i]
rewards[b, i] = (prices[b, i] - costs[b, i]) * demand[b, i]
```

`mu` must be finite and strictly positive.

## Shape Contract

Configuration dictionary:

```text
B: int
A: int, default 2
K: int
H: int
price_grid: float32 [K]
qualities: float32 [A] or [B, A]
costs: float32 [A] or [B, A]
outside_quality: float32
mu: float32 > 0
demand_scale: float32, default 1.0
random_seed: uint64
```

Python API:

```python
import calvano_market_cpp as cm

env = cm.create_env(config)
cm.reset(env, optional_seed=None)
cm.step(env, actions_idx)

cm.get_current_prices(env)      # float32 [B, A]
cm.get_demand(env)              # float32 [B, A]
cm.get_rewards(env)             # float32 [B, A]
cm.get_market_share(env)        # float32 [B, A]
cm.get_outside_share(env)       # float32 [B]
cm.get_margins(env)             # float32 [B, A]
cm.get_price_gap(env)           # float32 [B]
cm.get_mean_price(env)          # float32 [B]
cm.get_min_price(env)           # float32 [B]
cm.get_max_price(env)           # float32 [B]
cm.get_price_history_view(env)  # float32 [B, H, A]
```

For `A == 2`, `price_gap` is `abs(p0 - p1)`. For any `A`, it is `max_price - min_price`, which is equivalent when `A == 2`.

## Memory Layout

All internal arrays are contiguous row-major C++ `std::vector<float>` buffers. Per-market firm arrays use `[B, A]` layout with `A` as the fastest-moving dimension.

`step()` checks action bounds and writes into preallocated output buffers. The core C++ step function does not allocate market output memory.

## Zero-Copy History Contract

Price history is stored as a mirrored ring buffer:

```text
price_history_mirror: float32 [B, 2 * H, A]
```

At each step, current prices are written to both `write_idx` and `write_idx + H`, then `head` is incremented. `get_price_history_view(env)` returns a NumPy view over:

```text
price_history_mirror[:, start:start + H, :]
start = head % H
```

The returned view is zero-copy and has a valid lifetime tied to the env handle. `torch.from_numpy(view)` can consume it without copying when PyTorch is installed.

After every `step()`, `head` changes and therefore `start` changes. An old NumPy view may point at the previous history window. Call `get_price_history_view(env)` again after `step()` to get the current last-`H` window.

## Benchmarks

For one-market `A == 2` configs, the module also exposes:

```python
cm.compute_static_profit_matrix(config)  # float32 [K, K, 2]
cm.find_discrete_nash_prices(config)     # dict: actions, prices, profits
cm.find_joint_monopoly_prices(config)    # dict: actions, prices, total_profit, per_firm_profit
```

These are grid-search utilities for later Nash, monopoly, collusion-index, and supra-competitive-pricing metrics.

## Build And Test

```bash
python -m pip install -e .
pytest
```

The C++ market kernel is not connected to autograd. It accepts price action indices and returns market arrays; gradient tracking belongs outside this layer.

# Calvano Et Al. 2020 Q-Learning Replication

This layer reproduces the tabular independent Q-learning baseline from Calvano et al. 2020, "Artificial Intelligence, Algorithmic Pricing, and Collusion." It is deliberately procedural: dataclass configs, arrays, and functions. It does not introduce PyTorch, neural agents, LOLA, DiCE, opponent shaping, replay buffers, or Buy Box mechanics.

## Baseline Parameters

The default symmetric configuration is:

```text
n = 2
c_i = 1
a_i = 2
a_0 = 0
mu = 0.25
delta = 0.95
m = 15
xi = 0.1
k = 1
```

For each joint price vector `p`, logit demand and profits are:

```text
q_i(p) = exp((a_i - p_i) / mu) /
         (sum_j exp((a_j - p_j) / mu) + exp(a_0 / mu))

pi_i(p) = (p_i - c_i) * q_i(p)
```

The implementation uses a stable softmax.

## Static Benchmarks

`calvano_market.py` computes:

```text
continuous symmetric Bertrand-Nash price p_N
continuous symmetric joint monopoly price p_M
finite grid of m equally spaced prices:
[p_N - xi * (p_M - p_N), p_M + xi * (p_M - p_N)]
discrete profit_matrix[a0, a1, firm]
discrete Nash action pair
discrete monopoly action pair
pi_N and pi_M
```

The continuous solver currently targets the symmetric baseline used for the paper reproduction.

## State And Q-Tables

For `n = 2, k = 1`:

```text
state_id = prev_action_0 * m + prev_action_1
```

For general `n, k`, state ids are base-`m` encodings of the last `k` joint action vectors, flattened oldest to newest. The state space size is:

```text
S = m ** (n * k)
```

Q-tables have shape:

```text
Q: float64 [n, S, m]
```

Initialization follows Calvano Eq. 8:

```text
Q_i[s, a_i] =
  mean_{a_-i uniformly over A^(n-1)} pi_i(a_i, a_-i) / (1 - delta)
```

The value is constant across states for each player/action pair.

## Learning Rule

Exploration uses:

```text
epsilon_t = exp(-beta * t)
```

Each player independently chooses a random action with probability `epsilon_t`; otherwise it chooses the lowest-index greedy maximizer of `Q_i[state, :]`.

After simultaneous actions, each player updates only the visited table cell:

```text
Q_i[s, a_i] =
  (1 - alpha) * Q_i[s, a_i]
  + alpha * (pi_i + delta * max_a Q_i[s_next, a])
```

The main training loop is compiled with `numba`; Python handles orchestration and reporting. The loop keeps the greedy policy array in memory and, after each Bellman update, recomputes only the visited `(player, state)` greedy action with lowest-index tie-breaking.

## Convergence And Evaluation

A session converges when the greedy policy induced by `Q` has not changed for `convergence_window` consecutive periods. Concretely, after each period's Bellman updates, the implementation checks whether any visited `(player, state)` greedy action changed. If at least one changed, the stable counter resets to zero; otherwise it increments. The default window is `100_000`, with a paper-style `max_periods` cap of `1_000_000_000`.

After convergence or cap, the greedy policy is frozen and evaluated without exploration for `eval_periods` periods. The evaluation reports average long-run prices/profits and the first detected deterministic cycle length.

Profit gain is:

```text
Delta = (pi_bar - pi_N) / (pi_M - pi_N)
```

where `pi_bar` is the evaluation average per-firm profit, `pi_N` is the discrete Nash benchmark profit, and `pi_M` is the discrete joint monopoly benchmark profit.

## Seeding And Sessions

Independent sessions are deterministic by construction:

```text
session_seed = base_seed + global_session_id
```

For a parameter grid, `global_session_id` is ordered by cell and session. The CLI writes rows sorted by `cell_id, session`. Serial execution is the default. Passing `--workers N` runs independent sessions in a process pool while preserving deterministic per-session seeds and output ordering.

Unit tests and regression ranges are smoke checks for implementation sanity. They do not attempt paper-level replication, which requires many more sessions and substantially larger period caps.

## CLI

Representative check:

```bash
python run_calvano_replication.py \
  --representative-check \
  --sessions 10 \
  --workers 1 \
  --max-periods 200000 \
  --convergence-window 100000 \
  --eval-periods 10000 \
  --out results/calvano_representative_check.csv
```

Single custom cell:

```bash
python run_calvano_replication.py \
  --sessions 4 \
  --alpha 0.15 \
  --beta 4e-6 \
  --max-periods 200000 \
  --out results/calvano_replication.csv
```

Small development grid:

```bash
python run_calvano_replication.py \
  --grid-debug \
  --sessions 4 \
  --workers 2 \
  --max-periods 5000 \
  --convergence-window 50 \
  --eval-periods 500 \
  --out results/calvano_grid_debug.csv
```

The full grid is available through `calvano_qlearning.parameter_grid(debug=False)`: 100 alpha values in `[0.025, 0.25]` crossed with 100 beta values up to `2e-5`.

# Full Replication Runner

For reproducible experiment series with raw sessions, aggregates, plots, and a machine-readable report, use:

```bash
python scripts/run_full_calvano_replication.py --mode representative --out-dir results/calvano_representative
```

Supported modes:

```text
representative: alpha = 0.15, beta = 4e-6, m = 15
midpoint:       alpha = 0.125, beta = 1e-5, m = 15
debug-grid:     5 x 5 alpha/beta grid
full-grid:      100 x 100 alpha/beta grid
```

Representative local run:

```bash
python scripts/run_full_calvano_replication.py \
  --mode representative \
  --sessions 100 \
  --workers 4 \
  --out-dir results/calvano_representative \
  --overwrite
```

Midpoint run:

```bash
python scripts/run_full_calvano_replication.py \
  --mode midpoint \
  --sessions 100 \
  --workers 4 \
  --out-dir results/calvano_midpoint \
  --overwrite
```

Debug grid:

```bash
python scripts/run_full_calvano_replication.py \
  --mode debug-grid \
  --sessions 8 \
  --workers 4 \
  --max-periods 5000 \
  --convergence-window 50 \
  --eval-periods 500 \
  --out-dir results/calvano_debug_grid \
  --overwrite
```

Resume an interrupted run:

```bash
python scripts/run_full_calvano_replication.py \
  --mode debug-grid \
  --sessions 8 \
  --workers 4 \
  --out-dir results/calvano_debug_grid \
  --resume
```

If `out-dir` exists, the runner fails unless `--overwrite` or `--resume` is passed. Resume loads existing `raw_sessions.csv` or `raw_sessions.parquet`, skips completed `(cell_id, session)` pairs, appends missing rows, and sorts final raw output by `cell_id, session`. Session seeds are deterministic:

```text
session_seed = seed + cell_id * sessions + session
```

Each run writes:

```text
out_dir/
  config.json
  raw_sessions.csv or raw_sessions.parquet
  aggregate_by_cell.csv
  summary.json
  plots/
    profit_gain_heatmap.png
    convergence_rate_heatmap.png
    avg_price_heatmap.png
    representative_price_distribution.png
    cycle_length_distribution.png
    profit_gain_distribution.png
```

For representative and midpoint modes, heatmaps are skipped and distribution plots are generated. Plotting errors are captured in `summary.json` under `plot_warnings` instead of failing the experiment.

The full grid is computationally expensive: `100 x 100` cells times the requested session count, with potentially long convergence windows. Unit tests use tiny configs and only validate pipeline correctness, resume behavior, aggregation, and plotting. They do not validate paper-level quantitative replication.

# Neural Duopoly Rollout Layer

The `neural/` package is a separate differentiable policy layer for later Oracle/Victim experiments. It is not the Calvano tabular replication layer.

Scope:

```text
duopoly only: agent 0 = Oracle, agent 1 = Victim
functional PyTorch policies
manual pytree parameters, no nn.Module training interface
rollout dict with graph-bearing logp/entropy tensors
detached market tensors from the C++ environment
REINFORCE-compatible losses and manual SGD update
fixed reservoir features for optional memory expansion
```

The market remains non-differentiable. Gradients flow through policy log probabilities, not through prices, rewards, demand, or C++ simulator state. The rollout stores `logp` with the autograd graph intact, so later DiCE/LOLA-style objectives can call policies with alternate parameter pytrees such as `victim_policy_fn(phi_prime, buffers, obs, state)`.

Main modules:

```text
neural/observations.py          build detached market observations
neural/reservoir.py             fixed reservoir update and reservoir observations
neural/functional_policies.py   linear and MLP functional policies
neural/rollout.py               action sampling and duopoly rollout collection
neural/losses.py                returns, advantages, REINFORCE and actor-critic losses
```

Minimal usage:

```python
from neural.functional_policies import init_mlp_policy, mlp_policy_forward
from neural.observations import ObservationConfig, observation_dim
from neural.rollout import RolloutConfig, collect_duopoly_rollout
from neural.losses import OptimizerConfig, train_pg_step

Z = observation_dim(H)
oracle_params = init_mlp_policy(generator, Z, hidden_dim=32, K=K)
victim_params = init_mlp_policy(generator, Z, hidden_dim=32, K=K)

rollout = collect_duopoly_rollout(
    env,
    mlp_policy_forward,
    oracle_params,
    {},
    mlp_policy_forward,
    victim_params,
    {},
    ObservationConfig(price_min=float(price_grid[0]), price_max=float(price_grid[-1])),
    RolloutConfig(T=T, B=B, H=H, K=K),
    generator,
)
metrics = train_pg_step(oracle_params, victim_params, rollout, OptimizerConfig(lr=0.01))
```

This layer intentionally does not implement LOLA, DiCE, SEQ-JEPA, CFR, adversarial training loops, or a large experiment runner yet.

# Reservoir Oracle Experiments

`experiments/reservoir_oracle.py` is the first neural experiment layer for Oracle vs Victim pricing. Its purpose is to isolate whether giving the Oracle a deeper fixed memory of market history through reservoir computing creates a strategic advantage before adding LOLA, DiCE, or opponent-shaping objectives.

Scenarios:

```text
mlp_vs_mlp:
  Oracle = MLP policy
  Victim = MLP policy

reservoir_oracle_vs_mlp:
  Oracle = fixed reservoir + MLP policy
  Victim = MLP policy

reservoir_vs_reservoir:
  Oracle = fixed reservoir + MLP policy
  Victim = fixed reservoir + MLP policy

reservoir_oracle_vs_linear:
  Oracle = fixed reservoir + MLP policy
  Victim = linear policy
```

The policies are still functional parameter dictionaries. Reservoir matrices are fixed buffers with `requires_grad=False`; only policy and optional value-head parameters are updated. The C++ Calvano market remains non-differentiable, and gradients flow through policy log probabilities and actor-critic value losses.

Metrics recorded each update include average profits, average prices, profit and price asymmetry, policy losses, entropies, entropy coefficient, and actor-critic value diagnostics when enabled. Evaluation reports greedy-policy profit, price, distance to Nash/monopoly prices, per-agent profit gain, and Oracle-minus-Victim asymmetry.

Entropy can be annealed to reduce near-random behavior:

```text
entropy_coef_start -> entropy_coef_end over entropy_anneal_steps
```

If `entropy_coef_start` is omitted or `entropy_anneal_steps <= 0`, the constant `entropy_coef` path is used for backward compatibility.

Actor-critic mode adds separate functional value heads for Oracle and Victim. Value inputs match each agent's policy inputs, so a reservoir Oracle value head receives `base_obs + reservoir_state`, while a non-reservoir Victim value head receives only the base observation. Advantages are computed as `returns - values`, policy advantages are normalized, and value losses are optimized with separate value learning rates.

Example smoke run:

```bash
python -m experiments.reservoir_oracle \
  --scenario reservoir_oracle_vs_mlp \
  --updates 200 \
  --B 64 \
  --T 32 \
  --eval-every 20 \
  --eval-episodes 4 \
  --out-dir results/reservoir_oracle_smoke \
  --seed 0
```

Actor-critic run with entropy annealing:

```bash
python -m experiments.reservoir_oracle \
  --scenario reservoir_oracle_vs_mlp \
  --training-mode actor_critic \
  --updates 1000 \
  --B 64 \
  --T 32 \
  --entropy-coef-start 0.01 \
  --entropy-coef-end 0.0 \
  --entropy-anneal-steps 500 \
  --out-dir results/ac_reservoir_oracle_seed0 \
  --seed 0
```

Reservoir depth sweep:

```bash
python -m experiments.reservoir_oracle \
  --depth-sweep \
  --reservoir-depths 0,16,32,64,128,256 \
  --seeds 0,1,2,3,4 \
  --training-mode actor_critic \
  --updates 1000 \
  --B 64 \
  --T 32 \
  --entropy-coef-start 0.01 \
  --entropy-coef-end 0.0 \
  --entropy-anneal-steps 500 \
  --base-out-dir results/reservoir_depth_sweep
```

In a depth sweep, `depth=0` is the no-reservoir baseline (`mlp_vs_mlp`). Positive depths run `reservoir_oracle_vs_mlp` with `reservoir_dim_oracle=depth`. The sweep writes `depth_sweep_raw.csv`, `depth_sweep_aggregate.csv`, `depth_sweep_summary.json`, and depth plots for Oracle profit, asymmetry, and profit gain.

Outputs:

```text
out_dir/
  config.json
  train_metrics.csv
  eval_metrics.csv
  summary.json
  plots/
    profit_timeseries.png
    price_timeseries.png
    asymmetry_timeseries.png
    eval_profit_gain.png
```

This is a policy-gradient neural experiment, not the tabular Calvano replication. It is intended as infrastructure for later Oracle/Victim work; it does not implement LOLA, DiCE, SEQ-JEPA, CFR, higher-order gradients, or adversarial training.

# Reservoir-DQN Oracle Vs Adaptive Q-Learning Victim

`experiments/dqn_oracle_vs_qvictim.py` is the first experiment aligned with an adaptive-competitor setup. The Oracle is a neural DQN policy with fixed reservoir memory; the Victim is an online adaptive tabular Calvano Q-learning repricer.

Key mechanics:

```text
Oracle:
  observation = detached Calvano market features + fixed reservoir state
  action = epsilon-greedy DQN over discrete price grid
  learning = replay buffer + target network + DQN MSE loss

Victim:
  independent tabular Q-learning
  per-market Q table [B, S, K]
  state_id = previous_oracle_action * K + previous_victim_action
  online Q update after every market step
```

The Victim is not neural and has no gradients. It adapts during Oracle training and during the default evaluation mode. The market remains non-differentiable; Oracle learning is value-based and uses only detached transition tuples.

This layer does not implement LOLA, DiCE, SEQ-JEPA, CFR, higher-order gradients, or opponent shaping.

Example:

```bash
python -m experiments.dqn_oracle_vs_qvictim \
  --total-steps 50000 \
  --B 64 \
  --reservoir-dim 128 \
  --asymmetry-coef 0.0 \
  --out-dir results/dqn_oracle_qvictim_seed0 \
  --seed 0
```

Smoke-sized run:

```bash
python -m experiments.dqn_oracle_vs_qvictim \
  --total-steps 200 \
  --B 8 \
  --H 8 \
  --K 15 \
  --train-every 4 \
  --eval-every 100 \
  --eval-steps 200 \
  --batch-size 32 \
  --out-dir results/dqn_oracle_qvictim_smoke \
  --seed 0
```

Outputs:

```text
out_dir/
  config.json
  train_metrics.csv
  eval_metrics.csv
  summary.json
  plots/
    profit_timeseries.png
    price_timeseries.png
    asymmetry_timeseries.png
    dqn_loss.png
    eval_profit_gain.png
```
