# Adversarial RL in Price Wars

This repository is an experimental research stack for adversarial reinforcement
learning in Calvano-style algorithmic price competition.

The project starts from a symmetric Calvano et al. tabular Q-learning
replication, where two independent repricers learn supracompetitive prices. It
then replaces one learner with a stronger Oracle agent and asks whether that
agent can shape an adaptive Q-learning competitor while preserving high own
profit.

The central methodological rule is:

```text
Oracle reward = own profit
r_O = profit_O
```

Relative-profit or asymmetry rewards can be used as diagnostics, but they are
not the main research result.

## Research Question

Can a stronger pricing algorithm do more than locally undercut an adaptive
competitor?

The target outcome is not merely:

```text
Oracle profit > Victim profit
```

That can happen through destructive undercutting. The stronger result would be:

```text
Oracle preserves or improves absolute long-run profit
while pushing the adaptive competitor into a less favorable regime.
```

## Current Status

The Calvano Q-vs-Q baseline reproduces high-price tacit collusion:

```text
Nash price:          about 1.473
Monopoly price:      about 1.925
Q-vs-Q market price: about 1.803
Q-vs-Q firm profit:  about 0.322 / 0.322
```

Against an adaptive tabular Q-learning Victim, stronger Oracle variants
generally beat the Victim in relative terms, but they do not yet beat the
Q-vs-Q collusive profit benchmark in absolute terms.

The main observed failure modes are:

```text
destructive undercutting:
  Oracle gains relative profit while market price and industry margin fall

price umbrella:
  Oracle raises prices, but the Victim captures much of the benefit
```

See [RESEARCH_OVERVIEW.md](RESEARCH_OVERVIEW.md) for the current research
narrative and [RESEARCH_JOURNAL.md](RESEARCH_JOURNAL.md) for chronological
experiment notes.

## Installation

The package builds a small C++17 `pybind11` market kernel and installs the
Python experiment code in editable mode:

```bash
python -m pip install -e .
```

Run the test suite:

```bash
python -m pytest
```

## Quick Runs

Run the Calvano Q-learning baseline:

```bash
python run_calvano_replication.py --representative-check --sessions 10
```

Run a short Oracle-vs-Q-Victim smoke test:

```bash
python -m experiments.dqn_oracle_vs_qvictim \
  --oracle-kind tabular_cfr \
  --total-steps 200 \
  --B 8 --H 8 --K 15 \
  --eval-every 100 \
  --eval-steps 200 \
  --out-dir results/tabular_cfr_smoke \
  --seed 0
```

Try the multi-step rollout / MPC-style Oracle prototype:

```bash
python -m experiments.dqn_oracle_vs_qvictim \
  --oracle-kind tabular_rollout_lola \
  --rollout-lola-horizon 12 \
  --rollout-lola-num-particles 32 \
  --total-steps 2000 \
  --B 16 --H 8 --K 15 \
  --eval-every 1000 \
  --eval-steps 1000 \
  --out-dir results/rollout_lola_smoke \
  --seed 0
```

Experiment outputs are written under `results/`, which is intentionally ignored
by git.

## Implemented Oracle Modes

The main Oracle-vs-Q-Victim runner supports:

```text
actor_critic
dqn
dqn_jepa
dqn_regret
tabular_cfr
tabular_multi_cfr
tabular_lola
tabular_model_lola
tabular_rollout_lola
```

The first group tests memory, neural value learning, auxiliary prediction, and
counterfactual-regret signals. The later tabular modes test increasingly
explicit opponent-learning awareness against the adaptive Q-learning Victim.

## Repository Layout

```text
calvano_market.py                  Static Calvano benchmarks and profit matrix
calvano_qlearning.py               Tabular Calvano Q-vs-Q replication
run_calvano_replication.py         Baseline replication entrypoint
experiments/dqn_oracle_vs_qvictim.py
                                   Oracle-vs-adaptive-Q-Victim runner
experiments/reservoir_oracle.py    Reservoir actor-critic experiment runner
neural/                            PyTorch policies, observations, losses, rollout utilities
src/, include/                     C++ market kernel and pybind11 binding
tests/                             Regression, kernel, and experiment smoke tests
RESEARCH_OVERVIEW.md               Research narrative and current claims
RESEARCH_JOURNAL.md                Chronological experiment notes
EXPERIMENT_MATRIX_100K_PLAN.md     Long-run 100k-150k matrix plan
```

## Outputs

`experiments.dqn_oracle_vs_qvictim` writes the following files when `--out-dir`
is provided:

```text
config.json
train_metrics.csv
eval_metrics.csv
summary.json
progress.jsonl
```

The key summary fields are:

```text
final_eval_avg_profit_oracle
final_eval_avg_profit_victim
final_eval_profit_asymmetry
final_eval_avg_price_oracle
final_eval_avg_price_victim
final_eval_market_price_mean
final_eval_oracle_profit_gain
final_eval_victim_profit_gain
final_eval_distance_to_nash_price
final_eval_distance_to_monopoly_price
```

Prices are essential for interpretation: profit alone cannot distinguish
collusion, destructive undercutting, price umbrellas, and strategic dominance.

`progress.jsonl` is streamed during long runs when `--out-dir` is set. Each
record includes step, elapsed time, recent profit/price windows, rollout backend,
rollout horizon/particles, and device. The same progress is printed to stdout so
scheduler logs update while the run is active.

## Rollout LOLA Backend

`tabular_rollout_lola` supports the original NumPy backend and a vectorized
PyTorch backend:

```powershell
python -m experiments.dqn_oracle_vs_qvictim --oracle-kind tabular_rollout_lola --rollout-lola-backend torch --device cpu --total-steps 1000 --eval-every 500 --eval-steps 200
```

After installing a CUDA-enabled PyTorch build, switch to:

```powershell
python -m experiments.dqn_oracle_vs_qvictim --oracle-kind tabular_rollout_lola --rollout-lola-backend torch --device cuda --total-steps 1000 --eval-every 500 --eval-steps 200
```

The matrix scheduler exposes the same rollout-only controls:

```powershell
python -m scripts.run_experiment_matrix --blocks block4_rollout --rollout-backend torch --rollout-device cuda --max-rollout 1
```

Use the micro-benchmark helper for quick backend comparisons without touching
long-run result directories:

```powershell
python -m scripts.benchmark_rollout_lola --backend torch --device cpu --B 64 --K 15 --horizon 5 --particles 32
```

## Market Kernel

The lowest layer is a vectorized C++ market simulator exposed as
`calvano_market_cpp`. It accepts discrete price action indices:

```text
actions_idx: int64 [B, A]
```

and returns per-market arrays for prices, demand, shares, rewards, margins, and
price history. The model is a repeated differentiated Bertrand market with
multinomial-logit demand, a finite price grid, and an outside option.

The Python learning code uses this kernel for fast batched simulation while
keeping Q-learning, DQN, reservoir, CFR, and LOLA-style logic in Python/PyTorch.
