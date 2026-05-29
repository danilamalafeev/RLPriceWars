# 100k+ Experimental Matrix Plan

This document defines the next long-run experimental matrix for the adversarial
Calvano pricing project. The purpose is to replace short 20k-50k probes with
stable learning trajectories and enough seeds to distinguish transient
undercutting from converged dynamics.

## Global Rules

- Main Oracle reward remains own profit:

```text
r_O = profit_O
```

- Do not use asymmetry reward shaping for main results.
- Use `K=15` unless a specific robustness check says otherwise.
- Use 10 seeds for every main condition: `0,1,2,3,4,5,6,7,8,9`.
- Every run must write its own:

```text
config.json
train_metrics.csv
eval_metrics.csv
summary.json
plots/
```

- Every matrix directory must include aggregate files:

```text
aggregate_by_mode.csv
summary_by_seed.csv
```

- Recommended evaluation cadence:

```text
eval_every = 5000
eval_steps = 2000
```

For 150k-200k neural runs, keep the same cadence so convergence plots remain
comparable across architectures.

## Block 1: Sanity Checks And Core Controls

### 1. Symmetric Tabular Q-vs-Q Baseline

Purpose:

Verify that the Calvano baseline still produces stable tacit collusion over a
longer horizon.

Run:

```text
seeds: 0-9
total_steps: 100000
agents: tabular Q-learning vs tabular Q-learning
```

Expected reference:

```text
Nash price:       about 1.473
Monopoly price:   about 1.925
Q-vs-Q price:     about 1.803
Q-vs-Q profit:    about 0.322 / 0.322
```

Success criterion:

The long-run baseline should reproduce the previous representative collusive
benchmark without material drift.

### 2. Static-Victim Control

Purpose:

Check whether stronger Oracle agents can exploit a cooperative, non-adaptive
Victim when the strategic instability of online learning is removed.

Run:

```text
seeds: 0-9
total_steps: 100000
Victim: static cooperative policy
Oracle modes:
  - dqn
  - tabular_cfr
```

Question:

Can the Oracle reach clean high-margin exploitation when the Victim is frozen
near the cooperative/collusive price?

Implementation prerequisite:

If static Victim is not yet implemented, add it before running this block. It
should be logged as a control, not mixed with adaptive-Victim results.

## Block 2: Tabular Heterogeneity

Purpose:

Test whether asymmetric tabular Q-learning alone destroys collusion over long
horizons. This block is not an Oracle architecture test; it isolates learning
speed and patience asymmetry.

Common settings:

```text
agents: tabular Q-learning vs tabular Q-learning
seeds: 0-9
total_steps: 100000
```

### 1. Learning-Rate Grid

```text
alpha_oracle = 0.15, alpha_victim = 0.03
alpha_oracle = 0.03, alpha_victim = 0.15
```

Main diagnostic:

Does the faster learner become an undercutter, a price leader, or a price
umbrella provider?

### 2. Discount-Factor Grid

```text
delta_oracle = 0.95, delta_victim = 0.70
delta_oracle = 0.70, delta_victim = 0.95
```

Main diagnostic:

Does the more patient learner preserve collusion or exploit the less patient
learner?

Implementation prerequisite:

The runner must support independent alpha/delta values for each tabular agent.
If it does not, add this as a separate baseline runner before interpreting the
results.

## Block 3: Long-Run Myopic Architecture Sweep

Purpose:

Re-run the existing stronger-agent architectures for enough steps to determine
whether short-run undercutting is transient or converged.

Common settings:

```text
seeds: 0-9
total_steps: 150000
Victim: adaptive tabular Q-learning
```

Modes:

```text
actor_critic, reservoir_dim=512
dqn
dqn_jepa
dqn_regret
tabular_cfr
```

Note:

`tabular_cfr` likely does not need 150k steps, but keeping it in the same
matrix improves plot comparability.

Primary diagnostics:

```text
oracle_profit
victim_profit
profit_gap
oracle_price
victim_price
market_price
distance_to_nash_price
distance_to_monopoly_price
industry_profit
```

Interpretation rules:

```text
higher market_price + lower oracle_profit  -> price umbrella
higher gap + lower market_price            -> destructive exploitation
higher market_price + higher oracle_profit -> promising strategic shaping
no improvement over Q-vs-Q profit          -> no absolute-profit breakthrough
```

## Block 4: Opponent Shaping And Control

Purpose:

Test whether explicit opponent-learning control can break the undercutting /
price-umbrella tradeoff.

### 1. Multi-Step Model-Based MPC / Rollout Oracle

Run:

```text
seeds: 0-9
total_steps: 150000
Victim: adaptive tabular Q-learning
rollout horizons:
  - L = 5
  - L = 12
  - L = 25
```

Rationale:

The one-step model-based LOLA result exposed a dead-update problem: the Victim
Q-learning update often changes `Q[old_state, action]`, while the next Victim
choice is made from `Q[next_state, :]`. A multi-step cloned-Q rollout is needed
so Oracle actions can affect the Victim table through state revisits.

Implementation prerequisite:

Add a new mode such as:

```text
tabular_rollout_lola
```

It should:

- clone the Victim Q-table,
- simulate the Victim Q-learning process for `L` future steps,
- evaluate cumulative discounted Oracle own profit,
- choose the real Oracle action from the rollout values,
- keep the actual Oracle reward as `profit_O`.

### 2. State-Space Augmented DQN

Run:

```text
seeds: 0-9
total_steps: 150000
Victim: adaptive tabular Q-learning
Oracle: DQN with augmented opponent-response state
```

State augmentation:

Estimate a rolling conditional Victim response curve:

```text
E[p_V | p_O]
```

and feed it to the Oracle state representation.

Implementation prerequisite:

If this augmented DQN state is not implemented, add it as a separate
experimental mode. Do not mix it with vanilla DQN runs.

## Output Directory Layout

Recommended root:

```text
results/long_matrix_100k_plus/
```

Suggested structure:

```text
results/long_matrix_100k_plus/
  block1_q_vs_q_100k/
  block1_static_victim_100k/
  block2_tabular_heterogeneity_100k/
  block3_architectures_150k/
  block4_rollout_lola_150k/
  block4_augmented_dqn_150k/
```

Each condition should be nested by mode and seed:

```text
<block>/<mode_or_condition>/seed_<seed>/
```

Example:

```text
results/long_matrix_100k_plus/block3_architectures_150k/dqn_regret/seed_0/
```

## Runner Requirements

The batch runner should:

- fail fast on missing modes,
- skip finished runs only when `summary.json` exists,
- write stdout/stderr logs per run,
- aggregate each block after all seeds finish,
- preserve per-seed plots,
- write a top-level manifest with command lines and git commit hash.

Suggested manifest:

```text
run_manifest.json
```

Minimum manifest fields:

```text
timestamp
git_commit
python_version
torch_version
device
block
mode
seed
command
status
elapsed_seconds
```

## Decision Criteria

The matrix should answer four questions:

1. Does the Q-vs-Q collusive benchmark remain stable at 100k steps?
2. Are current Oracle gains just transient undercutting?
3. Does any architecture beat Q-vs-Q absolute Oracle profit around `0.322`?
4. Does multi-step opponent modeling raise both market price and Oracle profit,
   or does it only create another price umbrella?

The most important positive result would be:

```text
oracle_profit > Q-vs-Q collusive profit
market_price increases toward the collusive range
victim_profit does not capture most of the price increase
result is stable across 10 seeds
```

The most important negative result would be:

```text
all richer or opponent-aware Oracle modes either undercut or create a price
umbrella, even at 100k-150k steps
```

That negative result would still be valuable because it would support the
information-paradox interpretation: more strategic information can redistribute
profit without improving absolute margins in Bertrand-style adaptive pricing.
