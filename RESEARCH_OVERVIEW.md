# Research Overview

## Topic

This project studies adversarial algorithmic pricing in a Calvano-style
Bertrand-logit duopoly.

The starting point is the Calvano et al. result that independent tabular
Q-learning repricers can learn supracompetitive prices without explicit
communication. This repository reproduces that baseline and then replaces one
of the symmetric Q-learning agents with a stronger Oracle.

The core question is:

```text
Can a strategically stronger pricing agent destabilize tacit collusion between
adaptive repricers and redirect market dynamics in its own favor?
```

## Core Rule

The main Oracle reward remains own-profit maximization:

```text
r_O = profit_O
```

This constraint matters. A result is not treated as primary evidence if it
depends on artificial price floors, collusion bonuses, or direct penalties on
the competitor. Relative-profit rewards are useful diagnostic controls, but not
the main mechanism.

## Baseline

The reference point is symmetric tabular Q-learning vs tabular Q-learning.

Representative result:

```text
Nash price            about 1.473
Monopoly price        about 1.925
Q-vs-Q market price   about 1.803
Q-vs-Q firm profit    about 0.322 / 0.322
```

Interpretation:

Two symmetric adaptive Q-learning agents converge to a high-price tacit
collusion pattern. This benchmark is the main absolute-profit reference for all
Oracle experiments.

## What Counts As Success

The project separates relative exploitation from strategic dominance.

Possible outcomes:

```text
destructive exploitation:
  Oracle undercuts the Victim, the Victim loses profit, and market price falls

price umbrella:
  Oracle raises price or market price, but the Victim captures much of the gain

strategic dominance:
  Oracle preserves high market prices while shifting profit toward itself
```

The desired signal is:

```text
Oracle absolute profit >= Q-vs-Q collusive profit
and market price remains in the high-price range
and the Victim does not capture most of the surplus created by higher prices
```

No current experiment robustly satisfies that standard.

## Current Evidence

### Reservoir And DQN Oracles

Reservoir memory and DQN give the Oracle richer state representations of recent
market history.

Representative 50k 10-seed results:

```text
mode            oracle_profit       victim_profit       market_price
actor_critic    about 0.313         about 0.226         about 1.638
DQN             about 0.306         about 0.239         about 1.653
```

Interpretation:

Memory and value learning help the Oracle exploit the adaptive Victim, but
mostly through undercutting. The Oracle profit remains near, but below, the
Q-vs-Q collusive benchmark, while the Victim is damaged and market prices fall.

### Asymmetry Reward Sweep

Diagnostic reward:

```text
r_O = profit_O + asymmetry_coef * (profit_O - profit_V)
```

Result:

Increasing `asymmetry_coef` mostly increases undercutting. It widens the profit
gap by harming the Victim, but it does not robustly improve Oracle absolute
profit.

Interpretation:

Relative-profit reward shaping is not a clean solution. It confirms that the
environment can produce asymmetric outcomes, but those outcomes are not the
main own-profit mechanism.

### DQN-JEPA

DQN-JEPA adds an auxiliary prediction objective intended to improve the
Oracle's latent model of future market and Victim dynamics.

10-seed 50k result:

```text
mode      oracle_profit       victim_profit       market_price
DQN       0.3060 +/- 0.0142   0.2394 +/- 0.0398   1.6526 +/- 0.0430
DQN-JEPA  0.3014 +/- 0.0189   0.2477 +/- 0.0515   1.6614 +/- 0.0560
```

Interpretation:

JEPA raises prices on average and reduces destructive undercutting, but it does
not improve Oracle profit. The likely mechanism is a partial price umbrella:
the market becomes less destructive, but the Victim captures part of the gain.

### CFR And Multi-Step CFR

Tabular CFR-style agents test whether explicit counterfactual payoff accounting
can avoid local undercutting.

3-seed 20k comparison:

```text
mode                    oracle_profit  victim_profit  profit_gap  market_price
tabular_cfr             0.3080         0.2273         +0.0806     1.6401
tabular_multi_cfr       0.3099         0.2282         +0.0816     1.6412
```

Interpretation:

CFR produces a stable undercutting attractor. Multi-step value accounting is a
small improvement, but it remains far from the Q-vs-Q collusive benchmark. This
suggests the failure is not only neural instability, replay noise, or reservoir
state design.

### LOLA-Style Opponent Awareness

LOLA-style variants test whether the Oracle can account for the Victim's future
learning process while keeping the Oracle reward as own profit.

Tabular LOLA-lite:

```text
tau   oracle_profit  victim_profit  profit_gap  market_price
0.03  0.3028         0.2231         +0.0797     1.6354
0.05  0.2982         0.2370         +0.0611     1.6507
0.10  0.2913         0.2546         +0.0367     1.6702
```

One-step model-based LOLA:

```text
tau   oracle_profit  victim_profit  profit_gap  market_price
0.03  0.3015         0.2399         +0.0615     1.6537
0.05  0.2963         0.2496         +0.0467     1.6644
0.10  0.2898         0.2618         +0.0280     1.6782
```

Interpretation:

LOLA-style agents can raise prices, but higher prices mostly benefit the
adaptive Victim. This is a price-umbrella failure. The one-step Q-table update
is too shallow: it often changes `Q[old_state, action]`, while the next Victim
decision is made from `Q[next_state, :]`.

### Multi-Step Rollout / MPC Oracle

`tabular_rollout_lola` is implemented as the next opponent-shaping prototype.
It clones the Victim Q-table, simulates future Victim Q-learning updates over a
finite horizon, evaluates discounted Oracle own profit, and selects the real
Oracle action from those rollout values.

This mode exists in the runner, but the long-run matrix needed to evaluate it
is still pending. It should be treated as a prototype awaiting 100k-150k
multi-seed tests, not as a solved result.

## Main Interpretation

The project currently distinguishes two different advantages:

```text
state or information advantage
opponent-shaping advantage
```

The evidence so far is:

```text
Memory can create exploitation.
Prediction can reduce destructive pricing.
Counterfactual regret stabilizes undercutting.
One-step opponent-learning awareness creates price umbrellas.
No tested mechanism yet creates robust strategic dominance.
```

The likely missing component is not more data alone. The Oracle needs a better
way to control the Victim's future learning trajectory while preserving its own
absolute margin.

## Theoretical Links

### Data And Competition

Xu, Zhang, and Zhao's work on AI, data, and competition is relevant to the
pattern observed here. The working-paper version is associated with
"Algorithmic Collusion and Price Discrimination: The Over-Usage of Data"; the
later title appears as "Artificial Intelligence, Data and Competition".

Links:

```text
https://arxiv.org/abs/2403.06150
https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4741393
```

Connection to this project:

```text
More detailed state representations do not mechanically produce better
collusion. They can expose local exploitation opportunities and destabilize
high-price outcomes.
```

### Timing And Information In Oligopoly

Gal-Or's work is useful for interpreting why superior information or timing can
fail to increase profit in price competition.

Relevant references:

```text
Gal-Or, Esther. "First Mover and Second Mover Advantages", 1985.
Gal-Or, Esther. "Information Transmission: Cournot and Bertrand Equilibria",
Review of Economic Studies, 1986.
```

Links:

```text
https://www.econbiz.de/Record/first-mover-and-second-mover-advantages-revised-march-1985-gal-esther/10002208015
https://academic.oup.com/restud/article/53/1/85/1579985
```

The link is conceptual rather than literal: in price competition, more
information can create response opportunities for the opponent or intensify
local rivalry. That is consistent with the observed undercutting and
price-umbrella failures.

## Required Metrics

Every main experiment should report:

```text
oracle_profit
victim_profit
profit_gap = oracle_profit - victim_profit
oracle_price
victim_price
market_price
distance_to_nash_price
distance_to_monopoly_price
oracle_profit_gain
victim_profit_gain
variance across seeds
```

Prices must be reported with profits because welfare interpretation depends on
the mechanism:

```text
higher gap + lower market price            -> destructive exploitation
higher market price + lower Oracle profit  -> price umbrella
higher market price + higher Oracle profit -> promising strategic shaping
```

## Next Matrix

The next research step is a long-run 100k-150k matrix:

```text
100k: symmetric Q-vs-Q baseline and static-Victim controls
100k: tabular heterogeneity over learning rate and discount factor
150k: reservoir AC, DQN, DQN-JEPA, DQN-Regret, tabular CFR
150k: tabular_rollout_lola horizons and augmented opponent-response states
```

The full plan is in:

```text
EXPERIMENT_MATRIX_100K_PLAN.md
```

## Defensible Claim

The current defensible claim is:

```text
Richer state representations can destabilize symmetric tacit collusion against
an adaptive Q-learning competitor and redistribute profit toward the stronger
agent, even under pure own-profit maximization.
```

## Non-Claims

The current results do not support the following claims:

```text
The Oracle achieves monopoly power.
The Oracle robustly earns more than the Q-vs-Q collusive benchmark.
JEPA solves destructive undercutting.
CFR solves the absolute-margin problem.
One-step LOLA or one-step model-based LOLA successfully shapes the Victim.
```

The correct current position is:

```text
The experiments expose limits of memory, prediction, one-step counterfactual
regret, and one-step opponent-learning awareness. The next mechanism to test is
multi-step control over the Victim's Q-learning trajectory.
```
