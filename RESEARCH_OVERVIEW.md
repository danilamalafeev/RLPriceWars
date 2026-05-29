# Research Overview

## Topic

This project studies adversarial algorithmic pricing in a Calvano-style duopoly.

The baseline question is:

```text
Can a strategically stronger pricing agent destabilize tacit collusion between adaptive repricers and redirect market dynamics in its own favor?
```

The project starts from Calvano et al. (2020), where independent Q-learning agents in a Bertrand-logit market learn supracompetitive prices without explicit communication. We reproduce this baseline and then replace one of the symmetric Q-learning agents with a stronger Oracle agent.

## Core Setup

Market:

```text
two firms
discrete price grid
logit demand
Bertrand price competition
synthetic simulation data
```

Agents:

```text
Victim:
  adaptive tabular Q-learning repricer
  same family as the Calvano baseline
  updates online from observed profits

Oracle:
  stronger agent with richer state representations
  observes market history through reservoir / auxiliary models
  currently trained on own profit unless explicitly stated
```

Main methodological constraint:

```text
The main reward must remain own-profit maximization:

r_O = profit_O
```

This is important because the research should not rely on hand-shaped objectives such as price floors, artificial collusion bonuses, or direct penalties for the competitor.

## Baseline Logic

The Calvano Q-learning vs Q-learning baseline is the reference point.

Observed result:

```text
Q-vs-Q market price   ≈ 1.803
Q-vs-Q firm profit    ≈ 0.322 / 0.322
Nash price            ≈ 1.473
Monopoly price        ≈ 1.925
```

Interpretation:

Two symmetric adaptive Q-learning agents can converge to tacit collusion. Prices are far above Nash, and both firms earn high symmetric profits.

This baseline is not just a benchmark for profit. It is the equilibrium-like target that tells us what "successful collusive coordination" looks like in the simulated economy.

## What We Are Testing

The research asks whether replacing one Q-learning agent with a stronger Oracle changes the market.

There are three possible outcomes:

```text
1. Destructive exploitation:
   Oracle undercuts Victim, Victim loses profit, market price falls.

2. Price umbrella failure:
   Oracle raises prices, market becomes less destructive, but Victim captures the benefit.

3. Strategic dominance:
   Oracle preserves high market prices while shifting profit toward itself.
```

The desired research signal is not merely "Oracle beats Victim." That can happen through destructive undercutting.

The stronger claim would be:

```text
Oracle maintains or improves its absolute long-run profit
while shaping the Victim into a less favorable adaptive regime.
```

## Current Results

### Reservoir Oracle

Reservoir memory gives the Oracle a richer state representation of recent market history.

Observed pattern:

```text
Oracle price < Victim price
market price falls relative to Q-vs-Q
Oracle profit remains close to collusive profit
Victim profit falls substantially
```

Representative result:

```text
AC Reservoir vs Q:
  market_price   ≈ 1.638
  oracle_profit  ≈ 0.313
  victim_profit  ≈ 0.226
```

Interpretation:

Memory helps the Oracle exploit the adaptive Victim, but mostly through undercutting. It destabilizes symmetric tacit collusion but does not beat the Q-vs-Q collusive profit benchmark.

### Asymmetry Reward Sweep

We tested:

```text
r_O = profit_O + asymmetry_coef * (profit_O - profit_V)
```

Result:

Increasing `asymmetry_coef` mostly increases undercutting. It raises the profit gap by damaging the Victim, but it does not robustly improve Oracle absolute profit.

Interpretation:

Relative-profit reward shaping is not a clean solution. It is useful as a diagnostic control, but not as the main research mechanism.

### DQN-JEPA

JEPA was tested as an auxiliary prediction mechanism, not as reward shaping.

Goal:

```text
Improve the Oracle's latent model of future market/Victim dynamics.
```

10-seed 50k result:

```text
mode      oracle_profit       victim_profit       market_price
DQN       0.3060 +/- 0.0142   0.2394 +/- 0.0398   1.6526 +/- 0.0430
DQN-JEPA  0.3014 +/- 0.0189   0.2477 +/- 0.0515   1.6614 +/- 0.0560
```

Interpretation:

JEPA raises prices and reduces aggressive undercutting on average, but it does not improve Oracle profit. It creates a partial price umbrella: the market becomes less destructive, but the Victim captures part of the benefit.

This suggests that prediction alone is not enough. The Oracle becomes more cautious, but not more strategically dominant.

## Main Economic Interpretation So Far

The project has found a clear distinction between:

```text
state/memory advantage
```

and

```text
opponent-shaping advantage
```

Reservoir and JEPA improve the Oracle's representation of the market. But representation alone does not teach the Oracle how to control the Victim's future learning trajectory.

Current evidence:

```text
Memory can create exploitation.
Prediction can reduce destructive pricing.
Neither reliably creates strategic dominance.
```

This is why the next stage should move toward explicit opponent-aware mechanisms.

## Theoretical Links

### Over-Usage Of Data

Xu, Zhang and Zhao's work on AI, data and competition is directly relevant to the pattern observed in this project. Their paper studies how richer data inputs reshape algorithmic pricing competition. The working-paper version is associated with the title "Algorithmic Collusion and Price Discrimination: The Over-Usage of Data", and the later title appears as "Artificial Intelligence, Data and Competition".

Relevant link:

```text
https://arxiv.org/abs/2403.06150
https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4741393
```

The useful theoretical connection is:

```text
More detailed data/state representations do not mechanically produce better collusion.
They can expose local exploitation opportunities and destabilize high-price outcomes.
```

In this project:

```text
Calvano Q-vs-Q:
  limited state representation
  stable high-price tacit collusion

Reservoir / DQN / Regret-lite Oracle:
  richer history or counterfactual information
  local exploitation through undercutting
  lower market prices than Q-vs-Q
  lower Oracle absolute profit than symmetric collusion
```

This gives an economic interpretation of the negative results:

```text
The Oracle's informational advantage is being used for local exploitation,
not for strategic discipline.
```

### Gal-Or: Timing And Information In Oligopoly

Gal-Or's work is useful for interpreting why superior information or a more "advanced" pricing system can fail to increase profit in price competition.

Relevant references:

```text
Gal-Or, Esther. "First Mover and Second Mover Advantages", 1985.
Gal-Or, Esther. "Information Transmission: Cournot and Bertrand Equilibria", Review of Economic Studies, 1986.
```

Useful links:

```text
https://www.econbiz.de/Record/first-mover-and-second-mover-advantages-revised-march-1985-gal-esther/10002208015
https://academic.oup.com/restud/article/53/1/85/1579985
```

The relevant idea is not that these papers literally model our reinforcement-learning setup. The link is conceptual:

```text
In price competition, information and timing can change strategic advantage.
An informed or proactive firm can become vulnerable if its actions create
profitable response opportunities for the opponent.
```

In our experiments:

```text
DQN / DQN-Regret:
  uses richer information to find local undercutting opportunities
  damages Victim but also lowers market margin

DQN-JEPA:
  raises Oracle price and market price
  creates a partial price umbrella
  Victim captures part of the benefit
```

This matches the broad information-paradox logic:

```text
More information can reduce the informed firm's payoff if it induces
aggressive local reactions or creates exploitable response opportunities.
```

## Current Theoretical Narrative

The current research narrative can be stated as:

```text
1. Limited-information Q-learning can sustain tacit collusion.
2. Richer state representations create local exploit opportunities.
3. Local exploitation destabilizes margins and can reduce industry profit.
4. Prediction-based correction, such as JEPA, can reduce undercutting but may
   create a price umbrella that benefits the adaptive competitor.
5. Therefore the next required mechanism is not more data alone, but
   opponent-aware strategic reasoning.
```

This motivates CFR and LOLA:

```text
CFR / multi-step regret:
  use information to reason over counterfactual paths, not only one-step gains.

LOLA / opponent shaping:
  use information to affect the opponent's future learning process.
```

## Why CFR And LOLA Are Next

### CFR / Regret Accounting

CFR-style logic is relevant because the Oracle needs to understand counterfactual losses:

```text
What did I lose by choosing this price instead of another price,
given how the Victim behaved?
```

A regret-aware auxiliary model can test whether explicit counterfactual payoff information helps the Oracle avoid locally profitable but margin-destroying undercutting.

This is not reward shaping if the actual reward remains own profit and regret is used as an auxiliary learning signal.

### LOLA / Opponent-Learning Awareness

LOLA is relevant because the Victim is not a fixed opponent. It learns.

The key question becomes:

```text
How does today's Oracle action change tomorrow's Victim policy?
```

LOLA-style updates try to account for the opponent's learning gradient. In this project, the equivalent idea is to let the Oracle reason about the Victim's future Q-learning update, not just the immediate market response.

This directly targets the failure mode observed so far:

```text
Oracle reacts to the Victim,
but does not yet shape the Victim.
```

## Metrics That Must Always Be Reported

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

Prices are essential because profit alone cannot distinguish:

```text
collusion
destructive undercutting
price umbrella
strategic dominance
```

## Current Defensible Claim

The defensible claim at the current stage is:

```text
Richer state representations can destabilize symmetric tacit collusion
against an adaptive Q-learning competitor and redistribute profit toward
the stronger agent, even under pure own-profit maximization.
```

But:

```text
Reservoir and JEPA do not yet show robust absolute-profit improvement
over the symmetric Q-vs-Q collusive benchmark.
```

## Current Non-Claim

We should not yet claim:

```text
Oracle achieves monopoly power.
Oracle earns more than the Q-vs-Q collusive benchmark.
JEPA solves destructive undercutting.
CFR or LOLA will necessarily solve the problem.
```

The correct position is:

```text
Reservoir and JEPA expose the limitations of memory/prediction-only agents.
CFR and LOLA are justified next mechanisms to test because they explicitly
target regret and opponent adaptation.
```
