# Research Journal

## 2026-05-28 - Calvano Q-Learning Baseline And Reservoir Oracle Runs

### Core Reward Definition

Current Oracle reward in `experiments/dqn_oracle_vs_qvictim.py`:

```text
r_O = profit_O + asymmetry_coef * (profit_O - profit_V)
```

For the reported Reservoir Oracle vs Q-learning Victim sweep:

```text
asymmetry_coef = 0.0
```

Therefore the Oracle was trained on pure own-profit maximization:

```text
r_O = profit_O
```

This is important methodologically: the reported asymmetry was not directly injected through the reward. The Oracle was not given an explicit bonus for hurting the Victim, increasing relative advantage, maintaining a price floor, or moving prices toward monopoly.

### Calvano Q-Learning Vs Q-Learning Baseline

Source:

```text
results/calvano_representative_50/
```

Configuration:

```text
agents: tabular Q-learning vs tabular Q-learning
sessions: 50
alpha: 0.15
beta: 4e-06
delta: 0.95
K: 15
```

Benchmarks:

```text
p_Nash = 1.4729
p_Monopoly = 1.9250
pi_Nash = 0.2202
pi_Monopoly = 0.3375
```

Observed long-run results:

```text
average_long_run_price = 1.8026 +/- 0.0722
average_long_run_profit = 0.3220 +/- 0.0127
average_profit_gain = 0.8684
convergence_rate = 1.0
```

Interpretation:

The baseline reproduces tacit collusion: prices converge well above Nash and below monopoly, with symmetric high profits for both firms.

### Reservoir Oracle Vs Adaptive Q-Learning Victim

Source:

```text
results/qvictim_reservoir_compare_mem512_10seeds/
```

Configuration:

```text
Victim: adaptive tabular Calvano Q-learning
Oracle variants: actor_critic reservoir, DQN reservoir
seeds: 10 per Oracle variant
B: 64
H: 8
K: 15
total_steps: 50000
eval_steps: 5000
reservoir_dim: 512
reward: own_profit only
```

Aggregate results with prices:

```text
mode           oracle_price     victim_price     market_price     oracle_profit   victim_profit   profit_gap
actor_critic   1.579 +/- 0.022  1.698 +/- 0.000  1.638 +/- 0.011  0.3129         0.2259         +0.0870
DQN            1.608 +/- 0.086  1.698 +/- 0.000  1.653 +/- 0.043  0.3060         0.2394         +0.0667
```

Comparison against Q-learning vs Q-learning:

```text
mode                  market_price   oracle_or_firm_profit   victim_or_firm_profit
Q-learning vs Q       1.803          0.322                   0.322
AC reservoir vs Q     1.638          0.313                   0.226
DQN reservoir vs Q    1.653          0.306                   0.239
```

Interpretation:

The Reservoir Oracle does not increase its absolute profit above the symmetric Q-learning collusion baseline. Instead, it breaks the symmetric tacit-collusion outcome and creates an asymmetric allocation:

```text
Oracle price < Victim price
Oracle profit remains close to collusive profit
Victim profit falls substantially
market price falls relative to Q-vs-Q
```

Current defensible research claim:

An agent with richer state/memory can destabilize symmetric tacit collusion against an adaptive Q-learning competitor and redistribute profits in its own favor, even when trained only on own-profit maximization.

Current non-claim:

These results do not yet show that the Oracle achieves higher absolute profit than the symmetric Q-vs-Q collusive benchmark, nor that it creates stronger monopoly pricing.

### Immediate Methodological Notes

Prices must be included in every main table because welfare interpretation depends on whether the mechanism raises market prices, lowers market prices, or only reallocates demand between firms.

Main metrics to report from now on:

```text
profit_oracle
profit_victim
profit_gap = profit_oracle - profit_victim
price_oracle
price_victim
market_price
distance_to_nash_price
distance_to_monopoly_price
profit_gain_oracle
profit_gain_victim
```

Avoid reward shaping, staged training, or price-floor penalties as primary evidence unless separately motivated by an economic model. For the main line, keep the reward fixed as own profit and compare architectures under the same objective.

### Rationale For Moving Beyond Reservoir-Only Agents

The baseline Reservoir Computing experiments show that richer memory alone is not sufficient to reproduce or improve on the symmetric Q-learning collusive benchmark. With pure own-profit maximization, the Oracle learns an asymmetric undercutting pattern:

```text
Oracle price < Victim price
Oracle profit < symmetric Q-vs-Q collusive profit
Victim profit falls much more than Oracle profit
market price falls relative to Q-vs-Q
```

This means the current Oracle is effective at destabilizing the adaptive competitor, but not yet effective at preserving high absolute margins while doing so.

This motivates the next research step: architectures that explicitly reason about opponent adaptation instead of only encoding longer market history.

Candidate directions:

```text
CFR-style regret accounting:
  test whether explicit counterfactual regret over price trajectories helps the Oracle avoid locally profitable but margin-destroying undercutting.

LOLA / opponent-learning awareness:
  test whether differentiating through the Victim's learning update helps the Oracle choose actions that shape future Victim behavior rather than only exploit the current state.
```

Defensible wording:

Reservoir-only agents provide evidence that memory and own-profit optimization can create asymmetric exploitation, but they do not yet solve the problem of maximizing absolute long-run margins against an adaptive Q-learning competitor. CFR and LOLA are therefore justified as the next mechanisms to test, not as assumed solutions.

## 2026-05-28 - Exploratory Asymmetry Reward Sweep

Source:

```text
results/asymmetry_coef_quick_sweep_mem512/
```

Purpose:

Test whether adding an explicit relative-profit component helps the Oracle improve absolute profit or merely increases destructive undercutting.

Reward:

```text
r_O = profit_O + asymmetry_coef * (profit_O - profit_V)
```

Configuration:

```text
Victim: adaptive tabular Calvano Q-learning
Oracle variants: actor_critic reservoir, DQN reservoir
seeds: 2 per setting
B: 64
H: 8
K: 15
total_steps: 20000
eval_steps: 2000
reservoir_dim: 512
asymmetry_coef: 0.0, 0.25, 0.5, 1.0
extra hyperparameter check: asymmetry_coef=0.5 with lr=0.0003
```

Aggregate results:

```text
mode          asym  lr       oracle_profit  victim_profit  profit_gap  oracle_price  market_price
AC            0.00  0.0010   0.3139         0.2274         +0.0865     1.5827        1.6407
AC            0.25  0.0010   0.3125         0.2093         +0.1033     1.5440        1.6214
AC            0.50  0.0010   0.3103         0.2004         +0.1099     1.5246        1.6117
AC            1.00  0.0010   0.3004         0.1745         +0.1259     1.4665        1.5826
DQN           0.00  0.0010   0.2985         0.2622         +0.0363     1.6583        1.6785
DQN           0.25  0.0010   0.3132         0.2183         +0.0949     1.5633        1.6311
DQN           0.50  0.0010   0.3043         0.1860         +0.1183     1.4922        1.5955
DQN           1.00  0.0010   0.2989         0.1752         +0.1238     1.4671        1.5829
DQN           0.50  0.0003   0.3047         0.1846         +0.1202     1.4892        1.5940
```

Interpretation:

The explicit asymmetry reward mostly increases undercutting. It reliably raises the profit gap by lowering the Oracle price and damaging the Victim, but it does not robustly improve the Oracle's absolute profit. For actor_critic, the pure own-profit reward remains best in this quick sweep. For DQN, a mild asymmetry coefficient (`0.25`) improves the short-run result relative to the unstable `0.0` DQN run, but stronger coefficients again reduce absolute profit.

Methodological conclusion:

Relative-profit reward shaping can amplify exploitation, but it is not a clean solution to the absolute-margin problem. It should be treated as an exploratory control, not as the main research mechanism.

## 2026-05-28 - DQN-JEPA Oracle Vs Adaptive Q-Learning Victim

Source:

```text
results/dqn_jepa_comparison/
```

Purpose:

Test whether JEPA-style auxiliary prediction changes DQN Oracle behavior against the adaptive tabular Q-learning Victim without changing the reward.

Reward:

```text
r_O = profit_O
asymmetry_coef = 0.0
```

Configuration:

```text
Victim: adaptive tabular Calvano Q-learning
Oracle variants: DQN reservoir, DQN-JEPA reservoir
seeds: 3 per mode
B: 64
H: 8
K: 15
total_steps: 20000
eval_steps: 2000
reservoir_dim: 512
```

Aggregate results:

```text
mode      oracle_profit  victim_profit  profit_gap  oracle_price  victim_price  market_price
DQN       0.3114         0.2094         +0.1020     1.5440        1.6987        1.6213
DQN-JEPA  0.3115         0.2392         +0.0722     1.6077        1.6987        1.6532
```

Interpretation:

DQN-JEPA does not materially increase Oracle absolute profit relative to DQN in this short run. However, it changes the mechanism: Oracle profit is roughly preserved while the Oracle price and market price are higher, and Victim profit is less damaged. This suggests JEPA may reduce destructive undercutting rather than amplify exploitation.

Compared to the Q-vs-Q collusive benchmark:

```text
Q-vs-Q firm profit = 0.3220
Q-vs-Q market price = 1.8026
```

DQN-JEPA still does not reach the symmetric collusive benchmark, but it is directionally more disciplined than vanilla DQN on price while maintaining similar Oracle profit.

Status:

Preliminary only: 3 seeds and 20000 steps. Needs a longer 10-seed run before making a strong claim.

## 2026-05-28 - DQN-JEPA 50k 10-Seed Follow-Up

Source:

```text
results/dqn_jepa_comparison_50k_10seeds/
```

Purpose:

Validate the preliminary DQN-JEPA signal on a longer and wider run.

Reward:

```text
r_O = profit_O
asymmetry_coef = 0.0
```

Configuration:

```text
Victim: adaptive tabular Calvano Q-learning
Oracle variants: DQN reservoir, DQN-JEPA reservoir
seeds: 10 per mode
B: 64
H: 8
K: 15
total_steps: 50000
eval_steps: 5000
reservoir_dim: 512
```

Aggregate results:

```text
mode      oracle_profit       victim_profit       profit_gap        oracle_price       victim_price       market_price
DQN       0.3060 +/- 0.0142   0.2394 +/- 0.0398   +0.0667 +/- 0.0525 1.6075 +/- 0.0859 1.6977 +/- 0.0003 1.6526 +/- 0.0430
DQN-JEPA  0.3014 +/- 0.0189   0.2477 +/- 0.0515   +0.0537 +/- 0.0666 1.6252 +/- 0.1120 1.6977 +/- 0.0002 1.6614 +/- 0.0560
```

Interpretation:

The longer 10-seed run weakens the earlier optimistic JEPA interpretation. DQN-JEPA raises the Oracle and market prices on average, but does not improve Oracle absolute profit. It also increases variance and sometimes pushes the Oracle into too-high-price regimes where the relative advantage disappears or becomes negative.

Compared to vanilla DQN:

```text
Oracle profit: lower with JEPA
Market price: higher with JEPA
Victim profit: higher with JEPA
Profit gap: lower with JEPA
Variance: higher with JEPA
```

Conclusion:

JEPA-style auxiliary prediction, in the current implementation and hyperparameters, does not solve the absolute-margin problem. It appears to reduce aggressive undercutting on average, but overcorrects in some seeds and fails to preserve Oracle profit. Treat this as a mixed/negative result rather than a fix.

Economic interpretation:

The 10-seed JEPA follow-up is consistent with a price-umbrella mechanism. Relative to vanilla DQN, DQN-JEPA raises the Oracle's average price and the average market price:

```text
DQN Oracle price = 1.6075
DQN-JEPA Oracle price = 1.6252
DQN market price = 1.6526
DQN-JEPA market price = 1.6614
```

This means JEPA reduces destructive undercutting at the price level. However, the Victim remains at a high price around `1.6977`; it does not move into a symmetric collusive regime with the Oracle. Because the Oracle undercuts less aggressively, the Victim recovers demand and profit:

```text
DQN Victim profit = 0.2394
DQN-JEPA Victim profit = 0.2477
```

The Oracle effectively creates a partial price umbrella: it raises its own price enough to make the market less destructive, but not in a way that forces the Victim into a favorable coordinated regime. The benefit of the higher market price is partly captured by the Victim, while the Oracle loses some demand share and its own average profit falls:

```text
DQN Oracle profit = 0.3060
DQN-JEPA Oracle profit = 0.3014
```

This suggests that the current JEPA auxiliary objective makes the Oracle more cautious, but not more strategically dominant. It helps the Oracle encode/predict market dynamics, yet it does not actively shape the Victim's learning process. In economic terms, the Oracle becomes less destructive but also less able to dictate market behavior; it risks becoming a price leader whose umbrella is exploited by the adaptive competitor.

Next architectural implication:

If JEPA is revisited, it needs either a better target definition tied to opponent regime prediction or a stability mechanism. The current generic next-latent prediction objective is not enough to produce robust strategic pricing improvement.

## 2026-05-29 - DQN-Regret Exploratory Comparison

Source:

```text
results/dqn_regret_comparison/
```

Purpose:

Test a CFR/regret-inspired auxiliary head against DQN and DQN-JEPA on a short exploratory run.

Reward:

```text
r_O = profit_O
asymmetry_coef = 0.0
```

Configuration:

```text
Victim: adaptive tabular Calvano Q-learning
Oracle variants: DQN, DQN-JEPA, DQN-Regret
seeds: 3 per mode
B: 64
H: 8
K: 15
total_steps: 20000
eval_steps: 2000
reservoir_dim: 512
regret_coef: 0.1
```

Aggregate results:

```text
mode        oracle_profit  victim_profit  profit_gap  oracle_price  victim_price  market_price
DQN         0.3114         0.2094         +0.1020     1.5440        1.6987        1.6213
DQN-JEPA    0.3115         0.2392         +0.0722     1.6077        1.6987        1.6532
DQN-Regret  0.3113         0.2116         +0.0996     1.5486        1.6987        1.6236
```

Interpretation:

In this short 3-seed run, DQN-Regret behaves much closer to vanilla DQN than to JEPA. It keeps the Oracle profit approximately unchanged, but does not raise market prices meaningfully and does not reduce destructive undercutting. The Victim remains heavily damaged, and the market price remains close to the vanilla DQN regime.

Compared to DQN-JEPA, the regret auxiliary has a higher profit gap but a lower market price. This indicates destructive exploitation rather than margin-preserving strategic improvement.

Conclusion:

The current regret-lite auxiliary head is not enough. It may be too local: it estimates one-step counterfactual payoffs but does not reason about how pricing choices change the Victim's future Q-learning trajectory. If this direction continues, the regret target should become multi-step or opponent-learning-aware rather than one-step payoff reconstruction.

## 2026-05-29 - Tabular CFR Vs Adaptive Q-Learning Victim

Source:

```text
results/tabular_cfr_comparison/
```

Purpose:

Test a clean non-neural regret-matching Oracle against the adaptive tabular Q-learning Victim. This provides a "table vs table" baseline and separates regret matching from DQN/replay/reservoir instability.

Reward:

```text
r_O = profit_O
asymmetry_coef = 0.0
```

Configuration:

```text
Victim: adaptive tabular Calvano Q-learning
Oracle variants:
  tabular_cfr_victim_last_action
  tabular_cfr_joint_last_action
Comparison modes:
  DQN
  DQN-JEPA
  DQN-Regret
seeds: 3 per mode
B: 64
H: 8
K: 15
total_steps: 20000
eval_steps: 2000
```

Aggregate results:

```text
mode                             oracle_profit  victim_profit  profit_gap  oracle_price  victim_price  market_price
tabular_cfr_victim_last_action   0.3078         0.2274         +0.0805     1.5815        1.6987        1.6401
tabular_cfr_joint_last_action    0.3080         0.2273         +0.0806     1.5815        1.6987        1.6401
DQN                              0.3013         0.2691         +0.0322     1.6714        1.6987        1.6851
DQN-JEPA                         0.3032         0.1985         +0.1047     1.5180        1.6987        1.6083
DQN-Regret                       0.3112         0.2456         +0.0656     1.6210        1.6987        1.6599
```

Interpretation:

Tabular CFR is extremely stable across seeds, but it does not recover the Q-vs-Q collusive benchmark. Both CFR state modes converge to nearly the same pricing pattern:

```text
Oracle price around 1.5815
Victim price around 1.6987
market price around 1.6401
Oracle profit around 0.308
```

This is a stable undercutting regime, not strategic dominance. The Oracle exploits the Victim and earns more than the Victim, but still earns less than the symmetric Q-vs-Q collusive profit (`0.3220`).

The result is methodologically useful because it removes neural-network instability from the story. Even clean one-step regret matching finds the same basic attractor: local counterfactual regret favors undercutting the high-price Q-learning Victim.

Conclusion:

Pure tabular one-step CFR/regret matching is not enough. The failure is not caused only by DQN approximation, reservoir representations, or replay instability. The deeper issue is that one-step counterfactual advantage is myopic in a learning opponent environment.

Next implication:

The next CFR-like direction must be multi-step or opponent-learning-aware. It should estimate how today's price changes the Victim's future Q-table and the future market state, not only how much profit an alternative price would have earned against the Victim's current action.

## 2026-05-29 - Tabular Multi-Step CFR Vs Adaptive Q-Learning Victim

Source:

```text
results/tabular_multi_cfr_comparison/
```

Purpose:

Test whether adding a tabular continuation-value estimate to CFR-style regret matching reduces one-step myopia.

Reward:

```text
r_O = profit_O
asymmetry_coef = 0.0
```

Configuration:

```text
Victim: adaptive tabular Calvano Q-learning
Oracle variants:
  tabular_cfr_joint_last_action
  tabular_multi_cfr_joint_last_action
Comparison modes:
  DQN-Regret
  DQN-JEPA
seeds: 3 per mode
B: 64
H: 8
K: 15
total_steps: 20000
eval_steps: 2000
cfr_state_mode: joint_last_action
```

Aggregate results:

```text
mode                              oracle_profit  victim_profit  profit_gap  oracle_price  victim_price  market_price
tabular_cfr_joint_last_action      0.3080         0.2273         +0.0806     1.5815        1.6987        1.6401
tabular_multi_cfr_joint_last_action 0.3099        0.2282         +0.0816     1.5837        1.6987        1.6412
DQN-Regret                         0.3112         0.2456         +0.0656     1.6210        1.6987        1.6599
DQN-JEPA                           0.3032         0.1985         +0.1047     1.5180        1.6987        1.6083
```

Interpretation:

Tabular multi-step CFR produces a small and highly stable improvement over one-step tabular CFR:

```text
Oracle profit: 0.3080 -> 0.3099
Oracle price: 1.5815 -> 1.5837
market price: 1.6401 -> 1.6412
```

This is directionally positive, but the effect is small. The outcome remains an undercutting regime, not a return to Q-vs-Q style high-price tacit collusion.

Compared to the Q-vs-Q benchmark:

```text
Q-vs-Q firm profit = 0.3220
Q-vs-Q market price = 1.8026
```

Tabular multi-step CFR is still far below the collusive benchmark on both Oracle profit and market price.

Conclusion:

Adding a continuation-value table partially reduces one-step myopia, but it does not solve the strategic problem. The improvement is weak positive evidence for multi-step reasoning, but not evidence of strategic dominance.

Next implication:

The remaining missing component is likely not just a longer value horizon, but opponent-learning awareness: the Oracle must model how its current action changes the Victim's future Q-table/policy. This points more directly toward LOLA-style or explicit Q-table-shaping experiments.

## 2026-05-29 - Tabular LOLA-Lite Vs Adaptive Q-Learning Victim

Source:

```text
results/tabular_lola_comparison/
results/tabular_lola_sweep/
```

Purpose:

Test a procedural opponent-aware lookahead Oracle that estimates future profit after the Victim's next adaptive response.

Reward:

```text
r_O = profit_O
asymmetry_coef = 0.0
```

Configuration:

```text
Victim: adaptive tabular Calvano Q-learning
Oracle: tabular_lola
seeds: 3 per setting
B: 64
H: 8
K: 15
total_steps: 20000
eval_steps: 2000
```

Main comparison:

```text
mode                         oracle_profit  victim_profit  profit_gap  oracle_price  victim_price  market_price
tabular_cfr                  0.3080         0.2273         +0.0806     1.5815        1.6987        1.6401
tabular_multi_cfr            0.3099         0.2282         +0.0816     1.5837        1.6987        1.6412
tabular_lola tau=0.05 gamma=0.95 0.2982     0.2370         +0.0611     1.6027        1.6987        1.6507
DQN-Regret                   0.3112         0.2456         +0.0656     1.6210        1.6987        1.6599
DQN-JEPA                     0.3032         0.1985         +0.1047     1.5180        1.6987        1.6083
```

LOLA-lite sweep:

```text
tau   gamma  oracle_profit  victim_profit  profit_gap  oracle_price  market_price  victim_pred_accuracy
0.03  0.90   0.3028         0.2231         +0.0797     1.5721        1.6354        0.1029
0.03  0.95   0.3028         0.2231         +0.0797     1.5721        1.6354        0.1028
0.05  0.90   0.2982         0.2370         +0.0611     1.6028        1.6507        0.1030
0.05  0.95   0.2982         0.2370         +0.0611     1.6027        1.6507        0.1030
0.10  0.90   0.2913         0.2546         +0.0367     1.6417        1.6702        0.1029
0.10  0.95   0.2913         0.2546         +0.0367     1.6417        1.6702        0.1029
```

Interpretation:

Current LOLA-lite does not solve the problem. Increasing `tau` makes the Oracle less aggressive and raises market prices, but Oracle profit falls. This is another price-umbrella pattern:

```text
higher Oracle price
higher Victim profit
lower Oracle profit
lower profit gap
```

The key diagnostic is victim prediction accuracy. It is around `0.103`, while random guessing over `K=15` would be about `0.0667`. This is only weakly above random and far too low for reliable opponent-aware lookahead.

Conclusion:

The current LOLA-lite mechanism is not actually shaping the Victim. It is mostly a soft lookahead policy with poor prediction of the Victim's realized action. Because the lookahead often reasons about the wrong Victim action, it tends to either soften undercutting into a price umbrella or lose Oracle profit.

Next implication:

Before full LOLA, the Oracle needs a better model of the Victim policy/update. A useful next experiment is not another price rule, but explicit Victim-model learning:

```text
predict Victim action distribution
predict Victim Q-update / next greedy action
condition LOLA-lite lookahead on that learned model
```

Alternatively, use the actual Victim Q-table directly in a deterministic planner to compute future Victim actions under candidate Oracle actions, rather than using a weak greedy prediction heuristic.

## 2026-05-29 - Tabular Multi-Step CFR Vs Adaptive Q-Learning Victim

Source:

```text
results/tabular_multi_cfr_comparison/
```

Purpose:

Test whether adding a tabular value estimate to CFR-style regret matching fixes the myopia of one-step counterfactual regret. The Oracle remains non-neural and tabular. The regret update uses:

```text
cf_value[a] = immediate_cf_profit[a] + cfr_gamma * value_table[next_state_cf(a)]
```

Reward:

```text
r_O = profit_O
asymmetry_coef = 0.0
```

Configuration:

```text
Victim: adaptive tabular Calvano Q-learning
Oracle variants:
  tabular_cfr_joint_last_action
  tabular_multi_cfr_joint_last_action
Comparison modes:
  DQN-Regret
  DQN-JEPA
seeds: 3 per mode
B: 64
H: 8
K: 15
total_steps: 20000
eval_steps: 2000
cfr_state_mode: joint_last_action
cfr_gamma: 0.95
cfr_value_lr: 0.1
```

Aggregate results:

```text
mode                                  oracle_profit  victim_profit  profit_gap  oracle_price  victim_price  market_price
tabular_cfr_joint_last_action         0.3080         0.2273         +0.0806     1.5815        1.6987        1.6401
tabular_multi_cfr_joint_last_action   0.3099         0.2282         +0.0816     1.5837        1.6987        1.6412
DQN-Regret                            0.3112         0.2456         +0.0656     1.6210        1.6987        1.6599
DQN-JEPA                              0.3032         0.1985         +0.1047     1.5180        1.6987        1.6083
```

Interpretation:

Tabular multi-step CFR improves slightly over one-step tabular CFR:

```text
Oracle profit: 0.3080 -> 0.3099
market price: 1.6401 -> 1.6412
Oracle price: 1.5815 -> 1.5837
```

The effect is positive but small. The multi-step value estimate does not move the system close to the Q-vs-Q collusive benchmark:

```text
Q-vs-Q market_price around 1.803
Q-vs-Q profit around 0.322 per firm
```

It also does not create a JEPA-style price umbrella failure: Oracle profit rises slightly rather than falling while market price rises. However, the magnitude is too small to claim a margin-preserving strategic improvement. The result is still best described as the same stable undercutting attractor with a modest value-estimation correction.

Conclusion:

Tabular multi-step CFR is a cleaner and slightly better baseline than one-step CFR, but it does not solve the absolute-margin problem. Multi-step value over market states is not enough by itself; the missing component is likely opponent-learning awareness, i.e. modeling how Oracle prices change the Victim's future Q-table rather than only the next market state value.

## 2026-05-29 - Tabular LOLA-Lite Vs Adaptive Q-Learning Victim

Source:

```text
results/tabular_lola_comparison/
results/tabular_lola_sweep/
```

Purpose:

Test a procedural LOLA-lite / Q-table-shaping Oracle that explicitly estimates how an Oracle price would change the Victim's next Q-learning update. This is not differentiable LOLA. It is a tabular lookahead baseline using the known Victim Q-learning update and Calvano payoff matrix.

Reward:

```text
r_O = profit_O
asymmetry_coef = 0.0
```

The Oracle used opponent-aware lookahead for action selection, not reward shaping:

```text
LOLA_value(a_O) =
    immediate_profit_O(a_O, predicted_a_V)
  + lola_gamma * estimated_future_profit_O_after_victim_update(a_O)
```

Configuration:

```text
Victim: adaptive tabular Calvano Q-learning
Oracle variants:
  tabular_cfr_joint_last_action
  tabular_multi_cfr_joint_last_action
  tabular_lola
Comparison modes:
  DQN-Regret
  DQN-JEPA
seeds: 3 per mode
B: 64
H: 8
K: 15
total_steps: 20000
eval_steps: 2000
LOLA base: lola_tau=0.05, lola_gamma=0.95, lola_epsilon=0.05
```

Aggregate comparison:

```text
mode                                  oracle_profit  victim_profit  profit_gap  oracle_price  victim_price  market_price  victim_pred_accuracy
tabular_cfr_joint_last_action         0.3080         0.2273         +0.0806     1.5815        1.6987        1.6401        n/a
tabular_multi_cfr_joint_last_action   0.3099         0.2282         +0.0816     1.5837        1.6987        1.6412        n/a
tabular_lola_tau0.05_gamma0.95        0.2982         0.2370         +0.0611     1.6027        1.6987        1.6507        0.1030
DQN-Regret                            0.3112         0.2456         +0.0656     1.6210        1.6987        1.6599        n/a
DQN-JEPA                              0.3032         0.1985         +0.1047     1.5180        1.6987        1.6083        n/a
```

LOLA sweep:

```text
tau   gamma  oracle_profit  victim_profit  profit_gap  oracle_price  market_price  victim_pred_accuracy
0.03  0.90   0.3028         0.2231         +0.0797     1.5721        1.6354        0.1029
0.03  0.95   0.3028         0.2231         +0.0797     1.5721        1.6354        0.1028
0.05  0.90   0.2982         0.2370         +0.0611     1.6028        1.6507        0.1030
0.05  0.95   0.2982         0.2370         +0.0611     1.6027        1.6507        0.1030
0.10  0.90   0.2913         0.2546         +0.0367     1.6417        1.6702        0.1029
0.10  0.95   0.2913         0.2546         +0.0367     1.6417        1.6702        0.1029
```

Interpretation:

LOLA-lite raises prices as the softmax temperature increases, but it lowers Oracle profit. This is a price-umbrella failure rather than a margin-preserving improvement:

```text
tau 0.03: Oracle profit around 0.303, market price around 1.635
tau 0.05: Oracle profit around 0.298, market price around 1.651
tau 0.10: Oracle profit around 0.291, market price around 1.670
```

The discount parameter (`lola_gamma=0.90` vs `0.95`) has almost no effect in this implementation. The likely reason is that the one-step modeled Victim update rarely changes the relevant future greedy action enough to alter the best-response term.

The low prediction accuracy is important. The Oracle lookahead uses the Victim's greedy Q action, while the actual Victim in training remains highly exploratory under the Calvano schedule. The measured match rate is only about `0.103`, so much of the opponent-aware calculation is conditioned on an action the Victim does not actually take.

Conclusion:

This LOLA-lite baseline does not solve the absolute-margin problem. It is opponent-aware in a procedural sense, but the awareness is too shallow and too dependent on inaccurate one-step greedy Victim prediction. The result is not destructive exploitation, but also not successful collusive-margin recovery. A stronger next version would need either a stochastic Victim policy model aligned with the actual exploration schedule or a deeper simulation of the Victim Q-table trajectory.

## 2026-05-29 - Tabular Model-Based LOLA Vs Adaptive Q-Learning Victim

Source:

```text
results/tabular_model_lola_smoke/
results/tabular_model_lola_exploratory/
```

Purpose:

Test `tabular_model_lola`, a procedural model-based analogue of LOLA for a non-black-box tabular Q-learning Victim. The Oracle directly reads the Victim Q-table, current state, learning parameters, and payoff matrix, then evaluates each candidate Oracle action by explicitly simulating the Victim's counterfactual Q-learning update.

Reward:

```text
r_O = profit_O
asymmetry_coef = 0.0
```

The reward stayed own-profit. Opponent-learning awareness entered only through action selection:

```text
model_lola_value(a_O) =
    E_{a_V ~ current_victim_policy} [
        profit_O(a_O, a_V)
      + model_lola_gamma * future_best_profit_O_after_modeled_victim_Q_update
    ]
```

Configuration:

```text
Victim: adaptive tabular Calvano Q-learning
Oracle variants:
  tabular_cfr_joint_last_action
  tabular_multi_cfr_joint_last_action
  tabular_lola_tau0.05_gamma0.95
  tabular_model_lola sweep
Comparison modes:
  DQN-Regret
  DQN-JEPA
seeds: 0,1,2
B: 64
H: 8
K: 15
total_steps: 20000
eval_every: 5000
eval_steps: 2000
model_lola_tau: 0.03, 0.05, 0.10
model_lola_gamma: 0.90, 0.95
victim_policy: epsilon_greedy
future_policy: epsilon_greedy
```

Aggregate comparison:

```text
mode                                  oracle_profit  victim_profit  profit_gap  oracle_price  victim_price  market_price  victim_pred_accuracy  model_lola_entropy
tabular_cfr_joint_last_action         0.3080         0.2273         +0.0806     1.5815        1.6987        1.6401        n/a                   n/a
tabular_multi_cfr_joint_last_action   0.3099         0.2282         +0.0816     1.5837        1.6987        1.6412        n/a                   n/a
tabular_lola_tau0.05_gamma0.95        0.2982         0.2370         +0.0611     1.6027        1.6987        1.6507        0.1364                n/a
tabular_model_lola_tau0.03_gamma0.90  0.3015         0.2399         +0.0615     1.6088        1.6987        1.6537        0.1372                2.4413
tabular_model_lola_tau0.03_gamma0.95  0.3015         0.2399         +0.0615     1.6088        1.6987        1.6537        0.1373                2.4413
tabular_model_lola_tau0.05_gamma0.90  0.2963         0.2496         +0.0467     1.6302        1.6987        1.6644        0.1366                2.5694
tabular_model_lola_tau0.05_gamma0.95  0.2963         0.2496         +0.0467     1.6302        1.6987        1.6644        0.1365                2.5694
tabular_model_lola_tau0.10_gamma0.90  0.2898         0.2618         +0.0280     1.6576        1.6987        1.6782        0.1367                2.6625
tabular_model_lola_tau0.10_gamma0.95  0.2898         0.2618         +0.0280     1.6576        1.6987        1.6782        0.1367                2.6625
DQN-Regret                            0.3112         0.2456         +0.0656     1.6210        1.6987        1.6599        n/a                   n/a
DQN-JEPA                              0.3032         0.1985         +0.1047     1.5180        1.6987        1.6083        n/a                   n/a
```

Interpretation:

The modeled Victim Q-update consistently raises market price above `tabular_lola` and both CFR baselines as temperature increases:

```text
tabular_multi_cfr market_price: 1.6412
tabular_lola tau=0.05 market_price: 1.6507
model_lola tau=0.03 market_price: 1.6537
model_lola tau=0.05 market_price: 1.6644
model_lola tau=0.10 market_price: 1.6782
```

However, Oracle profit falls as market price rises:

```text
model_lola tau=0.03 oracle_profit: 0.3015
model_lola tau=0.05 oracle_profit: 0.2963
model_lola tau=0.10 oracle_profit: 0.2898
```

This is a price umbrella failure: the Oracle induces a higher-price environment that benefits the adaptive Victim more than the Oracle. Profit gap shrinks from about `+0.0615` at tau `0.03` to `+0.0280` at tau `0.10`, while Victim profit rises from `0.2399` to `0.2618`.

The gamma sweep (`0.90` vs `0.95`) again has no meaningful effect. The one-step modeled Q update changes the action values enough to alter policy entropy/price selection, but not enough for discount depth to matter. Prediction accuracy is also low, around `0.137`, because the actual Victim remains epsilon-greedy and exploratory.

Conclusion:

Model-based opponent-learning awareness is promising only in the limited sense that it can move prices upward while preserving the own-profit reward definition. It does not improve over `tabular_multi_cfr` on Oracle profit, and it underperforms `dqn_regret` on profit despite higher market prices. Under the requested classification this run is a price umbrella failure, not successful collusive-margin recovery. The next step should be multi-step rollout of the Victim Q-learning process or a differentiable LOLA-style objective; one-step modeled Q update is insufficient.

## 2026-05-29 - Long-Run 100k+ Matrix Planning

Source:

```text
EXPERIMENT_MATRIX_100K_PLAN.md
```

Decision:

Short 20k-50k probes are not enough to distinguish transient undercutting from
converged strategic behavior. The next experimental matrix should move the main
conditions to at least `100000` steps, with neural and opponent-shaping blocks
at `150000` steps where feasible.

Planned blocks:

```text
1. Sanity checks and controls:
   - symmetric tabular Q-vs-Q, 10 seeds, 100k steps
   - static-Victim controls, 10 seeds, 100k steps

2. Tabular heterogeneity:
   - asymmetric alpha grid, 10 seeds, 100k steps
   - asymmetric delta grid, 10 seeds, 100k steps

3. Long-run architecture sweep:
   - reservoir AC, DQN, DQN-JEPA, DQN-Regret, tabular CFR
   - 10 seeds, 150k steps

4. Opponent shaping and control:
   - multi-step rollout/MPC Oracle with horizons L=5,12,25
   - state-space augmented DQN
   - 10 seeds, 150k steps
```

Purpose:

The matrix is designed to answer whether the observed failures are transient or
structural:

```text
undercutting persists        -> local exploitation is the stable attractor
prices rise but profit falls -> price umbrella
profit and market price rise -> promising opponent shaping
```

Important implementation note:

Some planned controls are not yet implemented, especially static Victim,
independent alpha/delta tabular heterogeneity, and augmented DQN state.
`tabular_rollout_lola` is now available as a runner mode, but it still needs
the planned long-run multi-seed validation before it should be interpreted as a
result.
