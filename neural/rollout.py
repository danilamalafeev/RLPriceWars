from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

import calvano_market_cpp as cm
from neural.observations import ObservationConfig, build_observation
from neural.reservoir import reservoir_observation, reservoir_update


@dataclass(frozen=True)
class RolloutConfig:
    T: int
    B: int
    H: int
    K: int
    use_reservoir: bool = False
    use_oracle_reservoir: bool = False
    use_victim_reservoir: bool = False
    device: str | torch.device | None = None


def sample_actions(logits: torch.Tensor, generator: torch.Generator | None = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    dist = torch.distributions.Categorical(logits=logits)
    if generator is None:
        actions = dist.sample()
    else:
        probs = torch.softmax(logits, dim=-1)
        actions = torch.multinomial(probs, num_samples=1, replacement=True, generator=generator).squeeze(-1)
    logp = dist.log_prob(actions)
    entropy = dist.entropy()
    return actions.to(torch.int64), logp, entropy


def _to_detached_tensor(x, device=None) -> torch.Tensor:
    return torch.as_tensor(np.asarray(x), dtype=torch.float32, device=device).detach()


def _market_snapshot(env, obs_config: ObservationConfig, time_step: int):
    return build_observation(
        cm.get_price_history_view(env),
        cm.get_current_prices(env),
        cm.get_rewards(env),
        cm.get_market_share(env),
        cm.get_outside_share(env),
        cm.get_margins(env),
        obs_config,
        time_step=time_step,
    )


def collect_duopoly_rollout(
    env,
    oracle_policy_fn,
    oracle_params,
    oracle_buffers,
    victim_policy_fn,
    victim_params,
    victim_buffers,
    obs_config: ObservationConfig,
    rollout_config: RolloutConfig,
    torch_generator: torch.Generator | None = None,
    oracle_value_fn=None,
    oracle_value_params=None,
    victim_value_fn=None,
    victim_value_params=None,
) -> dict[str, torch.Tensor]:
    if rollout_config.K <= 0 or rollout_config.T <= 0:
        raise ValueError("rollout_config T and K must be positive")

    device = rollout_config.device
    use_oracle_reservoir = rollout_config.use_reservoir or rollout_config.use_oracle_reservoir
    use_victim_reservoir = rollout_config.use_reservoir or rollout_config.use_victim_reservoir
    obs_oracle_list = []
    obs_victim_list = []
    values_list = []
    logp_list = []
    entropy_list = []
    actions_arr = torch.empty((rollout_config.T, rollout_config.B, 2), dtype=torch.int64, device=device)
    rewards_arr = torch.empty((rollout_config.T, rollout_config.B, 2), dtype=torch.float32, device=device)
    prices_arr = torch.empty((rollout_config.T, rollout_config.B, 2), dtype=torch.float32, device=device)
    share_arr = torch.empty((rollout_config.T, rollout_config.B, 2), dtype=torch.float32, device=device)
    margins_arr = torch.empty((rollout_config.T, rollout_config.B, 2), dtype=torch.float32, device=device)

    oracle_state = None
    victim_state = None
    oracle_res_h = None
    victim_res_h = None

    for t in range(rollout_config.T):
        features_t = _market_snapshot(env, obs_config, t).to(device)

        oracle_obs = features_t
        victim_obs = features_t
        if use_oracle_reservoir:
            if "reservoir" not in oracle_buffers:
                raise ValueError("oracle reservoir buffers required")
            if oracle_res_h is None:
                R_o = oracle_buffers["reservoir"]["W_res"].shape[0]
                oracle_res_h = torch.zeros(rollout_config.B, R_o, dtype=torch.float32, device=device)
            oracle_res_h = reservoir_update(features_t, oracle_res_h, oracle_buffers["reservoir"])
            oracle_obs = reservoir_observation(features_t, oracle_res_h)

        if use_victim_reservoir:
            if "reservoir" not in victim_buffers:
                raise ValueError("victim reservoir buffers required")
            if victim_res_h is None:
                R_v = victim_buffers["reservoir"]["W_res"].shape[0]
                victim_res_h = torch.zeros(rollout_config.B, R_v, dtype=torch.float32, device=device)
            victim_res_h = reservoir_update(features_t, victim_res_h, victim_buffers["reservoir"])
            victim_obs = reservoir_observation(features_t, victim_res_h)

        logits_o, oracle_state = oracle_policy_fn(oracle_params, oracle_buffers, oracle_obs, oracle_state)
        logits_v, victim_state = victim_policy_fn(victim_params, victim_buffers, victim_obs, victim_state)
        if oracle_value_fn is not None and victim_value_fn is not None:
            value_o = oracle_value_fn(oracle_value_params, oracle_obs)
            value_v = victim_value_fn(victim_value_params, victim_obs)
            values_list.append(torch.stack([value_o, value_v], dim=1))
        action_o, logp_o, entropy_o = sample_actions(logits_o, torch_generator)
        action_v, logp_v, entropy_v = sample_actions(logits_v, torch_generator)

        actions = torch.stack([action_o.detach(), action_v.detach()], dim=1)
        cm.step(env, actions.cpu().numpy().astype(np.int64, copy=False))

        obs_oracle_list.append(oracle_obs)
        obs_victim_list.append(victim_obs)
        logp_list.append(torch.stack([logp_o, logp_v], dim=1))
        entropy_list.append(torch.stack([entropy_o, entropy_v], dim=1))

        actions_arr[t] = actions
        rewards_arr[t] = _to_detached_tensor(cm.get_rewards(env), device)
        prices_arr[t] = _to_detached_tensor(cm.get_current_prices(env), device)
        share_arr[t] = _to_detached_tensor(cm.get_market_share(env), device)
        margins_arr[t] = _to_detached_tensor(cm.get_margins(env), device)

    rollout = {
        "obs": torch.stack(obs_oracle_list, dim=0),
        "obs_oracle": torch.stack(obs_oracle_list, dim=0),
        "obs_victim": torch.stack(obs_victim_list, dim=0),
        "logp": torch.stack(logp_list, dim=0),
        "entropy": torch.stack(entropy_list, dim=0),
        "actions": actions_arr.detach(),
        "rewards": rewards_arr.detach(),
        "prices": prices_arr.detach(),
        "market_share": share_arr.detach(),
        "margins": margins_arr.detach(),
    }
    if values_list:
        rollout["values"] = torch.stack(values_list, dim=0)
    return rollout
