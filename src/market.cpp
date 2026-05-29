#include "calvano_market/market.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

namespace calvano_market {
namespace {

constexpr float kTieTolerance = 1.0e-6F;

inline int ba_index(int b, int a, int A) {
    return b * A + a;
}

inline int history_index(int b, int t, int a, int H, int A) {
    return (b * (2 * H) + t) * A + a;
}

MarketConfig as_one_market_config(const MarketConfig& config) {
    MarketConfig one = config;
    one.B = 1;
    if (static_cast<int>(one.qualities.size()) != one.A) {
        one.qualities.assign(config.qualities.begin(), config.qualities.begin() + one.A);
    }
    if (static_cast<int>(one.costs.size()) != one.A) {
        one.costs.assign(config.costs.begin(), config.costs.begin() + one.A);
    }
    validate_config(one);
    return one;
}

void compute_logit_profit_for_actions_unchecked(
    const MarketConfig& config,
    const std::int64_t* actions_idx,
    float* profits_out) {
    const float outside_utility = config.outside_quality / config.mu;
    float max_utility = outside_utility;

    for (int a = 0; a < config.A; ++a) {
        const std::int64_t action = actions_idx[a];
        if (action < 0 || action >= config.K) {
            throw std::out_of_range("actions_idx contains an action outside [0, K)");
        }
        const float price = config.price_grid[static_cast<int>(action)];
        const float utility = (config.qualities[a] - price) / config.mu;
        max_utility = std::max(max_utility, utility);
    }

    float denom = std::exp(outside_utility - max_utility);
    for (int a = 0; a < config.A; ++a) {
        const std::int64_t action = actions_idx[a];
        const float price = config.price_grid[static_cast<int>(action)];
        const float utility = (config.qualities[a] - price) / config.mu;
        profits_out[a] = std::exp(utility - max_utility);
        denom += profits_out[a];
    }

    const float inv_denom = 1.0F / denom;
    for (int a = 0; a < config.A; ++a) {
        const std::int64_t action = actions_idx[a];
        const float price = config.price_grid[static_cast<int>(action)];
        const float demand = config.demand_scale * profits_out[a] * inv_denom;
        profits_out[a] = (price - config.costs[a]) * demand;
    }
}

}  // namespace

MarketState::MarketState(MarketConfig cfg) : config(std::move(cfg)) {
    validate_config(config);
    const int BA = config.B * config.A;
    current_prices.assign(BA, 0.0F);
    demand.assign(BA, 0.0F);
    rewards.assign(BA, 0.0F);
    market_share.assign(BA, 0.0F);
    outside_share.assign(config.B, 0.0F);
    margins.assign(BA, 0.0F);
    price_gap.assign(config.B, 0.0F);
    mean_price.assign(config.B, 0.0F);
    min_price.assign(config.B, 0.0F);
    max_price.assign(config.B, 0.0F);
    price_history_mirror.assign(config.B * 2 * config.H * config.A, 0.0F);
}

void validate_config(const MarketConfig& config) {
    if (config.B <= 0) {
        throw std::invalid_argument("B must be positive");
    }
    if (config.A <= 0) {
        throw std::invalid_argument("A must be positive");
    }
    if (config.K <= 0) {
        throw std::invalid_argument("K must be positive");
    }
    if (config.H <= 0) {
        throw std::invalid_argument("H must be positive");
    }
    if (config.mu <= 0.0F || !std::isfinite(config.mu)) {
        throw std::invalid_argument("mu must be finite and > 0");
    }
    if (static_cast<int>(config.price_grid.size()) != config.K) {
        throw std::invalid_argument("price_grid must have shape [K]");
    }
    if (static_cast<int>(config.qualities.size()) != config.B * config.A) {
        throw std::invalid_argument("qualities must be expanded to shape [B, A]");
    }
    if (static_cast<int>(config.costs.size()) != config.B * config.A) {
        throw std::invalid_argument("costs must be expanded to shape [B, A]");
    }
}

void reset_market_state(MarketState& state, std::uint64_t optional_seed, bool use_optional_seed) {
    if (use_optional_seed) {
        state.config.random_seed = optional_seed;
    }
    state.head = 0;
    std::fill(state.current_prices.begin(), state.current_prices.end(), 0.0F);
    std::fill(state.demand.begin(), state.demand.end(), 0.0F);
    std::fill(state.rewards.begin(), state.rewards.end(), 0.0F);
    std::fill(state.market_share.begin(), state.market_share.end(), 0.0F);
    std::fill(state.outside_share.begin(), state.outside_share.end(), 0.0F);
    std::fill(state.margins.begin(), state.margins.end(), 0.0F);
    std::fill(state.price_gap.begin(), state.price_gap.end(), 0.0F);
    std::fill(state.mean_price.begin(), state.mean_price.end(), 0.0F);
    std::fill(state.min_price.begin(), state.min_price.end(), 0.0F);
    std::fill(state.max_price.begin(), state.max_price.end(), 0.0F);
    std::fill(state.price_history_mirror.begin(), state.price_history_mirror.end(), 0.0F);
}

void compute_logit_shares(
    int B,
    int A,
    const float* utilities,
    float outside_utility,
    float* market_share,
    float* outside_share) {
    for (int b = 0; b < B; ++b) {
        float max_utility = outside_utility;
        for (int a = 0; a < A; ++a) {
            max_utility = std::max(max_utility, utilities[ba_index(b, a, A)]);
        }

        float denom = std::exp(outside_utility - max_utility);
        for (int a = 0; a < A; ++a) {
            const int idx = ba_index(b, a, A);
            const float exp_utility = std::exp(utilities[idx] - max_utility);
            market_share[idx] = exp_utility;
            denom += exp_utility;
        }

        const float inv_denom = 1.0F / denom;
        for (int a = 0; a < A; ++a) {
            const int idx = ba_index(b, a, A);
            market_share[idx] *= inv_denom;
        }
        outside_share[b] = std::exp(outside_utility - max_utility) * inv_denom;
    }
}

void compute_rewards(
    int count,
    const float* prices,
    const float* costs,
    const float* demand,
    float* margins,
    float* rewards) {
    for (int idx = 0; idx < count; ++idx) {
        margins[idx] = prices[idx] - costs[idx];
        rewards[idx] = margins[idx] * demand[idx];
    }
}

void update_price_history(MarketState& state) {
    const MarketConfig& config = state.config;
    const int write_idx = static_cast<int>(state.head % static_cast<std::uint64_t>(config.H));

    for (int b = 0; b < config.B; ++b) {
        for (int a = 0; a < config.A; ++a) {
            const float price = state.current_prices[ba_index(b, a, config.A)];
            state.price_history_mirror[history_index(b, write_idx, a, config.H, config.A)] = price;
            state.price_history_mirror[history_index(b, write_idx + config.H, a, config.H, config.A)] = price;
        }
    }
    ++state.head;
}

void compute_logit_market_step(MarketState& state, const std::int64_t* actions_idx) {
    const MarketConfig& config = state.config;
    const float outside_utility = config.outside_quality / config.mu;

    for (int b = 0; b < config.B; ++b) {
        float max_utility = outside_utility;
        float min_price = std::numeric_limits<float>::infinity();
        float max_price = -std::numeric_limits<float>::infinity();
        float price_sum = 0.0F;

        for (int a = 0; a < config.A; ++a) {
            const int idx = ba_index(b, a, config.A);
            const std::int64_t action = actions_idx[idx];
            if (action < 0 || action >= config.K) {
                throw std::out_of_range("actions_idx contains an action outside [0, K)");
            }

            const float price = config.price_grid[static_cast<int>(action)];
            state.current_prices[idx] = price;

            min_price = std::min(min_price, price);
            max_price = std::max(max_price, price);
            price_sum += price;

            const float utility = (config.qualities[idx] - price) / config.mu;
            max_utility = std::max(max_utility, utility);
        }

        float denom = std::exp(outside_utility - max_utility);
        for (int a = 0; a < config.A; ++a) {
            const int idx = ba_index(b, a, config.A);
            const float utility = (config.qualities[idx] - state.current_prices[idx]) / config.mu;
            const float exp_utility = std::exp(utility - max_utility);
            state.market_share[idx] = exp_utility;
            denom += exp_utility;
        }

        const float inv_denom = 1.0F / denom;
        for (int a = 0; a < config.A; ++a) {
            const int idx = ba_index(b, a, config.A);
            state.market_share[idx] *= inv_denom;
            state.demand[idx] = config.demand_scale * state.market_share[idx];
        }

        state.outside_share[b] = std::exp(outside_utility - max_utility) * inv_denom;
        state.min_price[b] = min_price;
        state.max_price[b] = max_price;
        state.mean_price[b] = price_sum / static_cast<float>(config.A);
        state.price_gap[b] = max_price - min_price;
    }

    compute_rewards(
        config.B * config.A,
        state.current_prices.data(),
        config.costs.data(),
        state.demand.data(),
        state.margins.data(),
        state.rewards.data());

    update_price_history(state);
}

void compute_logit_profit_for_actions(
    const MarketConfig& one_market_config,
    const std::int64_t* actions_idx,
    float* profits_out) {
    const MarketConfig config = as_one_market_config(one_market_config);
    compute_logit_profit_for_actions_unchecked(config, actions_idx, profits_out);
}

std::vector<float> compute_static_profit_matrix(const MarketConfig& one_market_config) {
    MarketConfig config = as_one_market_config(one_market_config);
    if (config.A != 2) {
        throw std::invalid_argument("compute_static_profit_matrix requires A == 2");
    }

    std::vector<float> matrix(config.K * config.K * 2, 0.0F);
    std::int64_t actions[2] = {0, 0};
    float profits[2] = {0.0F, 0.0F};

    for (int i = 0; i < config.K; ++i) {
        for (int j = 0; j < config.K; ++j) {
            actions[0] = i;
            actions[1] = j;
            compute_logit_profit_for_actions_unchecked(config, actions, profits);
            const int out = (i * config.K + j) * 2;
            matrix[out] = profits[0];
            matrix[out + 1] = profits[1];
        }
    }

    return matrix;
}

NashResult find_discrete_nash_prices(const MarketConfig& one_market_config) {
    MarketConfig config = as_one_market_config(one_market_config);
    if (config.A != 2) {
        throw std::invalid_argument("find_discrete_nash_prices requires A == 2");
    }

    const std::vector<float> profits = compute_static_profit_matrix(config);

    for (int i = 0; i < config.K; ++i) {
        for (int j = 0; j < config.K; ++j) {
            const int profile = (i * config.K + j) * 2;
            const float p0 = profits[profile];
            const float p1 = profits[profile + 1];
            bool firm0_best = true;
            bool firm1_best = true;

            for (int alt = 0; alt < config.K; ++alt) {
                const float alt0 = profits[(alt * config.K + j) * 2];
                const float alt1 = profits[(i * config.K + alt) * 2 + 1];
                if (alt0 > p0 + kTieTolerance) {
                    firm0_best = false;
                }
                if (alt1 > p1 + kTieTolerance) {
                    firm1_best = false;
                }
            }

            if (firm0_best && firm1_best) {
                NashResult result;
                result.actions[0] = i;
                result.actions[1] = j;
                result.prices[0] = config.price_grid[i];
                result.prices[1] = config.price_grid[j];
                result.profits[0] = p0;
                result.profits[1] = p1;
                return result;
            }
        }
    }

    throw std::runtime_error("no pure-strategy discrete Nash equilibrium found");
}

MonopolyResult find_joint_monopoly_prices(const MarketConfig& one_market_config) {
    MarketConfig config = as_one_market_config(one_market_config);
    if (config.A != 2) {
        throw std::invalid_argument("find_joint_monopoly_prices requires A == 2");
    }

    const std::vector<float> profits = compute_static_profit_matrix(config);
    float best_total = -std::numeric_limits<float>::infinity();
    MonopolyResult result;

    for (int i = 0; i < config.K; ++i) {
        for (int j = 0; j < config.K; ++j) {
            const int profile = (i * config.K + j) * 2;
            const float total = profits[profile] + profits[profile + 1];
            if (total > best_total) {
                best_total = total;
                result.actions[0] = i;
                result.actions[1] = j;
                result.prices[0] = config.price_grid[i];
                result.prices[1] = config.price_grid[j];
                result.total_profit = total;
                result.per_firm_profit[0] = profits[profile];
                result.per_firm_profit[1] = profits[profile + 1];
            }
        }
    }

    return result;
}

}  // namespace calvano_market
