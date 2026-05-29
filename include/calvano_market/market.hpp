#pragma once

#include <cstdint>
#include <vector>

namespace calvano_market {

struct MarketConfig {
    int B = 0;
    int A = 2;
    int K = 0;
    int H = 0;
    std::vector<float> price_grid;
    std::vector<float> qualities;
    std::vector<float> costs;
    float outside_quality = 0.0F;
    float mu = 1.0F;
    float demand_scale = 1.0F;
    std::uint64_t random_seed = 0;
};

struct MarketState {
    explicit MarketState(MarketConfig cfg);

    MarketConfig config;
    std::uint64_t head = 0;

    std::vector<float> current_prices;
    std::vector<float> demand;
    std::vector<float> rewards;
    std::vector<float> market_share;
    std::vector<float> outside_share;
    std::vector<float> margins;
    std::vector<float> price_gap;
    std::vector<float> mean_price;
    std::vector<float> min_price;
    std::vector<float> max_price;
    std::vector<float> price_history_mirror;
};

struct NashResult {
    int actions[2] = {-1, -1};
    float prices[2] = {0.0F, 0.0F};
    float profits[2] = {0.0F, 0.0F};
};

struct MonopolyResult {
    int actions[2] = {-1, -1};
    float prices[2] = {0.0F, 0.0F};
    float total_profit = 0.0F;
    float per_firm_profit[2] = {0.0F, 0.0F};
};

void validate_config(const MarketConfig& config);
void reset_market_state(MarketState& state, std::uint64_t optional_seed, bool use_optional_seed);

void compute_logit_shares(
    int B,
    int A,
    const float* utilities,
    float outside_utility,
    float* market_share,
    float* outside_share);

void compute_rewards(
    int count,
    const float* prices,
    const float* costs,
    const float* demand,
    float* margins,
    float* rewards);

void update_price_history(MarketState& state);

void compute_logit_market_step(MarketState& state, const std::int64_t* actions_idx);

void compute_logit_profit_for_actions(
    const MarketConfig& one_market_config,
    const std::int64_t* actions_idx,
    float* profits_out);

std::vector<float> compute_static_profit_matrix(const MarketConfig& one_market_config);
NashResult find_discrete_nash_prices(const MarketConfig& one_market_config);
MonopolyResult find_joint_monopoly_prices(const MarketConfig& one_market_config);

}  // namespace calvano_market
