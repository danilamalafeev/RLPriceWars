#include "calvano_market/market.hpp"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace py = pybind11;

namespace {

using calvano_market::MarketConfig;
using calvano_market::MarketState;
using calvano_market::MonopolyResult;
using calvano_market::NashResult;
using EnvHandle = std::shared_ptr<calvano_market::MarketState>;

int get_required_int(const py::dict& dict, const char* key) {
    if (!dict.contains(key)) {
        throw std::invalid_argument(std::string("missing required config key: ") + key);
    }
    return py::cast<int>(dict[key]);
}

float get_optional_float(const py::dict& dict, const char* key, float default_value) {
    if (!dict.contains(key)) {
        return default_value;
    }
    return py::cast<float>(dict[key]);
}

std::uint64_t get_optional_u64(const py::dict& dict, const char* key, std::uint64_t default_value) {
    if (!dict.contains(key)) {
        return default_value;
    }
    return py::cast<std::uint64_t>(dict[key]);
}

std::vector<float> read_price_grid(const py::dict& dict, int K) {
    if (!dict.contains("price_grid")) {
        throw std::invalid_argument("missing required config key: price_grid");
    }
    py::array_t<float, py::array::c_style | py::array::forcecast> arr(dict["price_grid"]);
    py::buffer_info info = arr.request();
    if (info.ndim != 1 || info.shape[0] != K) {
        throw std::invalid_argument("price_grid must have shape [K]");
    }
    const float* ptr = static_cast<const float*>(info.ptr);
    return std::vector<float>(ptr, ptr + K);
}

std::vector<float> read_ba_array(const py::dict& dict, const char* key, int B, int A) {
    if (!dict.contains(key)) {
        throw std::invalid_argument(std::string("missing required config key: ") + key);
    }

    py::array_t<float, py::array::c_style | py::array::forcecast> arr(dict[key]);
    py::buffer_info info = arr.request();
    const float* ptr = static_cast<const float*>(info.ptr);
    std::vector<float> out(B * A, 0.0F);

    if (info.ndim == 1 && info.shape[0] == A) {
        for (int b = 0; b < B; ++b) {
            for (int a = 0; a < A; ++a) {
                out[b * A + a] = ptr[a];
            }
        }
        return out;
    }

    if (info.ndim == 2 && info.shape[0] == B && info.shape[1] == A) {
        std::copy(ptr, ptr + B * A, out.begin());
        return out;
    }

    std::ostringstream oss;
    oss << key << " must have shape [A] or [B, A]";
    throw std::invalid_argument(oss.str());
}

MarketConfig parse_config(const py::dict& dict) {
    MarketConfig config;
    config.B = get_required_int(dict, "B");
    config.A = dict.contains("A") ? py::cast<int>(dict["A"]) : 2;
    config.K = get_required_int(dict, "K");
    config.H = get_required_int(dict, "H");
    config.price_grid = read_price_grid(dict, config.K);
    config.qualities = read_ba_array(dict, "qualities", config.B, config.A);
    config.costs = read_ba_array(dict, "costs", config.B, config.A);
    config.outside_quality = get_optional_float(dict, "outside_quality", 0.0F);
    config.mu = get_optional_float(dict, "mu", 1.0F);
    config.demand_scale = get_optional_float(dict, "demand_scale", 1.0F);
    config.random_seed = get_optional_u64(dict, "random_seed", 0);
    calvano_market::validate_config(config);
    return config;
}

py::array_t<float> view_1d(const EnvHandle& env, std::vector<float>& data) {
    py::object base = py::cast(env);
    return py::array_t<float>(
        {env->config.B},
        {static_cast<py::ssize_t>(sizeof(float))},
        data.data(),
        base);
}

py::array_t<float> view_ba(const EnvHandle& env, std::vector<float>& data) {
    py::object base = py::cast(env);
    return py::array_t<float>(
        {env->config.B, env->config.A},
        {
            static_cast<py::ssize_t>(env->config.A * sizeof(float)),
            static_cast<py::ssize_t>(sizeof(float)),
        },
        data.data(),
        base);
}

py::array_t<float> get_price_history_view(const EnvHandle& env) {
    const int B = env->config.B;
    const int H = env->config.H;
    const int A = env->config.A;
    const int start = static_cast<int>(env->head % static_cast<std::uint64_t>(H));
    float* ptr = env->price_history_mirror.data() + start * A;
    py::object base = py::cast(env);

    return py::array_t<float>(
        {B, H, A},
        {
            static_cast<py::ssize_t>(2 * H * A * sizeof(float)),
            static_cast<py::ssize_t>(A * sizeof(float)),
            static_cast<py::ssize_t>(sizeof(float)),
        },
        ptr,
        base);
}

void validate_actions_shape(const MarketState& env, const py::buffer_info& info) {
    if (info.ndim != 2 || info.shape[0] != env.config.B || info.shape[1] != env.config.A) {
        std::ostringstream oss;
        oss << "actions_idx must have shape [" << env.config.B << ", " << env.config.A << "]";
        throw std::invalid_argument(oss.str());
    }
}

py::array_t<std::int64_t> int64_array_1d(const int values[2]) {
    py::array_t<std::int64_t> arr({2});
    auto out = arr.mutable_unchecked<1>();
    out(0) = values[0];
    out(1) = values[1];
    return arr;
}

py::array_t<float> float_array_1d2(const float values[2]) {
    py::array_t<float> arr({2});
    auto out = arr.mutable_unchecked<1>();
    out(0) = values[0];
    out(1) = values[1];
    return arr;
}

}  // namespace

PYBIND11_MODULE(calvano_market_cpp, m) {
    m.doc() = "Vectorized Calvano-style differentiated Bertrand market kernel";

    py::class_<MarketState, EnvHandle>(m, "MarketState")
        .def_property_readonly("B", [](const MarketState& s) { return s.config.B; })
        .def_property_readonly("A", [](const MarketState& s) { return s.config.A; })
        .def_property_readonly("K", [](const MarketState& s) { return s.config.K; })
        .def_property_readonly("H", [](const MarketState& s) { return s.config.H; })
        .def_property_readonly("head", [](const MarketState& s) { return s.head; });

    m.def("create_env", [](const py::dict& config_dict) {
        return std::make_shared<MarketState>(parse_config(config_dict));
    }, py::arg("config_dict"));

    m.def("reset", [](const EnvHandle& env, py::object optional_seed) {
        if (optional_seed.is_none()) {
            calvano_market::reset_market_state(*env, 0, false);
        } else {
            calvano_market::reset_market_state(*env, py::cast<std::uint64_t>(optional_seed), true);
        }
    }, py::arg("env"), py::arg("optional_seed") = py::none());

    m.def("step", [](const EnvHandle& env, py::array_t<std::int64_t, py::array::c_style> actions_idx) {
        py::buffer_info info = actions_idx.request();
        validate_actions_shape(*env, info);
        calvano_market::compute_logit_market_step(*env, static_cast<const std::int64_t*>(info.ptr));
    }, py::arg("env"), py::arg("actions_idx_numpy"));

    m.def("get_current_prices", [](const EnvHandle& env) { return view_ba(env, env->current_prices); }, py::arg("env"));
    m.def("get_demand", [](const EnvHandle& env) { return view_ba(env, env->demand); }, py::arg("env"));
    m.def("get_rewards", [](const EnvHandle& env) { return view_ba(env, env->rewards); }, py::arg("env"));
    m.def("get_market_share", [](const EnvHandle& env) { return view_ba(env, env->market_share); }, py::arg("env"));
    m.def("get_outside_share", [](const EnvHandle& env) { return view_1d(env, env->outside_share); }, py::arg("env"));
    m.def("get_margins", [](const EnvHandle& env) { return view_ba(env, env->margins); }, py::arg("env"));
    m.def("get_price_gap", [](const EnvHandle& env) { return view_1d(env, env->price_gap); }, py::arg("env"));
    m.def("get_mean_price", [](const EnvHandle& env) { return view_1d(env, env->mean_price); }, py::arg("env"));
    m.def("get_min_price", [](const EnvHandle& env) { return view_1d(env, env->min_price); }, py::arg("env"));
    m.def("get_max_price", [](const EnvHandle& env) { return view_1d(env, env->max_price); }, py::arg("env"));
    m.def("get_price_history_view", &get_price_history_view, py::arg("env"));

    m.def("compute_static_profit_matrix", [](const py::dict& config_dict) {
        MarketConfig config = parse_config(config_dict);
        std::vector<float> matrix = calvano_market::compute_static_profit_matrix(config);
        py::array_t<float> arr({config.K, config.K, 2});
        std::copy(matrix.begin(), matrix.end(), static_cast<float*>(arr.request().ptr));
        return arr;
    }, py::arg("config_dict"));

    m.def("find_discrete_nash_prices", [](const py::dict& config_dict) {
        MarketConfig config = parse_config(config_dict);
        NashResult result = calvano_market::find_discrete_nash_prices(config);
        py::dict out;
        out["actions"] = int64_array_1d(result.actions);
        out["prices"] = float_array_1d2(result.prices);
        out["profits"] = float_array_1d2(result.profits);
        return out;
    }, py::arg("config_dict"));

    m.def("find_joint_monopoly_prices", [](const py::dict& config_dict) {
        MarketConfig config = parse_config(config_dict);
        MonopolyResult result = calvano_market::find_joint_monopoly_prices(config);
        py::dict out;
        out["actions"] = int64_array_1d(result.actions);
        out["prices"] = float_array_1d2(result.prices);
        out["total_profit"] = result.total_profit;
        out["per_firm_profit"] = float_array_1d2(result.per_firm_profit);
        return out;
    }, py::arg("config_dict"));
}
