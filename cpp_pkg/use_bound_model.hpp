#pragma once

#include <algorithm>
#include <cmath>
#include <functional>
#include <stdexcept>
#include <string>

#include "offline/models/bound/mix_all_bound_offline.hpp"
#include "offline/models/bound/mix_gen_bound_offline.hpp"
#include "offline/models/bound/mix_real_bound_offline.hpp"
#include "offline/models/bound/randomD_bound_offline.hpp"
#include "offline/models/bound/randomE_bound_offline.hpp"
#include "offline/models/bound/randomG_bound_offline.hpp"
#include "offline/models/bound/randomH_bound_offline.hpp"
#include "offline/models/bound/randomT_bound_offline.hpp"
#include "offline/models/bound/RF_bound_offline.hpp"
#include "offline/models/bound/RD_bound_offline.hpp"

#include "online/models/bound/mix_all_bound_online.hpp"
#include "online/models/bound/mix_gen_bound_online.hpp"
#include "online/models/bound/mix_real_bound_online.hpp"
#include "online/models/bound/randomD_bound_online.hpp"
#include "online/models/bound/randomE_bound_online.hpp"
#include "online/models/bound/randomG_bound_online.hpp"
#include "online/models/bound/randomH_bound_online.hpp"
#include "online/models/bound/randomT_bound_online.hpp"
#include "online/models/bound/RF_bound_online.hpp"
#include "online/models/bound/RD_bound_online.hpp"
#include "online/models/bound/blank.hpp"

struct BoundState {
    double delta_mu = 0.0;
};

struct BoundPrediction {
    double log_max_dist = 0.0;
    double max_dist = 0.0;
    double bound = 0.0;
};

struct BoundModel {
    using InferFn = std::function<BoundPrediction(const BoundState&, int, int, int, int)>;
    using UpdateFn = std::function<void(BoundState&, double, double)>;

    bool is_online = false;
    InferFn infer_fn;
    UpdateFn update_fn;

    static int compute_k(int n) {
        return std::max(1, static_cast<int>(std::floor(std::pow(std::log(std::max(n, 2)), 1.0 / 3.0))));
    }

    static int compute_t(int n) {
        return std::max(1, static_cast<int>(std::floor(std::pow(std::log(std::max(n, 2)), 2.0 / 3.0))));
    }

    static int compute_l(int n) {
        const int t = compute_t(n);
        return std::max(1, static_cast<int>(std::ceil(std::log2(std::max(n, 2)) / std::max(1, t))));
    }

    BoundPrediction infer(const BoundState& st, int n, int k, int t, int l) const {
        if (!infer_fn) {
            throw std::logic_error("BoundModel infer_fn is not set.");
        }
        return infer_fn(st, n, k, t, l);
    }

    BoundPrediction infer(const BoundState& st, int n) const {
        return infer(st, n, compute_k(n), compute_t(n), compute_l(n));
    }

    double predict_log_max_dist(const BoundState& st, int n, int k, int t, int l) const {
        return infer(st, n, k, t, l).log_max_dist;
    }

    double predict_max_dist(const BoundState& st, int n, int k, int t, int l) const {
        return infer(st, n, k, t, l).max_dist;
    }

    double predict_bound(const BoundState& st, int n, int k, int t, int l) const {
        return infer(st, n, k, t, l).bound;
    }

    double predict(const BoundState& st, int n, int k, int t, int l) const {
        return predict_bound(st, n, k, t, l);
    }

    double predict_log_max_dist(const BoundState& st, int n) const {
        return infer(st, n).log_max_dist;
    }

    double predict_max_dist(const BoundState& st, int n) const {
        return infer(st, n).max_dist;
    }

    double predict_bound(const BoundState& st, int n) const {
        return infer(st, n).bound;
    }

    double predict(const BoundState& st, int n) const {
        return predict_bound(st, n);
    }

    void update(BoundState& st, double predicted_log_mean, double observed_max_dist) const {
        if (update_fn) {
            update_fn(st, predicted_log_mean, observed_max_dist);
        }
    }
};

#define MAKE_BOUND_OFFLINE_ENTRY(graph_literal, ns_name)                                      \
    if (graph_name == graph_literal) {                                                        \
        return BoundModel{                                                                    \
            false,                                                                            \
            [](const BoundState&, int n, int k, int t, int l) -> BoundPrediction {           \
                static ns_name::Model model;                                                  \
                const double log_with_quantile = model.predict_log_max_dist(n, k, t, l);     \
                const double log_mean = log_with_quantile - model.z_value * model.sigma;      \
                const double max_dist = model.predict_max_dist(n, k, t, l);                   \
                const double bound = model.predict_bound(n, k, t, l);                         \
                return BoundPrediction{log_mean, max_dist, bound};                            \
            },                                                                                \
            [](BoundState&, double, double) {}                                                \
        };                                                                                    \
    }

#define MAKE_BOUND_ONLINE_ENTRY(graph_literal, ns_name)                                       \
    if (graph_name == graph_literal) {                                                        \
        return BoundModel{                                                                    \
            true,                                                                             \
            [](const BoundState& st, int n, int k, int t, int l) -> BoundPrediction {        \
                static ns_name::Model model;                                                  \
                ns_name::State native_st;                                                     \
                native_st.delta_mu = st.delta_mu;                                             \
                const auto pred = model.infer(native_st, n, k, t, l);                         \
                return BoundPrediction{pred.log_max_dist, pred.max_dist, pred.bound};         \
            },                                                                                \
            [](BoundState& st, double predicted_log_mean, double observed_max_dist) {         \
                static ns_name::Model model;                                                  \
                ns_name::State native_st;                                                     \
                native_st.delta_mu = st.delta_mu;                                             \
                model.update(native_st, predicted_log_mean, observed_max_dist);               \
                st.delta_mu = native_st.delta_mu;                                             \
            }                                                                                 \
        };                                                                                    \
    }

inline BoundModel get_bound_model(const std::string& graph_name, const std::string& mode) {
    if (mode == "offline") {
        MAKE_BOUND_OFFLINE_ENTRY("mix_all", bound_offline_mix_all)
        MAKE_BOUND_OFFLINE_ENTRY("mix_gen", bound_offline_mix_gen)
        MAKE_BOUND_OFFLINE_ENTRY("mix_real", bound_offline_mix_real)
        MAKE_BOUND_OFFLINE_ENTRY("randomD", bound_offline_randomD)
        MAKE_BOUND_OFFLINE_ENTRY("randomE", bound_offline_randomE)
        MAKE_BOUND_OFFLINE_ENTRY("randomG", bound_offline_randomG)
        MAKE_BOUND_OFFLINE_ENTRY("randomH", bound_offline_randomH)
        MAKE_BOUND_OFFLINE_ENTRY("randomT", bound_offline_randomT)
        MAKE_BOUND_OFFLINE_ENTRY("RF", bound_offline_RF)
        MAKE_BOUND_OFFLINE_ENTRY("RD", bound_offline_RD)
        throw std::invalid_argument("Unknown offline bound model graph name: " + graph_name);
    }

    if (mode == "online") {
        MAKE_BOUND_ONLINE_ENTRY("mix_all", bound_online_mix_all)
        MAKE_BOUND_ONLINE_ENTRY("mix_gen", bound_online_mix_gen)
        MAKE_BOUND_ONLINE_ENTRY("mix_real", bound_online_mix_real)
        MAKE_BOUND_ONLINE_ENTRY("randomD", bound_online_randomD)
        MAKE_BOUND_ONLINE_ENTRY("randomE", bound_online_randomE)
        MAKE_BOUND_ONLINE_ENTRY("randomG", bound_online_randomG)
        MAKE_BOUND_ONLINE_ENTRY("randomH", bound_online_randomH)
        MAKE_BOUND_ONLINE_ENTRY("randomT", bound_online_randomT)
        MAKE_BOUND_ONLINE_ENTRY("RF", bound_online_RF)
        MAKE_BOUND_ONLINE_ENTRY("RD", bound_online_RD)
        throw std::invalid_argument("Unknown online bound model graph name: " + graph_name);
    }

    if (mode == "blank") {
        return BoundModel{
            true,
            [](const BoundState& st, int n, int k, int t, int l) -> BoundPrediction {
                static blank_bound::Model model;
                blank_bound::State native_st;
                native_st.delta_mu = st.delta_mu;
                const auto pred = model.infer(native_st, n, k, t, l);
                return BoundPrediction{pred.log_max_dist, pred.max_dist, pred.bound};
            },
            [](BoundState& st, double predicted_log_mean, double observed_max_dist) {
                static blank_bound::Model model;
                blank_bound::State native_st;
                native_st.delta_mu = st.delta_mu;
                model.update(native_st, predicted_log_mean, observed_max_dist);
                st.delta_mu = native_st.delta_mu;
            }
        };
    }

    throw std::invalid_argument(
        "Unknown model mode: " + mode + ". Expected 'online', 'offline', or 'blank'."
    );
}

#undef MAKE_BOUND_OFFLINE_ENTRY
#undef MAKE_BOUND_ONLINE_ENTRY
