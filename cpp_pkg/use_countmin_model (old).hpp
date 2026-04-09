#pragma once

#include <functional>
#include <stdexcept>
#include <string>

#include "offline/models/countmin/mix_all_countmin_offline.hpp"
#include "offline/models/countmin/mix_gen_countmin_offline.hpp"
#include "offline/models/countmin/mix_real_countmin_offline.hpp"
#include "offline/models/countmin/randomD_countmin_offline.hpp"
#include "offline/models/countmin/randomE_countmin_offline.hpp"
#include "offline/models/countmin/randomG_countmin_offline.hpp"
#include "offline/models/countmin/randomH_countmin_offline.hpp"
#include "offline/models/countmin/randomT_countmin_offline.hpp"
#include "offline/models/countmin/RF_countmin_offline.hpp"
#include "offline/models/countmin/RD_countmin_offline.hpp"

#include "online/models/countmin/mix_all_countmin_online.hpp"
#include "online/models/countmin/mix_gen_countmin_online.hpp"
#include "online/models/countmin/mix_real_countmin_online.hpp"
#include "online/models/countmin/randomD_countmin_online.hpp"
#include "online/models/countmin/randomE_countmin_online.hpp"
#include "online/models/countmin/randomG_countmin_online.hpp"
#include "online/models/countmin/randomH_countmin_online.hpp"
#include "online/models/countmin/randomT_countmin_online.hpp"
#include "online/models/countmin/RF_countmin_online.hpp"
#include "online/models/countmin/RD_countmin_online.hpp"

struct CountMinState {
    double delta_b = 0.0;
};

struct CountMinDecision {
    double prob = 0.0;
    bool pred = false;
};

struct CountMinModel {
    using InferFn = std::function<CountMinDecision(
        const CountMinState&,
        int,
        int,
        int,
        int,
        int,
        int,
        double
    )>;

    using UpdateFn = std::function<void(CountMinState&, double, int)>;

    bool is_online = false;
    int prefix_round = 0;
    InferFn infer_fn;
    UpdateFn update_fn;

    CountMinDecision infer(
        const CountMinState& st,
        int l,
        int S_size,
        int k,
        int prefix_owner_count,
        int rank_in_S,
        int W_prefix,
        double top_owner_mass
    ) const {
        if (!infer_fn) {
            throw std::logic_error("CountMinModel infer_fn is not set.");
        }
        return infer_fn(
            st,
            l,
            S_size,
            k,
            prefix_owner_count,
            rank_in_S,
            W_prefix,
            top_owner_mass
        );
    }

    double predict_proba(
        const CountMinState& st,
        int l,
        int S_size,
        int k,
        int prefix_owner_count,
        int rank_in_S,
        int W_prefix,
        double top_owner_mass
    ) const {
        return infer(
            st,
            l,
            S_size,
            k,
            prefix_owner_count,
            rank_in_S,
            W_prefix,
            top_owner_mass
        ).prob;
    }

    bool predict(
        const CountMinState& st,
        int l,
        int S_size,
        int k,
        int prefix_owner_count,
        int rank_in_S,
        int W_prefix,
        double top_owner_mass
    ) const {
        return infer(
            st,
            l,
            S_size,
            k,
            prefix_owner_count,
            rank_in_S,
            W_prefix,
            top_owner_mass
        ).pred;
    }

    void update(CountMinState& st, double prob, int label) const {
        if (update_fn) {
            update_fn(st, prob, label);
        }
    }
};

#define MAKE_COUNTMIN_OFFLINE_ENTRY(graph_literal, ns_name)                                  \
    if (graph_name == graph_literal) {                                                        \
        return CountMinModel{                                                                 \
            false,                                                                            \
            0,                                                                                \
            [](const CountMinState&,                                                          \
               int l,                                                                         \
               int S_size,                                                                    \
               int k,                                                                         \
               int prefix_owner_count,                                                        \
               int rank_in_S,                                                                 \
               int W_prefix,                                                                  \
               double top_owner_mass) -> CountMinDecision {                                   \
                static ns_name::Model model;                                                  \
                const double p = model.predict_proba(                                         \
                    l,                                                                        \
                    S_size,                                                                   \
                    k,                                                                        \
                    prefix_owner_count,                                                       \
                    rank_in_S,                                                                \
                    W_prefix,                                                                 \
                    top_owner_mass                                                            \
                );                                                                            \
                return CountMinDecision{p, p >= model.threshold};                            \
            },                                                                                \
            [](CountMinState&, double, int) {}                                                \
        };                                                                                    \
    }

#define MAKE_COUNTMIN_ONLINE_ENTRY(graph_literal, ns_name)                                   \
    if (graph_name == graph_literal) {                                                        \
        static ns_name::Model model;                                                          \
        return CountMinModel{                                                                 \
            true,                                                                             \
            model.prefix_round,                                                               \
            [](const CountMinState& st,                                                       \
               int l,                                                                         \
               int S_size,                                                                    \
               int k,                                                                         \
               int prefix_owner_count,                                                        \
               int rank_in_S,                                                                 \
               int W_prefix,                                                                  \
               double top_owner_mass) -> CountMinDecision {                                   \
                static ns_name::Model inner_model;                                            \
                ns_name::State native_st;                                                     \
                native_st.delta_b = st.delta_b;                                               \
                const auto d = inner_model.infer(                                             \
                    native_st,                                                                \
                    l,                                                                        \
                    S_size,                                                                   \
                    k,                                                                        \
                    prefix_owner_count,                                                       \
                    rank_in_S,                                                                \
                    W_prefix,                                                                 \
                    top_owner_mass                                                            \
                );                                                                            \
                return CountMinDecision{d.prob, d.pred};                                      \
            },                                                                                \
            [](CountMinState& st, double prob, int label) {                                   \
                static ns_name::Model inner_model;                                            \
                ns_name::State native_st;                                                     \
                native_st.delta_b = st.delta_b;                                               \
                inner_model.update(native_st, prob, label);                                   \
                st.delta_b = native_st.delta_b;                                               \
            }                                                                                 \
        };                                                                                    \
    }

inline CountMinModel get_countmin_model(const std::string& graph_name, const std::string& mode) {
    if (mode == "offline") {
        MAKE_COUNTMIN_OFFLINE_ENTRY("mix_all", countmin_offline_mix_all)
        MAKE_COUNTMIN_OFFLINE_ENTRY("mix_gen", countmin_offline_mix_gen)
        MAKE_COUNTMIN_OFFLINE_ENTRY("mix_real", countmin_offline_mix_real)
        MAKE_COUNTMIN_OFFLINE_ENTRY("randomD", countmin_offline_randomD)
        MAKE_COUNTMIN_OFFLINE_ENTRY("randomE", countmin_offline_randomE)
        MAKE_COUNTMIN_OFFLINE_ENTRY("randomG", countmin_offline_randomG)
        MAKE_COUNTMIN_OFFLINE_ENTRY("randomH", countmin_offline_randomH)
        MAKE_COUNTMIN_OFFLINE_ENTRY("randomT", countmin_offline_randomT)
        MAKE_COUNTMIN_OFFLINE_ENTRY("RF", countmin_offline_RF)
        MAKE_COUNTMIN_OFFLINE_ENTRY("RD", countmin_offline_RD)
        throw std::invalid_argument("Unknown offline countmin model graph name: " + graph_name);
    }

    if (mode == "online") {
        MAKE_COUNTMIN_ONLINE_ENTRY("mix_all", countmin_online_mix_all)
        MAKE_COUNTMIN_ONLINE_ENTRY("mix_gen", countmin_online_mix_gen)
        MAKE_COUNTMIN_ONLINE_ENTRY("mix_real", countmin_online_mix_real)
        MAKE_COUNTMIN_ONLINE_ENTRY("randomD", countmin_online_randomD)
        MAKE_COUNTMIN_ONLINE_ENTRY("randomE", countmin_online_randomE)
        MAKE_COUNTMIN_ONLINE_ENTRY("randomG", countmin_online_randomG)
        MAKE_COUNTMIN_ONLINE_ENTRY("randomH", countmin_online_randomH)
        MAKE_COUNTMIN_ONLINE_ENTRY("randomT", countmin_online_randomT)
        MAKE_COUNTMIN_ONLINE_ENTRY("RF", countmin_online_RF)
        MAKE_COUNTMIN_ONLINE_ENTRY("RD", countmin_online_RD)
        throw std::invalid_argument("Unknown online countmin model graph name: " + graph_name);
    }

    throw std::invalid_argument(
        "Unknown model mode: " + mode + ". Expected 'online' or 'offline'."
    );
}

#undef MAKE_COUNTMIN_OFFLINE_ENTRY
#undef MAKE_COUNTMIN_ONLINE_ENTRY
