#pragma once

#include <functional>
#include <stdexcept>
#include <string>
#include <utility>

#include "offline/models/pscase/mix_all_fixed.hpp"
#include "offline/models/pscase/mix_gen_fixed.hpp"
#include "offline/models/pscase/mix_real_fixed.hpp"
#include "offline/models/pscase/randomD_fixed.hpp"
#include "offline/models/pscase/randomE_fixed.hpp"
#include "offline/models/pscase/randomG_fixed.hpp"
#include "offline/models/pscase/randomH_fixed.hpp"
#include "offline/models/pscase/randomT_fixed.hpp"
#include "offline/models/pscase/RF_fixed.hpp"
#include "offline/models/pscase/RD_fixed.hpp"

#include "online/models/pscase/mix_all_online.hpp"
#include "online/models/pscase/mix_gen_online.hpp"
#include "online/models/pscase/mix_real_online.hpp"
#include "online/models/pscase/randomD_online.hpp"
#include "online/models/pscase/randomE_online.hpp"
#include "online/models/pscase/randomG_online.hpp"
#include "online/models/pscase/randomH_online.hpp"
#include "online/models/pscase/randomT_online.hpp"
#include "online/models/pscase/RF_online.hpp"
#include "online/models/pscase/RD_online.hpp"
#include "online/models/pscase/blank.hpp"

struct PSEqSState {
    double delta_b = 0.0;
};

struct PSEqSDecision {
    double prob = 0.0;
    bool pred = false;
};

struct PSEqSModel {
    using InferFn = std::function<PSEqSDecision(
        const PSEqSState&,
        int,
        int,
        int,
        int,
        int,
        int,
        int
    )>;

    using UpdateFn = std::function<void(PSEqSState&, double, int)>;

    bool is_online = false;
    int prefix_round = 0;
    InferFn infer_fn;
    UpdateFn update_fn;

    PSEqSDecision infer(
        const PSEqSState& st,
        int l,
        int round_idx,
        int S_size,
        int k,
        int W_cumulative,
        int W_i,
        int W_prev
    ) const {
        if (!infer_fn) {
            throw std::logic_error("PSEqSModel infer_fn is not set.");
        }
        return infer_fn(st, l, round_idx, S_size, k, W_cumulative, W_i, W_prev);
    }

    double predict_proba(
        const PSEqSState& st,
        int l,
        int round_idx,
        int S_size,
        int k,
        int W_cumulative,
        int W_i,
        int W_prev
    ) const {
        return infer(st, l, round_idx, S_size, k, W_cumulative, W_i, W_prev).prob;
    }

    bool predict(
        const PSEqSState& st,
        int l,
        int round_idx,
        int S_size,
        int k,
        int W_cumulative,
        int W_i,
        int W_prev
    ) const {
        return infer(st, l, round_idx, S_size, k, W_cumulative, W_i, W_prev).pred;
    }

    void update(PSEqSState& st, double prob, int label) const {
        if (update_fn) {
            update_fn(st, prob, label);
        }
    }
};

#define MAKE_PSCASE_OFFLINE_ENTRY(graph_literal, ns_name)                                     \
    if (graph_name == graph_literal) {                                                        \
        return PSEqSModel{                                                                    \
            false,                                                                            \
            0,                                                                                \
            [](const PSEqSState&,                                                             \
               int l,                                                                         \
               int round_idx,                                                                 \
               int S_size,                                                                    \
               int k,                                                                         \
               int W_cumulative,                                                              \
               int W_i,                                                                       \
               int W_prev) -> PSEqSDecision {                                                 \
                static ns_name::Model model;                                                  \
                const double p = model.predict_proba(                                         \
                    l, round_idx, S_size, k, W_cumulative, W_i, W_prev                        \
                );                                                                            \
                return PSEqSDecision{p, p >= model.threshold};                                \
            },                                                                                \
            [](PSEqSState&, double, int) {}                                                   \
        };                                                                                    \
    }

#define MAKE_PSCASE_ONLINE_ENTRY(graph_literal, ns_name)                                      \
    if (graph_name == graph_literal) {                                                        \
        static ns_name::Model model;                                                          \
        return PSEqSModel{                                                                    \
            true,                                                                             \
            model.prefix_round,                                                               \
            [](const PSEqSState& st,                                                          \
               int l,                                                                         \
               int /*round_idx*/,                                                             \
               int S_size,                                                                    \
               int k,                                                                         \
               int W_cumulative,                                                              \
               int W_i,                                                                       \
               int W_prev) -> PSEqSDecision {                                                 \
                static ns_name::Model inner_model;                                            \
                ns_name::State native_st;                                                     \
                native_st.delta_b = st.delta_b;                                               \
                const auto d = inner_model.infer(                                             \
                    native_st, l, S_size, k, W_cumulative, W_i, W_prev                        \
                );                                                                            \
                return PSEqSDecision{d.prob, d.pred};                                         \
            },                                                                                \
            [](PSEqSState& st, double prob, int label) {                                      \
                static ns_name::Model inner_model;                                            \
                ns_name::State native_st;                                                     \
                native_st.delta_b = st.delta_b;                                               \
                inner_model.update(native_st, prob, label);                                   \
                st.delta_b = native_st.delta_b;                                               \
            }                                                                                 \
        };                                                                                    \
    }

inline PSEqSModel get_P_eq_S_model(const std::string& graph_name, const std::string& mode) {
    if (mode == "offline") {
        MAKE_PSCASE_OFFLINE_ENTRY("mix_all", pscase_fixed_mix_all)
        MAKE_PSCASE_OFFLINE_ENTRY("mix_gen", pscase_fixed_mix_gen)
        MAKE_PSCASE_OFFLINE_ENTRY("mix_real", pscase_fixed_mix_real)
        MAKE_PSCASE_OFFLINE_ENTRY("randomD", pscase_fixed_randomD)
        MAKE_PSCASE_OFFLINE_ENTRY("randomE", pscase_fixed_randomE)
        MAKE_PSCASE_OFFLINE_ENTRY("randomG", pscase_fixed_randomG)
        MAKE_PSCASE_OFFLINE_ENTRY("randomH", pscase_fixed_randomH)
        MAKE_PSCASE_OFFLINE_ENTRY("randomT", pscase_fixed_randomT)
        MAKE_PSCASE_OFFLINE_ENTRY("RF", pscase_fixed_RF)
        MAKE_PSCASE_OFFLINE_ENTRY("RD", pscase_fixed_RD)
        throw std::invalid_argument("Unknown offline P=S model graph name: " + graph_name);
    }

    if (mode == "online") {
        MAKE_PSCASE_ONLINE_ENTRY("mix_all", pscase_online_mix_all)
        MAKE_PSCASE_ONLINE_ENTRY("mix_gen", pscase_online_mix_gen)
        MAKE_PSCASE_ONLINE_ENTRY("mix_real", pscase_online_mix_real)
        MAKE_PSCASE_ONLINE_ENTRY("randomD", pscase_online_randomD)
        MAKE_PSCASE_ONLINE_ENTRY("randomE", pscase_online_randomE)
        MAKE_PSCASE_ONLINE_ENTRY("randomG", pscase_online_randomG)
        MAKE_PSCASE_ONLINE_ENTRY("randomH", pscase_online_randomH)
        MAKE_PSCASE_ONLINE_ENTRY("randomT", pscase_online_randomT)
        MAKE_PSCASE_ONLINE_ENTRY("RF", pscase_online_RF)
        MAKE_PSCASE_ONLINE_ENTRY("RD", pscase_online_RD)
        throw std::invalid_argument("Unknown online P=S model graph name: " + graph_name);
    }

    if (mode == "blank") {
        static blank_pscase::Model model;
        return PSEqSModel{
            true,
            model.prefix_round,
            [](const PSEqSState& st,
               int l,
               int /*round_idx*/,
               int S_size,
               int k,
               int W_cumulative,
               int W_i,
               int W_prev) -> PSEqSDecision {
                static blank_pscase::Model inner_model;
                blank_pscase::State native_st;
                native_st.delta_b = st.delta_b;
                const auto d = inner_model.infer(
                    native_st, l, S_size, k, W_cumulative, W_i, W_prev
                );
                return PSEqSDecision{d.prob, d.pred};
            },
            [](PSEqSState& st, double prob, int label) {
                static blank_pscase::Model inner_model;
                blank_pscase::State native_st;
                native_st.delta_b = st.delta_b;
                inner_model.update(native_st, prob, label);
                st.delta_b = native_st.delta_b;
            }
        };
    }

    throw std::invalid_argument(
        "Unknown model mode: " + mode + ". Expected 'online', 'offline', or 'blank'."
    );
}

#undef MAKE_PSCASE_OFFLINE_ENTRY
#undef MAKE_PSCASE_ONLINE_ENTRY
