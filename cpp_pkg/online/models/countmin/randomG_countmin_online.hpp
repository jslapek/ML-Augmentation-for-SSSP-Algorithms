#pragma once
#include <array>
#include <cmath>
#include <algorithm>

namespace countmin_online_randomG {

struct State {
    double delta_b = 0.0;
};

struct Decision {
    double prob = 0.0;
    bool pred = false;
};

struct Model {
    static constexpr int kNumFeatures = 6;

    std::array<double, kNumFeatures> mean = {1.0295652173913044, 2.8317402213344023, 0.049212666638110782, 0.53130434782608693, 2.313043478258725, 0.10424843457748184};
    std::array<double, kNumFeatures> scale = {0.16938451910313659, 0.01691682843738904, 0.012129917889404037, 0.28810917178865858, 0.90824720636727196, 0.029633436547415678};
    std::array<double, kNumFeatures> weights = {-1.9859891429165646, 2.0617455089643824, 3.1398954857761519, -1.3106657319485648, 13.483436608910996, 2.7221149925338706};
    double bias = 10.461564668904;
    double threshold = 0.99999999967689879;

    int prefix_round = 2;
    double eta = 0.02;
    double l2 = 0.0001;
    double bias_clip = 2;

    static double sigmoid(double z) {
        if (z >= 0.0) {
            const double ez = std::exp(-z);
            return 1.0 / (1.0 + ez);
        }
        const double ez = std::exp(z);
        return ez / (1.0 + ez);
    }

    double base_logit(
        int l,
        int S_size,
        int k,
        int prefix_owner_count,
        int rank_in_S,
        int W_prefix,
        double top_owner_mass
    ) const {
        const double denom = std::max(
            1.0,
            static_cast<double>(k) * static_cast<double>(S_size)
        );

        std::array<double, kNumFeatures> x = {
            static_cast<double>(l),
            std::log1p(static_cast<double>(S_size)),
            static_cast<double>(prefix_owner_count) /
                std::max(1.0, static_cast<double>(W_prefix)),
            static_cast<double>(rank_in_S) /
                std::max(1.0, static_cast<double>(S_size)),
            static_cast<double>(W_prefix) / denom,
            top_owner_mass
        };

        double z = bias;
        for (int i = 0; i < kNumFeatures; ++i) {
            const double s = (scale[i] == 0.0 ? 1.0 : scale[i]);
            const double xs = (x[i] - mean[i]) / s;
            z += weights[i] * xs;
        }
        return z;
    }

    Decision infer(
        const State& st,
        int l,
        int S_size,
        int k,
        int prefix_owner_count,
        int rank_in_S,
        int W_prefix,
        double top_owner_mass
    ) const {
        const double z = base_logit(
            l,
            S_size,
            k,
            prefix_owner_count,
            rank_in_S,
            W_prefix,
            top_owner_mass
        ) + st.delta_b;
        const double p = sigmoid(z);
        return Decision{p, p >= threshold};
    }

    double predict_proba(
        int l,
        int S_size,
        int k,
        int prefix_owner_count,
        int rank_in_S,
        int W_prefix,
        double top_owner_mass
    ) const {
        return infer(State{}, l, S_size, k, prefix_owner_count, rank_in_S, W_prefix, top_owner_mass).prob;
    }

    bool predict(
        int l,
        int S_size,
        int k,
        int prefix_owner_count,
        int rank_in_S,
        int W_prefix,
        double top_owner_mass
    ) const {
        return infer(State{}, l, S_size, k, prefix_owner_count, rank_in_S, W_prefix, top_owner_mass).pred;
    }

    void update(State& st, double prob, int label) const {
        const double y = static_cast<double>(label);
        const double grad = (prob - y) + l2 * st.delta_b;
        st.delta_b -= eta * grad;
        st.delta_b = std::clamp(st.delta_b, -bias_clip, bias_clip);
    }
};

} // namespace countmin_online_randomG
