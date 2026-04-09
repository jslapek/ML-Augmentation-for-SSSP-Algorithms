#pragma once
#include <array>
#include <cmath>
#include <algorithm>

namespace pscase_online_randomG {

struct State {
    double delta_b = 0.0;
};

struct Decision {
    double prob = 0.0;
    bool pred = false;
};

struct Model {
    static constexpr int kNumFeatures = 5;

    std::array<double, kNumFeatures> mean = {1.027027027027027, 2.7923752410008027, 2.2789149443562522, 0.6480723370429351, 0.62741088611824558};
    std::array<double, kNumFeatures> scale = {0.16216216216228849, 0.23495172261748953, 0.92070814775868981, 0.51054230571696424, 0.4815322480646354};
    std::array<double, kNumFeatures> weights = {5.0166272308346615e-06, 0.063550535870063304, 10.292823409939095, 6.1145563637766802, 6.4830718422576368};
    double bias = 17.665672192666971;
    double threshold = 1.8125927600078768e-06;

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
        int W_pref,
        int W_r,
        int W_prev
    ) const {
        const double denom = std::max(
            1.0,
            static_cast<double>(k) * static_cast<double>(S_size)
        );

        std::array<double, kNumFeatures> x = {
            static_cast<double>(l),
            std::log1p(static_cast<double>(S_size)),
            static_cast<double>(W_pref) / denom,
            static_cast<double>(W_r) / denom,
            static_cast<double>(W_r) / std::max(1.0, static_cast<double>(W_prev))
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
        int W_pref,
        int W_r,
        int W_prev
    ) const {
        const double z = base_logit(l, S_size, k, W_pref, W_r, W_prev) + st.delta_b;
        const double p = sigmoid(z);
        return Decision{p, p >= threshold};
    }

    void update(State& st, double prob, int label) const {
        const double y = static_cast<double>(label);
        const double grad = (prob - y) + l2 * st.delta_b;
        st.delta_b -= eta * grad;
        st.delta_b = std::clamp(st.delta_b, -bias_clip, bias_clip);
    }
};

} // namespace pscase_online_randomG
