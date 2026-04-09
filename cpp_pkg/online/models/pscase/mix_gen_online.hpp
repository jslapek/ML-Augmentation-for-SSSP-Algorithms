#pragma once
#include <array>
#include <cmath>
#include <algorithm>

namespace pscase_online_mix_gen {

struct State {
    double delta_b = 0.0;
};

struct Decision {
    double prob = 0.0;
    bool pred = false;
};

struct Model {
    static constexpr int kNumFeatures = 5;

    std::array<double, kNumFeatures> mean = {1.0432770640645916, 2.6447375397194937, 1.5220585591905409, 0.26750849294869106, 0.32032632735395461};
    std::array<double, kNumFeatures> scale = {0.21937957647538181, 0.73680773010690637, 0.83873379152835659, 0.54123590745691186, 0.59770996303767598};
    std::array<double, kNumFeatures> weights = {-0.57712073324682811, 1.3247155997584552, 24.08825805954347, 0.057245403839796191, -0.24312689112074046};
    double bias = -15.051385569881056;
    double threshold = 4.0184381676148938e-14;

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

} // namespace pscase_online_mix_gen
