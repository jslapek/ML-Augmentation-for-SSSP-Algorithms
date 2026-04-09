#pragma once
#include <array>
#include <cmath>
#include <algorithm>

namespace pscase_online_RF {

struct State {
    double delta_b = 0.0;
};

struct Decision {
    double prob = 0.0;
    bool pred = false;
};

struct Model {
    static constexpr int kNumFeatures = 5;

    std::array<double, kNumFeatures> mean = {2.2285318559556786, 0.69934990126196683, 1.5425467451523547, 0.032556171129578329, 0.043640350877192986};
    std::array<double, kNumFeatures> scale = {0.83214413707418378, 0.11984914964739099, 0.62644537567054503, 0.2516838976643479, 0.34134608465243782};
    std::array<double, kNumFeatures> weights = {-0.099356441474791737, -0.037181633054475381, -0.0041131770633636988, 2.2513638239817544, 3.2658195486879249};
    double bias = -6.8675016365391484;
    double threshold = 0.00059635367901647346;

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

} // namespace pscase_online_RF
