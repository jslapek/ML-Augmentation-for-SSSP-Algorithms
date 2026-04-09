#pragma once
#include <array>
#include <cmath>
#include <algorithm>

namespace pscase_online_randomE {

struct State {
    double delta_b = 0.0;
};

struct Decision {
    double prob = 0.0;
    bool pred = false;
};

struct Model {
    static constexpr int kNumFeatures = 5;

    std::array<double, kNumFeatures> mean = {1.0919910523095748, 1.096173394454373, 2.68158935770699, 1.0221338347972211, 1.3002825179795214};
    std::array<double, kNumFeatures> scale = {0.35290075996951864, 0.3617958971651703, 0.90317761601014479, 0.73849021192973818, 0.80279259650328505};
    std::array<double, kNumFeatures> weights = {-0.063332485725463683, 1.7862972330199391, 28.011692888645801, 0.84590466685278698, -0.16370248558090933};
    double bias = 14.331877925932147;
    double threshold = 2.1076565622763191e-18;

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

} // namespace pscase_online_randomE
