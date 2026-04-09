#pragma once
#include <array>
#include <cmath>
#include <algorithm>

namespace pscase_fixed_randomH {

struct Model {
    static constexpr int kNumFeatures = 7;

    std::array<double, kNumFeatures> mean = {1.0816686963757294, 1.4856974392602675, 1.4856974392602675, 2.827884529020475, 1.1957747011506743, 0.15589183564577799, -0.38045871350199617};
    std::array<double, kNumFeatures> scale = {0.31631513637434067, 0.49979539489223174, 0.49979539489223174, 0.67332329859334705, 0.65285836567766575, 0.63286373688897724, 0.83516809895561051};
    std::array<double, kNumFeatures> weights = {-0.85078439956962759, -0.0053664056211312171, -0.0053664056211312171, -1.328156628563038, -47.992672560146154, -2.0312088970447211, -4.0594575960637034};
    double bias = -15.921203495752463;
    double threshold = 0.69177123798625328;

    static double sigmoid(double z) {
        if (z >= 0.0) {
            const double ez = std::exp(-z);
            return 1.0 / (1.0 + ez);
        }
        const double ez = std::exp(z);
        return ez / (1.0 + ez);
    }

    double predict_proba(
        int l,
        int round_idx,
        int S_size,
        int k,
        int W_cumulative,
        int W_i,
        int W_prev
    ) const {
        const double denom = std::max(
            1.0,
            static_cast<double>(k) * static_cast<double>(S_size)
        );

        std::array<double, kNumFeatures> x = {
            static_cast<double>(l),
            static_cast<double>(round_idx),
            static_cast<double>(round_idx) / std::max(1.0, static_cast<double>(k)),
            std::log1p(static_cast<double>(S_size)),
            static_cast<double>(W_cumulative) / denom,
            static_cast<double>(W_i) / denom,
            std::log1p(static_cast<double>(W_i)) -
                std::log1p(std::max(1.0, static_cast<double>(W_prev)))
        };

        double z = bias;
        for (int i = 0; i < kNumFeatures; ++i) {
            const double s = (scale[i] == 0.0 ? 1.0 : scale[i]);
            const double xs = (x[i] - mean[i]) / s;
            z += weights[i] * xs;
        }
        return sigmoid(z);
    }

    bool predict(
        int l,
        int round_idx,
        int S_size,
        int k,
        int W_cumulative,
        int W_i,
        int W_prev
    ) const {
        return predict_proba(
            l,
            round_idx,
            S_size,
            k,
            W_cumulative,
            W_i,
            W_prev
        ) >= threshold;
    }
};

} // namespace pscase_fixed_randomH
