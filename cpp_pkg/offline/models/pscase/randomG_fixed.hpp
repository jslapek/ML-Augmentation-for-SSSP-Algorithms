#pragma once
#include <array>
#include <cmath>
#include <algorithm>

namespace pscase_fixed_randomG {

struct Model {
    static constexpr int kNumFeatures = 7;

    std::array<double, kNumFeatures> mean = {1.0641025641025641, 1.4743589743589745, 1.4743589743589745, 2.7096700859841953, 2.0501907123801564, 0.76259644390315529, 0.47734457679737935};
    std::array<double, kNumFeatures> scale = {0.29263364642372064, 0.49934210497732595, 0.49934210497732595, 0.46402947640897962, 0.8639686783252607, 0.66403952826562462, 1.253159247373469};
    std::array<double, kNumFeatures> weights = {-0.52829451700071106, 0.062900486691830473, 0.062900486691830473, -6.5981806180870031, -2.9635162695678172, -3.8086338982326287, -0.86645197988258982};
    double bias = -16.972417756759295;
    double threshold = 0.00034464061727600495;

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

} // namespace pscase_fixed_randomG
