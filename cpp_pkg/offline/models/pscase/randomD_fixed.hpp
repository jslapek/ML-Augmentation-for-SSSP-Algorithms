#pragma once
#include <array>
#include <cmath>
#include <algorithm>

namespace pscase_fixed_randomD {

struct Model {
    static constexpr int kNumFeatures = 7;

    std::array<double, kNumFeatures> mean = {1.0792202509639601, 1.4837380480085587, 1.4837380480085587, 2.8356215812429029, 1.2024593502589367, 0.16533017857438145, -0.37824516433492056};
    std::array<double, kNumFeatures> scale = {0.3099039241544092, 0.4997354789458508, 0.4997354789458508, 0.66608350245124182, 0.66221233020835713, 0.65849788899304507, 0.84734535046838333};
    std::array<double, kNumFeatures> weights = {-1.0362627286788075, -0.0054455462102895209, -0.0054455462102895209, -1.4073679729720356, -48.147186276934711, -2.0868850333428144, -4.0125953406723891};
    double bias = -16.329232432815839;
    double threshold = 0.68845247079394256;

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

} // namespace pscase_fixed_randomD
