#pragma once
#include <array>
#include <cmath>
#include <algorithm>

namespace pscase_fixed_randomE {

struct Model {
    static constexpr int kNumFeatures = 7;

    std::array<double, kNumFeatures> mean = {1.1065033418681209, 1.4348086323280018, 1.4348086323280018, 1.0647589069399246, 2.3396925124093433, 1.0905229618871333, 0.20636690271006483};
    std::array<double, kNumFeatures> scale = {0.38241915341913729, 0.49573186863580138, 0.49573186863580138, 0.3586561934956532, 0.878616200481809, 0.75005206165136162, 0.38967034419166557};
    std::array<double, kNumFeatures> weights = {-0.48497850107731527, -0.16275931609976654, -0.16275931609976654, -1.8219300531817646, -15.730408269662027, -7.9853976813547058, -3.2285679520835489};
    double bias = -33.282658595985545;
    double threshold = 0.030644825239812309;

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

} // namespace pscase_fixed_randomE
