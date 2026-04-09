#pragma once
#include <array>
#include <cmath>
#include <algorithm>

namespace pscase_fixed_mix_all {

struct Model {
    static constexpr int kNumFeatures = 7;

    std::array<double, kNumFeatures> mean = {1.0711031273461176, 1.4770319847052136, 1.4395153814122146, 2.6595402978742757, 1.5457303169912395, 0.48454635253862371, -0.2241693586330199};
    std::array<double, kNumFeatures> scale = {0.30713694166980032, 0.49947219169078566, 0.51539040210414588, 0.80533486373264063, 2.5438836163468492, 2.512738226819462, 0.92149846941930269};
    std::array<double, kNumFeatures> weights = {0.53586684128259221, -2.8680333121296901, 2.9364873053534652, -1.9218515697130962, -29.332820895704291, -48.919552994938556, -0.41792371644137927};
    double bias = -15.307549991913653;
    double threshold = 0.85826359389251028;

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

} // namespace pscase_fixed_mix_all
