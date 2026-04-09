#pragma once
#include <array>
#include <cmath>
#include <algorithm>

namespace pscase_fixed_mix_real {

struct Model {
    static constexpr int kNumFeatures = 7;

    std::array<double, kNumFeatures> mean = {1.0670807453416149, 1.4788635716796974, 1.3061358423599136, 2.9808038183978129, 0.98936835116239763, 0.093211398281236715, -0.5800562762522955};
    std::array<double, kNumFeatures> scale = {0.30753488344312713, 0.49955305163541919, 0.54867479832421129, 0.66791319623609746, 0.66595219233889202, 0.60474174653757062, 0.58215159735127897};
    std::array<double, kNumFeatures> weights = {0.035341336042790931, -0.57986506716346553, 0.70596246256880513, -0.67420974023335223, -1.3129733772542982, -10.181982086760556, 0.22284390756828448};
    double bias = -1.7045083571773063;
    double threshold = 0.55729774313562863;

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

} // namespace pscase_fixed_mix_real
