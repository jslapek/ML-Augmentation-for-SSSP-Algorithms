#pragma once
#include <array>
#include <cmath>
#include <algorithm>

namespace pscase_fixed_RF {

struct Model {
    static constexpr int kNumFeatures = 7;

    std::array<double, kNumFeatures> mean = {2.0813302134297667, 1.3104304048783946, 1.1573778628660569, 0.71940725272703321, 2.1337089710733608, 1.0342076829313582, -0.23159154153838107};
    std::array<double, kNumFeatures> scale = {0.83799860864962916, 0.4626698267722501, 0.4690256898464093, 0.19768649822787895, 2.6415030569833173, 2.6641556506792163, 0.66389367483336237};
    std::array<double, kNumFeatures> weights = {-0.20279366144991656, 0.59488620751086418, 0.62360105199461602, 1.948479757520734, -0.57976197289526366, -0.29992176810910992, -2.822523265153722};
    double bias = -13.262860655412537;
    double threshold = 1;

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

} // namespace pscase_fixed_RF
