#pragma once
#include <array>
#include <cmath>
#include <algorithm>

namespace pscase_fixed_RD {

struct Model {
    static constexpr int kNumFeatures = 7;

    std::array<double, kNumFeatures> mean = {1.042784572552496, 1.4828279960108592, 1.3146174995844646, 3.0278884026207593, 0.96601638405744072, 0.071324273939226884, -0.58758855185106873};
    std::array<double, kNumFeatures> scale = {0.23410518105241748, 0.4997050352740745, 0.54937490968339753, 0.57090408453938968, 0.48087099572085479, 0.40022994949469592, 0.57914950745114135};
    std::array<double, kNumFeatures> weights = {1.3345540937212232, -0.072191874524770994, 0.092461118849407123, -1.2377364250357854, -0.73616901121191347, -19.941753994446184, 0.58776239745471681};
    double bias = -3.4463506950733023;
    double threshold = 0.5421974954232367;

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

} // namespace pscase_fixed_RD
