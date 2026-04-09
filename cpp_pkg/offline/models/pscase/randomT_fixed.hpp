#pragma once
#include <array>
#include <cmath>
#include <algorithm>

namespace pscase_fixed_randomT {

struct Model {
    static constexpr int kNumFeatures = 7;

    std::array<double, kNumFeatures> mean = {1.0436084225307276, 1.486914202609092, 1.486914202609092, 2.8142569435430409, 1.9942839840933455, 0.96537751231869728, -0.25072343954412435};
    std::array<double, kNumFeatures> scale = {0.24320792961395199, 0.49982873257632637, 0.49982873257632637, 0.4975538295688125, 5.4058147762390343, 5.4127765551665687, 0.97206330629270932};
    std::array<double, kNumFeatures> weights = {-0.034349150113332075, -0.10338438129436221, -0.10338438129436221, -3.3635010828846696, -160.94071351696482, -33.592382823400122, -2.2668648040775503};
    double bias = -35.881869541714337;
    double threshold = 0.72518521006290093;

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

} // namespace pscase_fixed_randomT
