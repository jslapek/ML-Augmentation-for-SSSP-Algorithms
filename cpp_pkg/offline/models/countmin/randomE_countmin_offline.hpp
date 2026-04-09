#pragma once
#include <array>
#include <cmath>
#include <algorithm>

namespace countmin_offline_randomE {

struct Model {
    static constexpr int kNumFeatures = 6;

    std::array<double, kNumFeatures> mean = {1.0593300000000001, 1.2969771512576742, 0.31085320770070241, 0.72674641865079292, 2.3478841507938939, 0.67764436461361544};
    std::array<double, kNumFeatures> scale = {0.27269021086223094, 0.36740133257539453, 0.17844510167444744, 0.27901732644791738, 0.7773846129337747, 0.19506145720528636};
    std::array<double, kNumFeatures> weights = {-0.048349864854805746, 5.6689023829362188, 9.8400440456977556, -0.41711895620739792, 10.436341148793673, 0.43014464464776681};
    double bias = 9.0084726297355662;
    double threshold = 0.99999999995517685;
    int prefix_round = 2;

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
        int S_size,
        int k,
        int prefix_owner_count,
        int rank_in_S,
        int W_prefix,
        double top_owner_mass
    ) const {
        const double denom = std::max(
            1.0,
            static_cast<double>(k) * static_cast<double>(S_size)
        );

        std::array<double, kNumFeatures> x = {
            static_cast<double>(l),
            std::log1p(static_cast<double>(S_size)),
            static_cast<double>(prefix_owner_count) /
                std::max(1.0, static_cast<double>(W_prefix)),
            static_cast<double>(rank_in_S) /
                std::max(1.0, static_cast<double>(S_size)),
            static_cast<double>(W_prefix) / denom,
            top_owner_mass
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
        int S_size,
        int k,
        int prefix_owner_count,
        int rank_in_S,
        int W_prefix,
        double top_owner_mass
    ) const {
        return predict_proba(
            l,
            S_size,
            k,
            prefix_owner_count,
            rank_in_S,
            W_prefix,
            top_owner_mass
        ) >= threshold;
    }
};

} // namespace countmin_offline_randomE
