#pragma once
#include <array>
#include <cmath>
#include <algorithm>

namespace countmin_offline_mix_all {

struct Model {
    static constexpr int kNumFeatures = 6;

    std::array<double, kNumFeatures> mean = {1.23576, 3.4952808826630553, 0.046897470540409512, 0.52580010948604283, 1.2461949975072704, 0.075418731182202561};
    std::array<double, kNumFeatures> scale = {0.42501437905107459, 1.1134913658159016, 0.045842278747041575, 0.28975922374349916, 0.59831296260950839, 0.082259788161646613};
    std::array<double, kNumFeatures> weights = {0.53920890415352174, 2.5631626779463135, 5.3366987562484374, -0.45979152877290963, 3.325534868112423, -1.2648627951331004};
    double bias = -2.3117042834791666;
    double threshold = 0.49285633995220374;
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

} // namespace countmin_offline_mix_all
