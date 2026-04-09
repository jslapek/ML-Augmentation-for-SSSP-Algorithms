#pragma once
#include <array>
#include <cmath>
#include <algorithm>

namespace countmin_offline_RF {

struct Model {
    static constexpr int kNumFeatures = 6;

    std::array<double, kNumFeatures> mean = {2.2236238532110093, 0.700967125937216, 0.98499344692005153, 0.99793577981651371, 1.548623853211009, 0.99682427664079076};
    std::array<double, kNumFeatures> scale = {0.83322171558411662, 0.11519498868247141, 0.091168724928397935, 0.036098177856717775, 0.62708217258518639, 0.046781326310478065};
    std::array<double, kNumFeatures> weights = {-3.1265809092578394, -0.26770405413189008, -0.13351469633875784, 1.2810583336999162, 3.926191806685003, 0.26770405413041876};
    double bias = 6.1901751319180933;
    double threshold = 0.9999519400640916;
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

} // namespace countmin_offline_RF
