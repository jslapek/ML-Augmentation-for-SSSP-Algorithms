#pragma once
#include <array>
#include <cmath>
#include <algorithm>

namespace countmin_offline_mix_real {

struct Model {
    static constexpr int kNumFeatures = 6;

    std::array<double, kNumFeatures> mean = {1.1768099999999999, 3.6047847743108332, 0.04037046547217113, 0.52109405181638058, 0.93223447257342607, 0.045938967599488387};
    std::array<double, kNumFeatures> scale = {0.38222797372782519, 1.0860660071473875, 0.03247106236629866, 0.28864165668021485, 0.4472675031863762, 0.039693634304356976};
    std::array<double, kNumFeatures> weights = {-0.28632627937819205, 2.7628472301040023, 1.267634097096157, 0.032443185727770836, 2.5339623553458326, 0.34526329050690224};
    double bias = -4.9707998466444732;
    double threshold = 0.005709662011214151;
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

} // namespace countmin_offline_mix_real
