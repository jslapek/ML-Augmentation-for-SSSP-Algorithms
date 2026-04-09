#pragma once
#include <array>
#include <cmath>
#include <algorithm>

namespace countmin_offline_RD {

struct Model {
    static constexpr int kNumFeatures = 6;

    std::array<double, kNumFeatures> mean = {1.18242, 3.6440255299396296, 0.039170350718377218, 0.5190895906402706, 0.91864773206280326, 0.044492672400839338};
    std::array<double, kNumFeatures> scale = {0.38619029454321135, 1.1269648592054633, 0.022479847527035898, 0.28883078945965485, 0.44516557481763924, 0.031634761305446048};
    std::array<double, kNumFeatures> weights = {1.1177853323463494, 3.494528747159217, 2.9948227354505028, -0.057643791743623253, 2.7364359147270663, 0.27958269081124115};
    double bias = -6.0389023673315885;
    double threshold = 0.0053637660068431483;
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

} // namespace countmin_offline_RD
