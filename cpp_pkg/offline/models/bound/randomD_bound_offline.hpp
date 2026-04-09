#pragma once
#include <array>
#include <algorithm>
#include <cmath>

namespace bound_offline_randomD {

struct Model {
    static constexpr int kNumFeatures = 4;

    std::array<double, kNumFeatures> mean = {6.9087547793158768, 1, 3, 4};
    std::array<double, kNumFeatures> scale = {6.5636385215839255e-13, 1, 1, 1};
    std::array<double, kNumFeatures> weights = {-0.010122067295014858, 0, 0, 0};
    double bias = 13.033976554870605;
    double sigma = 0.1061910092830658;
    double safety_margin = 1;
    double z_value = 0;

    static int compute_k(int n) {
        return std::max(1, static_cast<int>(std::floor(std::pow(std::log(std::max(n, 2)), 1.0 / 3.0))));
    }

    static int compute_t(int n) {
        return std::max(1, static_cast<int>(std::floor(std::pow(std::log(std::max(n, 2)), 2.0 / 3.0))));
    }

    static int compute_l(int n) {
        const int t = compute_t(n);
        return std::max(1, static_cast<int>(std::ceil(std::log2(std::max(n, 2)) / std::max(1, t))));
    }

    std::array<double, kNumFeatures> make_features(
        int n,
        int k,
        int t,
        int l
    ) const {
        return {
            std::log1p(static_cast<double>(n)),
            static_cast<double>(k),
            static_cast<double>(t),
            static_cast<double>(l)
        };
    }

    double predict_log_max_dist(
        int n,
        int k,
        int t,
        int l
    ) const {
        const auto x = make_features(n, k, t, l);
        double mu = bias;
        for (int i = 0; i < kNumFeatures; ++i) {
            const double s = (scale[i] == 0.0 ? 1.0 : scale[i]);
            const double xs = (x[i] - mean[i]) / s;
            mu += weights[i] * xs;
        }
        return mu + z_value * sigma;
    }

    double predict_max_dist(
        int n,
        int k,
        int t,
        int l
    ) const {
        const double log_y = predict_log_max_dist(n, k, t, l);
        return std::max(0.0, std::expm1(log_y));
    }

    double predict_bound(
        int n,
        int k,
        int t,
        int l
    ) const {
        return predict_max_dist(n, k, t, l) + safety_margin;
    }

    double predict(
        int n,
        int k,
        int t,
        int l
    ) const {
        return predict_bound(n, k, t, l);
    }

    double predict_log_max_dist(int n) const {
        return predict_log_max_dist(n, compute_k(n), compute_t(n), compute_l(n));
    }

    double predict_max_dist(int n) const {
        return predict_max_dist(n, compute_k(n), compute_t(n), compute_l(n));
    }

    double predict_bound(int n) const {
        return predict_bound(n, compute_k(n), compute_t(n), compute_l(n));
    }

    double predict(int n) const {
        return predict_bound(n);
    }
};

} // namespace bound_offline_randomD
