#pragma once
#include <array>
#include <algorithm>
#include <cmath>

namespace bound_offline_mix_all {

struct Model {
    static constexpr int kNumFeatures = 4;

    std::array<double, kNumFeatures> mean = {6.8688870729820346, 1.04575, 2.9962499999999999, 3.9275000000000002};
    std::array<double, kNumFeatures> scale = {0.72475447983936869, 0.20894242628053186, 0.31182677482856924, 0.25931399885081768};
    std::array<double, kNumFeatures> weights = {-3.7781643867492676, -3.1075737476348877, 5.7591919898986816, 2.3072099685668945};
    double bias = 8.6129999160766602;
    double sigma = 4.5695772171020508;
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

} // namespace bound_offline_mix_all
