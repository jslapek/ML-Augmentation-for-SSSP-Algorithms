#pragma once
#include <array>
#include <algorithm>
#include <cmath>

namespace bound_online_randomD {

struct State {
    double delta_mu = 0.0;
};

struct Prediction {
    double log_max_dist = 0.0;
    double max_dist = 0.0;
    double bound = 0.0;
};

struct Model {
    static constexpr int kNumFeatures = 4;

    std::array<double, kNumFeatures> mean = {6.9087547793158768, 1, 3, 4};
    std::array<double, kNumFeatures> scale = {6.5636385215839255e-13, 1, 1, 1};
    std::array<double, kNumFeatures> weights = {-0.010122067295014858, 0, 0, 0};
    double bias = 13.033976554870605;
    double sigma = 0.10619103163480759;
    double safety_margin = 1;
    double z_value = 0;
    double eta = 0.050000000000000003;
    double l2 = 0.0001;
    double delta_clip = 2;

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

    double base_log_mean(
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
        return mu;
    }

    Prediction infer(
        const State& st,
        int n,
        int k,
        int t,
        int l
    ) const {
        const double log_mean = base_log_mean(n, k, t, l) + st.delta_mu;
        const double max_dist = std::max(0.0, std::expm1(log_mean + z_value * sigma));
        return Prediction{log_mean, max_dist, max_dist + safety_margin};
    }

    void update(State& st, double predicted_log_mean, double observed_max_dist) const {
        const double sigma_sq = std::max(sigma * sigma, 1e-12);
        const double y_log = std::log1p(std::max(0.0, observed_max_dist));
        const double grad = (predicted_log_mean - y_log) / sigma_sq + l2 * st.delta_mu;
        st.delta_mu -= eta * grad;
        st.delta_mu = std::clamp(st.delta_mu, -delta_clip, delta_clip);
    }

    double predict_log_max_dist(
        int n,
        int k,
        int t,
        int l
    ) const {
        return infer(State{}, n, k, t, l).log_max_dist;
    }

    double predict_max_dist(
        int n,
        int k,
        int t,
        int l
    ) const {
        return infer(State{}, n, k, t, l).max_dist;
    }

    double predict_bound(
        int n,
        int k,
        int t,
        int l
    ) const {
        return infer(State{}, n, k, t, l).bound;
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

} // namespace bound_online_randomD
