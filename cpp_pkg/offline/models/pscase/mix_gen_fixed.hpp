#pragma once
#include <array>
#include <cmath>
#include <algorithm>

namespace pscase_fixed_mix_gen {

struct Model {
    static constexpr int kNumFeatures = 7;

    std::array<double, kNumFeatures> mean = {1.0714374391888186, 1.47667016015175, 1.47667016015175, 2.5611278591880424, 1.7228261852782802, 0.61173115565629377, -0.10776726766264975};
    std::array<double, kNumFeatures> scale = {0.30436215404999939, 0.49945542200705784, 0.49945542200705784, 0.81846986975352609, 2.8902253749758819, 2.868743477954482, 0.98243276367363397};
    std::array<double, kNumFeatures> weights = {-0.23983675397523821, -0.046194906596493189, -0.046194906596493189, -2.8717464436567353, -95.487554228739967, -13.899803774088838, -2.5036768646981122};
    double bias = -26.952873277462274;
    double threshold = 0.64594762783646043;

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

} // namespace pscase_fixed_mix_gen
