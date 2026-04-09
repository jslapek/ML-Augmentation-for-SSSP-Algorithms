import json
import math
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import GroupShuffleSplit


OUTPUT_DIR = Path("models/bound")
GRAPH_NAMES = [
    "randomD",
    "randomE",
    "randomG",
    "randomH",
    "randomT",
    "RF",
    "RD",
    "mix_real",
    "mix_gen",
    "mix_all",
]
SAFETY_MARGIN = 1.0


def compute_k(n: int) -> int:
    return max(1, int(math.floor(math.log(max(n, 2)) ** (1.0 / 3.0))))


def compute_t(n: int) -> int:
    return max(1, int(math.floor(math.log(max(n, 2)) ** (2.0 / 3.0))))


def compute_l(n: int) -> int:
    t = compute_t(n)
    return max(1, int(math.ceil(math.log2(max(n, 2)) / max(1, t))))


def cpp_identifier(s: str) -> str:
    out = []
    for ch in s:
        if ch.isalnum() or ch == "_":
            out.append(ch)
        else:
            out.append("_")

    name = "".join(out)
    if not name:
        name = "default"
    if name[0].isdigit():
        name = "_" + name
    return name


def make_graph_features(n: int, k: int, t: int, l: int) -> np.ndarray:
    return np.array(
        [
            math.log1p(float(n)),
            float(k),
            float(t),
            float(l),
        ],
        dtype=np.float64,
    )


def load_dataset(root_dir: str, graph_name: str):
    X = []
    y = []
    groups = []

    files = sorted(Path(root_dir).glob("graph_*.json"))
    print(f"Found {len(files)} files in {Path(root_dir).resolve()}")

    for file_idx, path in enumerate(files):
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)

        stats = obj.get("stats", {})
        graphs = stats.get("graphs", [])
        if not graphs:
            continue

        graph_stats = graphs[0]
        n = int(graph_stats["n"])
        max_dist = float(graph_stats["max_dist"])
        k = compute_k(n)
        t = compute_t(n)
        l = compute_l(n)

        X.append(make_graph_features(n=n, k=k, t=t, l=l))
        y.append(max_dist)
        groups.append(file_idx)

    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    groups = np.asarray(groups, dtype=np.int64)

    print(f"Collected {len(X)} examples")

    if len(X) == 0:
        raise RuntimeError("No training examples found.")

    return X, y, groups


class GaussianLinear(nn.Module):
    def __init__(self, n_features: int, init_mu: float, init_sigma: float = 0.5):
        super().__init__()
        self.linear = nn.Linear(n_features, 1)
        nn.init.zeros_(self.linear.weight)
        nn.init.constant_(self.linear.bias, float(init_mu))
        inv_softplus = math.log(math.exp(init_sigma) - 1.0) if init_sigma > 1e-6 else -6.0
        self.raw_sigma = nn.Parameter(torch.tensor(inv_softplus, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mu = self.linear(x).squeeze(-1)
        sigma = F.softplus(self.raw_sigma) + 1e-4
        return mu, sigma.expand_as(mu)


LOG_SQRT_2PI = 0.5 * math.log(2.0 * math.pi)


def gaussian_nll(y: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    z = (y - mu) / sigma
    return 0.5 * z * z + torch.log(sigma) + LOG_SQRT_2PI


def fit_gaussian_log_model(
    X_train_s: np.ndarray,
    y_train_log: np.ndarray,
    X_val_s: np.ndarray,
    y_val_log: np.ndarray,
    epochs: int = 2500,
    lr: float = 0.03,
    weight_decay: float = 1e-4,
    patience: int = 300,
):
    x_train = torch.tensor(X_train_s, dtype=torch.float32)
    y_train = torch.tensor(y_train_log, dtype=torch.float32)
    x_val = torch.tensor(X_val_s, dtype=torch.float32)
    y_val = torch.tensor(y_val_log, dtype=torch.float32)

    init_mu = float(np.mean(y_train_log))
    model = GaussianLinear(X_train_s.shape[1], init_mu=init_mu)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_state = None
    best_val = float("inf")
    best_epoch = -1

    for epoch in range(epochs):
        model.train()
        opt.zero_grad()
        mu_train, sigma_train = model(x_train)
        loss = gaussian_nll(y_train, mu_train, sigma_train).mean()
        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            mu_val, sigma_val = model(x_val)
            val_loss = float(gaussian_nll(y_val, mu_val, sigma_val).mean().item())

        if val_loss < best_val - 1e-7:
            best_val = val_loss
            best_epoch = epoch
            best_state = {
                "linear_weight": model.linear.weight.detach().cpu().clone(),
                "linear_bias": model.linear.bias.detach().cpu().clone(),
                "raw_sigma": model.raw_sigma.detach().cpu().clone(),
            }
        elif epoch - best_epoch >= patience:
            break

    if best_state is None:
        raise RuntimeError("Training failed to produce a valid model state.")

    model.linear.weight.data.copy_(best_state["linear_weight"])
    model.linear.bias.data.copy_(best_state["linear_bias"])
    model.raw_sigma.data.copy_(best_state["raw_sigma"])
    model.eval()
    return model, best_val


def save_val_metrics_json(
    target_max_dist: np.ndarray,
    pred_max_dist: np.ndarray,
    pred_bound: np.ndarray,
    pred_log_mean: np.ndarray,
    out_path,
):
    target_max_dist = np.asarray(target_max_dist, dtype=np.float64)
    pred_max_dist = np.asarray(pred_max_dist, dtype=np.float64)
    pred_bound = np.asarray(pred_bound, dtype=np.float64)
    pred_log_mean = np.asarray(pred_log_mean, dtype=np.float64)

    abs_err = np.abs(pred_max_dist - target_max_dist)
    sq_err = (pred_max_dist - target_max_dist) ** 2
    rel_err = abs_err / np.maximum(target_max_dist, 1e-12)
    bound_gap = pred_bound - target_max_dist

    metrics = {
        "n_examples": int(len(target_max_dist)),
        "target_max_dist_mean": float(np.mean(target_max_dist)),
        "prediction_max_dist_mean": float(np.mean(pred_max_dist)),
        "prediction_bound_mean": float(np.mean(pred_bound)),
        "mae_vs_max_dist": float(np.mean(abs_err)),
        "rmse_vs_max_dist": float(np.sqrt(np.mean(sq_err))),
        "mean_relative_error_vs_max_dist": float(np.mean(rel_err)),
        "median_absolute_error_vs_max_dist": float(np.median(abs_err)),
        "mean_signed_error_vs_max_dist": float(np.mean(pred_max_dist - target_max_dist)),
        "mean_bound_gap_vs_max_dist": float(np.mean(bound_gap)),
        "safe_rate_vs_max_dist": float(np.mean(pred_bound > target_max_dist)),
        "underprediction_rate_vs_max_dist": float(np.mean(pred_max_dist < target_max_dist)),
        "within_1pct_rate": float(np.mean(rel_err <= 0.01)),
        "within_5pct_rate": float(np.mean(rel_err <= 0.05)),
        "pred_log_mean_mean": float(np.mean(pred_log_mean)),
    }

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"Wrote {out_path}")
    print(json.dumps(metrics, indent=2))


def export_bound_cpp_header(path, model_dict, graph_name: str):
    def arr(xs: Iterable[float]) -> str:
        return ", ".join(f"{float(x):.17g}" for x in xs)

    namespace_name = f"bound_offline_{cpp_identifier(graph_name)}"

    text = f"""#pragma once
#include <array>
#include <algorithm>
#include <cmath>

namespace {namespace_name} {{

struct Model {{
    static constexpr int kNumFeatures = {len(model_dict['feature_names'])};

    std::array<double, kNumFeatures> mean = {{{arr(model_dict['mean'])}}};
    std::array<double, kNumFeatures> scale = {{{arr(model_dict['scale'])}}};
    std::array<double, kNumFeatures> weights = {{{arr(model_dict['weights'])}}};
    double bias = {float(model_dict['bias']):.17g};
    double sigma = {float(model_dict['sigma']):.17g};
    double safety_margin = {float(model_dict['safety_margin']):.17g};
    double z_value = {float(model_dict['z_value']):.17g};

    static int compute_k(int n) {{
        return std::max(1, static_cast<int>(std::floor(std::pow(std::log(std::max(n, 2)), 1.0 / 3.0))));
    }}

    static int compute_t(int n) {{
        return std::max(1, static_cast<int>(std::floor(std::pow(std::log(std::max(n, 2)), 2.0 / 3.0))));
    }}

    static int compute_l(int n) {{
        const int t = compute_t(n);
        return std::max(1, static_cast<int>(std::ceil(std::log2(std::max(n, 2)) / std::max(1, t))));
    }}

    std::array<double, kNumFeatures> make_features(
        int n,
        int k,
        int t,
        int l
    ) const {{
        return {{
            std::log1p(static_cast<double>(n)),
            static_cast<double>(k),
            static_cast<double>(t),
            static_cast<double>(l)
        }};
    }}

    double predict_log_max_dist(
        int n,
        int k,
        int t,
        int l
    ) const {{
        const auto x = make_features(n, k, t, l);
        double mu = bias;
        for (int i = 0; i < kNumFeatures; ++i) {{
            const double s = (scale[i] == 0.0 ? 1.0 : scale[i]);
            const double xs = (x[i] - mean[i]) / s;
            mu += weights[i] * xs;
        }}
        return mu + z_value * sigma;
    }}

    double predict_max_dist(
        int n,
        int k,
        int t,
        int l
    ) const {{
        const double log_y = predict_log_max_dist(n, k, t, l);
        return std::max(0.0, std::expm1(log_y));
    }}

    double predict_bound(
        int n,
        int k,
        int t,
        int l
    ) const {{
        return predict_max_dist(n, k, t, l) + safety_margin;
    }}

    double predict(
        int n,
        int k,
        int t,
        int l
    ) const {{
        return predict_bound(n, k, t, l);
    }}

    double predict_log_max_dist(int n) const {{
        return predict_log_max_dist(n, compute_k(n), compute_t(n), compute_l(n));
    }}

    double predict_max_dist(int n) const {{
        return predict_max_dist(n, compute_k(n), compute_t(n), compute_l(n));
    }}

    double predict_bound(int n) const {{
        return predict_bound(n, compute_k(n), compute_t(n), compute_l(n));
    }}

    double predict(int n) const {{
        return predict_bound(n);
    }}
}};

}} // namespace {namespace_name}
"""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def train_bound(root_dir: str, graph_name: str, safety_margin: float = SAFETY_MARGIN):
    X, y, groups = load_dataset(root_dir=root_dir, graph_name=graph_name)

    unique_groups = np.unique(groups)
    if len(unique_groups) < 2:
        raise RuntimeError(
            "Need at least 2 graph files to do a grouped train/validation split."
        )

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=0)
    train_idx, val_idx = next(splitter.split(X, y, groups=groups))

    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    y_train_log = np.log1p(np.maximum(y_train, 0.0))
    y_val_log = np.log1p(np.maximum(y_val, 0.0))

    mean = X_train.mean(axis=0)
    scale = X_train.std(axis=0)
    scale = np.where(scale == 0.0, 1.0, scale)

    X_train_s = (X_train - mean) / scale
    X_val_s = (X_val - mean) / scale

    model, best_val_nll = fit_gaussian_log_model(
        X_train_s=X_train_s,
        y_train_log=y_train_log,
        X_val_s=X_val_s,
        y_val_log=y_val_log,
    )

    with torch.no_grad():
        x_val_t = torch.tensor(X_val_s, dtype=torch.float32)
        mu_val, sigma_val = model(x_val_t)
        pred_log = mu_val.cpu().numpy()
        pred_sigma = sigma_val.cpu().numpy()

    z_value = 0.0
    pred_max_dist = np.maximum(0.0, np.expm1(pred_log + z_value * pred_sigma))
    pred_bound = pred_max_dist + float(safety_margin)

    model_dict = {
        "feature_names": [
            "log1p_n",
            "k",
            "t",
            "l",
        ],
        "mean": mean.tolist(),
        "scale": scale.tolist(),
        "weights": model.linear.weight.detach().cpu().numpy()[0].tolist(),
        "bias": float(model.linear.bias.detach().cpu().numpy()[0]),
        "sigma": float((F.softplus(model.raw_sigma) + 1e-4).detach().cpu().item()),
        "safety_margin": float(safety_margin),
        "z_value": float(z_value),
        "best_val_nll": float(best_val_nll),
    }

    json_path = OUTPUT_DIR / f"{graph_name}_bound_offline.json"
    header_path = OUTPUT_DIR / f"{graph_name}_bound_offline.hpp"

    save_val_metrics_json(
        target_max_dist=y_val,
        pred_max_dist=pred_max_dist,
        pred_bound=pred_bound,
        pred_log_mean=pred_log,
        out_path=json_path,
    )
    export_bound_cpp_header(header_path, model_dict, graph_name)
    print(f"Wrote {header_path}")


# Compatibility alias for the same call shape as the offline p=s trainer.
train_fixed = train_bound


if __name__ == "__main__":
    for graph_name in GRAPH_NAMES:
        train_bound(f"../../experiments/5k_ml/{graph_name}", graph_name)
