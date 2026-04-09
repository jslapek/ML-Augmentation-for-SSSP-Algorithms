import json
import math
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler


def compute_k(n: int) -> int:
    return max(1, int(math.floor(math.log(max(n, 2)) ** (1.0 / 3.0))))


def make_features_any_round(l, round_idx, S_size, k, W_cumulative, W_i, W_prev):
    denom = max(1.0, float(k) * float(S_size))
    return np.array(
        [
            float(l),
            float(round_idx),
            float(round_idx) / max(1.0, float(k)),
            math.log1p(float(S_size)),
            float(W_cumulative) / denom,
            float(W_i) / denom,
            math.log1p(float(W_i)) - math.log1p(max(1.0, float(W_prev))),
        ],
        dtype=np.float64,
    )


def load_dataset(root_dir: str):
    X = []
    y = []
    groups = []

    files = sorted(Path(root_dir).glob("graph_*.json"))
    print(f"Found {len(files)} files in {Path(root_dir).resolve()}")

    for file_idx, path in enumerate(files):
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)

        stats = obj["stats"]
        calls = {int(row["call_id"]): row for row in stats["bmssp_calls"]}

        graphs = stats.get("graphs", [])
        if not graphs:
            continue

        n = int(graphs[0]["n"])
        k = compute_k(n)

        rounds_by_call = {}
        for row in stats.get("findpivots_rounds", []):
            call_id = int(row["call_id"])
            rounds_by_call.setdefault(call_id, []).append(row)

        for call_id, rows in rounds_by_call.items():
            rows.sort(key=lambda z: int(z["round_idx"]))

            call = calls.get(call_id)
            if call is None:
                continue

            total_rounds = int(call.get("findpivot_rounds", 0))
            if total_rounds <= 0:
                continue

            final_label = int(bool(call["label_P_eq_S"]))

            for j, row in enumerate(rows):
                round_idx = int(row["round_idx"])
                W_i = int(row["W_i_size"])
                W_cumulative = int(row["W_cumulative"])
                W_prev = 0 if j == 0 else int(rows[j - 1]["W_i_size"])

                feat = make_features_any_round(
                    l=int(call["l"]),
                    round_idx=round_idx,
                    S_size=int(call["S_size"]),
                    k=k,
                    W_cumulative=W_cumulative,
                    W_i=W_i,
                    W_prev=W_prev,
                )

                X.append(feat)
                y.append(final_label)
                groups.append(file_idx)

    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.int64)
    groups = np.asarray(groups, dtype=np.int64)

    print(f"Collected {len(X)} examples")

    if len(X) == 0:
        raise RuntimeError("No training examples found.")

    return X, y, groups


def choose_threshold_for_precision(y_true, p_pred, min_precision=0.995):
    precision, recall, thresholds = precision_recall_curve(y_true, p_pred)

    best_t = 1.0
    best_recall = -1.0

    for i, t in enumerate(thresholds):
        if precision[i] >= min_precision and recall[i] > best_recall:
            best_recall = recall[i]
            best_t = float(t)

    return best_t


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


def export_fixed_cpp_header(path, model_dict, graph_name: str):
    def arr(xs):
        return ", ".join(f"{float(x):.17g}" for x in xs)

    namespace_name = f"pscase_fixed_{cpp_identifier(graph_name)}"

    text = f"""#pragma once
#include <array>
#include <cmath>
#include <algorithm>

namespace {namespace_name} {{

struct Model {{
    static constexpr int kNumFeatures = 7;

    std::array<double, kNumFeatures> mean = {{{arr(model_dict["mean"])}}};
    std::array<double, kNumFeatures> scale = {{{arr(model_dict["scale"])}}};
    std::array<double, kNumFeatures> weights = {{{arr(model_dict["weights"])}}};
    double bias = {float(model_dict["bias"]):.17g};
    double threshold = {float(model_dict["threshold"]):.17g};

    static double sigmoid(double z) {{
        if (z >= 0.0) {{
            const double ez = std::exp(-z);
            return 1.0 / (1.0 + ez);
        }}
        const double ez = std::exp(z);
        return ez / (1.0 + ez);
    }}

    double predict_proba(
        int l,
        int round_idx,
        int S_size,
        int k,
        int W_cumulative,
        int W_i,
        int W_prev
    ) const {{
        const double denom = std::max(
            1.0,
            static_cast<double>(k) * static_cast<double>(S_size)
        );

        std::array<double, kNumFeatures> x = {{
            static_cast<double>(l),
            static_cast<double>(round_idx),
            static_cast<double>(round_idx) / std::max(1.0, static_cast<double>(k)),
            std::log1p(static_cast<double>(S_size)),
            static_cast<double>(W_cumulative) / denom,
            static_cast<double>(W_i) / denom,
            std::log1p(static_cast<double>(W_i)) -
                std::log1p(std::max(1.0, static_cast<double>(W_prev)))
        }};

        double z = bias;
        for (int i = 0; i < kNumFeatures; ++i) {{
            const double s = (scale[i] == 0.0 ? 1.0 : scale[i]);
            const double xs = (x[i] - mean[i]) / s;
            z += weights[i] * xs;
        }}
        return sigmoid(z);
    }}

    bool predict(
        int l,
        int round_idx,
        int S_size,
        int k,
        int W_cumulative,
        int W_i,
        int W_prev
    ) const {{
        return predict_proba(
            l,
            round_idx,
            S_size,
            k,
            W_cumulative,
            W_i,
            W_prev
        ) >= threshold;
    }}
}};

}} // namespace {namespace_name}
"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def train_fixed(root_dir: str, graph_name: str):
    X, y, groups = load_dataset(root_dir)

    unique_groups = np.unique(groups)
    if len(unique_groups) < 2:
        raise RuntimeError(
            "Need at least 2 graph files to do a grouped train/validation split."
        )

    unique_labels = np.unique(y)
    if len(unique_labels) < 2:
        raise RuntimeError(
            f"Need both classes present for training, but found labels: {unique_labels.tolist()}"
        )

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=0)
    train_idx, val_idx = next(splitter.split(X, y, groups=groups))

    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)

    clf = LogisticRegression(
        penalty="l2",
        C=1.0,
        solver="lbfgs",
        class_weight="balanced",
        max_iter=2000,
    )
    clf.fit(X_train_s, y_train)

    p_val = clf.predict_proba(X_val_s)[:, 1]
    threshold = choose_threshold_for_precision(y_val, p_val, min_precision=0.2)

    model = {
        "feature_names": [
            "l",
            "round_idx",
            "round_frac",
            "log1p_S_size",
            "prefix_frac",
            "step_frac",
            "log_growth",
        ],
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist(),
        "weights": clf.coef_[0].tolist(),
        "bias": float(clf.intercept_[0]),
        "threshold": float(threshold),
    }

    y_val = np.asarray(y_val)
    p_val = np.asarray(p_val)
    y_pred = (p_val >= threshold).astype(int)

    positive_rate = float(np.mean(y_val))
    precision = float(precision_score(y_val, y_pred, zero_division=0))

    metrics = {
        "threshold": float(threshold),
        "n_examples": int(len(y_val)),
        "n_positives": int(np.sum(y_val)),
        "n_negatives": int(np.sum(y_val == 0)),
        "n_predicted_positives": int(np.sum(y_pred)),
        "n_predicted_negatives": int(np.sum(y_pred == 0)),
        "accuracy": float(accuracy_score(y_val, y_pred)),
        "precision": precision,
        "recall": float(recall_score(y_val, y_pred, zero_division=0)),
        "f1": float(f1_score(y_val, y_pred, zero_division=0)),
        "positive_rate": positive_rate,
        "precision_lift_over_base_rate": (
            float(precision / positive_rate) if positive_rate > 0.0 else 0.0
        ),
        "pr_auc": float(average_precision_score(y_val, p_val)),
    }

    out_path = Path("models/pscase/" + graph_name + "_fixed.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"Wrote {out_path}")
    print(json.dumps(metrics, indent=2))

    header_path = "models/pscase/" + graph_name + "_fixed.hpp"
    export_fixed_cpp_header(header_path, model, graph_name)
    print(f"Wrote {header_path}")


if __name__ == "__main__":
    for graph_name in ["randomD", "randomE", "randomG", "randomH", "randomT", "RF", "RD", "mix_real", "mix_gen", "mix_all"]:
        train_fixed("../../experiments/5k_ml/" + graph_name, graph_name)