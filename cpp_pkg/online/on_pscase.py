import json
import math
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler


OUTPUT_DIR = Path("models/pscase/")
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


def compute_k(n: int) -> int:
    return max(1, int(math.floor(math.log(max(n, 2)) ** (1.0 / 3.0))))


def make_features(l, S_size, k, W_pref, W_r, W_prev):
    denom = max(1.0, float(k) * float(S_size))
    return np.array(
        [
            float(l),
            math.log1p(float(S_size)),
            float(W_pref) / denom,
            float(W_r) / denom,
            float(W_r) / max(1.0, float(W_prev)),
        ],
        dtype=np.float64,
    )


def load_dataset(root_dir: str, prefix_round: int = 2):
    X = []
    y = []
    groups = []

    files = sorted(Path(root_dir).glob("graph_*.json"))
    print(f"Found {len(files)} files in {Path(root_dir).resolve()}")

    for file_idx, path in enumerate(files):
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)

        stats = obj["stats"]
        calls = {int(row["call_id"]): row for row in stats.get("bmssp_calls", [])}
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
            if len(rows) < prefix_round:
                continue

            call = calls.get(call_id)
            if call is None:
                continue

            total_rounds = int(call.get("findpivot_rounds", 0))
            if total_rounds < prefix_round:
                continue

            row_r = rows[prefix_round - 1]
            row_prev = rows[prefix_round - 2] if prefix_round >= 2 else rows[prefix_round - 1]

            X.append(
                make_features(
                    l=int(call["l"]),
                    S_size=int(call["S_size"]),
                    k=k,
                    W_pref=int(row_r["W_cumulative"]),
                    W_r=int(row_r["W_i_size"]),
                    W_prev=int(row_prev["W_i_size"]),
                )
            )
            y.append(int(bool(row_r["label_P_eq_S"])))
            groups.append(file_idx)

    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.int64)
    groups = np.asarray(groups, dtype=np.int64)

    print(f"Collected {len(X)} examples")

    if len(X) == 0:
        raise RuntimeError("No training examples found.")

    return X, y, groups


def choose_threshold_for_precision(y_true, p_pred, min_precision=0.2):
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



def export_adaptive_cpp_header(path, model_dict, graph_name: str):
    def arr(xs):
        return ", ".join(f"{float(x):.17g}" for x in xs)

    namespace_name = f"pscase_online_{cpp_identifier(graph_name)}"

    text = f"""#pragma once
#include <array>
#include <cmath>
#include <algorithm>

namespace {namespace_name} {{

struct State {{
    double delta_b = 0.0;
}};

struct Decision {{
    double prob = 0.0;
    bool pred = false;
}};

struct Model {{
    static constexpr int kNumFeatures = 5;

    std::array<double, kNumFeatures> mean = {{{arr(model_dict["mean"])}}};
    std::array<double, kNumFeatures> scale = {{{arr(model_dict["scale"])}}};
    std::array<double, kNumFeatures> weights = {{{arr(model_dict["weights"])}}};
    double bias = {float(model_dict["bias"]):.17g};
    double threshold = {float(model_dict["threshold"]):.17g};

    int prefix_round = {int(model_dict["prefix_round"])};
    double eta = {float(model_dict["eta"]):.17g};
    double l2 = {float(model_dict["l2"]):.17g};
    double bias_clip = {float(model_dict["bias_clip"]):.17g};

    static double sigmoid(double z) {{
        if (z >= 0.0) {{
            const double ez = std::exp(-z);
            return 1.0 / (1.0 + ez);
        }}
        const double ez = std::exp(z);
        return ez / (1.0 + ez);
    }}

    double base_logit(
        int l,
        int S_size,
        int k,
        int W_pref,
        int W_r,
        int W_prev
    ) const {{
        const double denom = std::max(
            1.0,
            static_cast<double>(k) * static_cast<double>(S_size)
        );

        std::array<double, kNumFeatures> x = {{
            static_cast<double>(l),
            std::log1p(static_cast<double>(S_size)),
            static_cast<double>(W_pref) / denom,
            static_cast<double>(W_r) / denom,
            static_cast<double>(W_r) / std::max(1.0, static_cast<double>(W_prev))
        }};

        double z = bias;
        for (int i = 0; i < kNumFeatures; ++i) {{
            const double s = (scale[i] == 0.0 ? 1.0 : scale[i]);
            const double xs = (x[i] - mean[i]) / s;
            z += weights[i] * xs;
        }}
        return z;
    }}

    Decision infer(
        const State& st,
        int l,
        int S_size,
        int k,
        int W_pref,
        int W_r,
        int W_prev
    ) const {{
        const double z = base_logit(l, S_size, k, W_pref, W_r, W_prev) + st.delta_b;
        const double p = sigmoid(z);
        return Decision{{p, p >= threshold}};
    }}

    void update(State& st, double prob, int label) const {{
        const double y = static_cast<double>(label);
        const double grad = (prob - y) + l2 * st.delta_b;
        st.delta_b -= eta * grad;
        st.delta_b = std::clamp(st.delta_b, -bias_clip, bias_clip);
    }}
}};

}} // namespace {namespace_name}
"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)



def save_val_metrics_json(y_val, p_val, threshold, out_path):
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

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"Wrote {out_path}")
    print(json.dumps(metrics, indent=2))



def train_online(root_dir: str, graph_name: str, prefix_round: int = 2):
    X, y, groups = load_dataset(root_dir, prefix_round=prefix_round)

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
            "log1p_S_size",
            "prefix_frac",
            "last_frac",
            "growth",
        ],
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist(),
        "weights": clf.coef_[0].tolist(),
        "bias": float(clf.intercept_[0]),
        "threshold": float(threshold),
        "prefix_round": int(prefix_round),
        "eta": 0.02,
        "l2": 1e-4,
        "bias_clip": 2.0,
    }

    json_path = OUTPUT_DIR / f"{graph_name}_online.json"
    header_path = OUTPUT_DIR / f"{graph_name}_online.hpp"

    save_val_metrics_json(y_val, p_val, threshold, out_path=json_path)
    export_adaptive_cpp_header(header_path, model, graph_name)
    print(f"Wrote {header_path}")


# Compatibility alias if you want the same call shape as the offline script.
train_adaptive = train_online


if __name__ == "__main__":
    for graph_name in GRAPH_NAMES:
        train_online(f"../../experiments/5k_ml/{graph_name}", graph_name)
