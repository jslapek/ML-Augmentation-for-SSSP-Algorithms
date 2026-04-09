import json
import math
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler


OUTPUT_DIR = Path("models/countmin")
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


def safe_div(num: float, den: float) -> float:
    den = float(den)
    if abs(den) < 1e-12:
        return 0.0 if abs(num) < 1e-12 else float(num)
    return float(num) / den


# Keep the model intentionally tiny and online-feasible.
# All inputs are available after a very short FindPivots prefix.
def make_features(
    l: int,
    S_size: int,
    k: int,
    prefix_owner_count: int,
    rank_in_S: int,
    W_prefix: int,
    top_owner_mass: float,
) -> np.ndarray:
    denom = max(1.0, float(k) * float(S_size))
    return np.array(
        [
            float(l),
            math.log1p(float(S_size)),
            safe_div(prefix_owner_count, max(1, W_prefix)),
            safe_div(rank_in_S, max(1, S_size)),
            safe_div(W_prefix, denom),
            float(top_owner_mass),
        ],
        dtype=np.float64,
    )


def load_dataset(
    root_dir: str,
    prefix_round: int = 2,
    label_key: str = "pivot_label",
    weight_key: str = "final_f_s",
    skip_ps_case: bool = True,
):
    X = []
    y = []
    w = []
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

        n = int(graphs[0]["n"])
        k = compute_k(n)

        calls = {int(row["call_id"]): row for row in stats.get("bmssp_calls", [])}
        rounds_by_call = {}
        for row in stats.get("findpivots_rounds", []):
            call_id = int(row["call_id"])
            rounds_by_call.setdefault(call_id, []).append(row)

        for rows in rounds_by_call.values():
            rows.sort(key=lambda z: int(z["round_idx"]))

        for row in stats.get("pivot_sources", []):
            call_id = int(row["call_id"])
            call = calls.get(call_id)
            if call is None:
                continue
            if skip_ps_case and bool(call.get("label_P_eq_S", False)):
                continue

            call_rounds = rounds_by_call.get(call_id, [])
            if len(call_rounds) < prefix_round:
                continue
            if int(call.get("findpivot_rounds", 0)) < prefix_round:
                continue

            prefix_row = call_rounds[prefix_round - 1]
            X.append(
                make_features(
                    l=int(call["l"]),
                    S_size=int(call["S_size"]),
                    k=k,
                    prefix_owner_count=int(row.get("prefix_owner_count", 0)),
                    rank_in_S=int(row.get("rank_in_S", 0)),
                    W_prefix=int(prefix_row.get("W_cumulative", 0)),
                    top_owner_mass=float(prefix_row.get("top_owner_mass", 0.0)),
                )
            )
            y.append(int(bool(row.get(label_key, False))))
            w.append(max(1.0, float(row.get(weight_key, 1.0))))
            groups.append(file_idx)

    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.int64)
    w = np.asarray(w, dtype=np.float64)
    groups = np.asarray(groups, dtype=np.int64)

    print(f"Collected {len(X)} examples")
    if len(X) == 0:
        raise RuntimeError("No training examples found.")

    return X, y, w, groups


def subsample_train(X, y, w, max_train_examples: int, seed: int = 0):
    if max_train_examples <= 0 or len(X) <= max_train_examples:
        return X, y, w

    rng = np.random.default_rng(seed)
    pos_idx = np.flatnonzero(y == 1)
    neg_idx = np.flatnonzero(y == 0)

    if len(pos_idx) == 0 or len(neg_idx) == 0:
        keep = rng.choice(np.arange(len(X)), size=max_train_examples, replace=False)
        keep.sort()
        return X[keep], y[keep], w[keep]

    pos_frac = float(len(pos_idx)) / float(len(X))
    pos_keep_n = int(round(max_train_examples * pos_frac))
    pos_keep_n = max(1, min(len(pos_idx), pos_keep_n))
    neg_keep_n = max_train_examples - pos_keep_n
    neg_keep_n = max(1, min(len(neg_idx), neg_keep_n))

    total = pos_keep_n + neg_keep_n
    if total > max_train_examples:
        overflow = total - max_train_examples
        if pos_keep_n >= neg_keep_n and pos_keep_n - overflow >= 1:
            pos_keep_n -= overflow
        else:
            neg_keep_n = max(1, neg_keep_n - overflow)
    elif total < max_train_examples:
        extra = max_train_examples - total
        pos_room = len(pos_idx) - pos_keep_n
        neg_room = len(neg_idx) - neg_keep_n
        add_pos = min(pos_room, extra)
        pos_keep_n += add_pos
        extra -= add_pos
        neg_keep_n += min(neg_room, extra)

    pos_keep = rng.choice(pos_idx, size=pos_keep_n, replace=False)
    neg_keep = rng.choice(neg_idx, size=neg_keep_n, replace=False)
    keep = np.concatenate([pos_keep, neg_keep])
    keep.sort()
    return X[keep], y[keep], w[keep]


def choose_budget_threshold(p_pred: np.ndarray, max_predicted_fraction: float) -> float:
    p_pred = np.asarray(p_pred, dtype=np.float64)
    if len(p_pred) == 0:
        return 1.0
    q = max(0.0, min(1.0, 1.0 - float(max_predicted_fraction)))
    thr = float(np.quantile(p_pred, q))
    if q > 0.0:
        thr = min(1.0 + 1e-12, thr + 1e-12)
    return thr


def safe_average_precision(y_true: np.ndarray, p_pred: np.ndarray, sample_weight: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.int64)
    if len(y_true) == 0:
        return 0.0
    positives = int(np.sum(y_true == 1))
    negatives = int(np.sum(y_true == 0))
    if positives == 0:
        return 0.0
    if negatives == 0:
        return 1.0
    return float(average_precision_score(y_true, p_pred, sample_weight=sample_weight))


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
    def arr(xs) -> str:
        return ", ".join(f"{float(x):.17g}" for x in xs)

    namespace_name = f"countmin_offline_{cpp_identifier(graph_name)}"

    text = f"""#pragma once
#include <array>
#include <cmath>
#include <algorithm>

namespace {namespace_name} {{

struct Model {{
    static constexpr int kNumFeatures = {len(model_dict['feature_names'])};

    std::array<double, kNumFeatures> mean = {{{arr(model_dict['mean'])}}};
    std::array<double, kNumFeatures> scale = {{{arr(model_dict['scale'])}}};
    std::array<double, kNumFeatures> weights = {{{arr(model_dict['weights'])}}};
    double bias = {float(model_dict['bias']):.17g};
    double threshold = {float(model_dict['threshold']):.17g};
    int prefix_round = {int(model_dict['prefix_round'])};

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
        int S_size,
        int k,
        int prefix_owner_count,
        int rank_in_S,
        int W_prefix,
        double top_owner_mass
    ) const {{
        const double denom = std::max(
            1.0,
            static_cast<double>(k) * static_cast<double>(S_size)
        );

        std::array<double, kNumFeatures> x = {{
            static_cast<double>(l),
            std::log1p(static_cast<double>(S_size)),
            static_cast<double>(prefix_owner_count) /
                std::max(1.0, static_cast<double>(W_prefix)),
            static_cast<double>(rank_in_S) /
                std::max(1.0, static_cast<double>(S_size)),
            static_cast<double>(W_prefix) / denom,
            top_owner_mass
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
        int S_size,
        int k,
        int prefix_owner_count,
        int rank_in_S,
        int W_prefix,
        double top_owner_mass
    ) const {{
        return predict_proba(
            l,
            S_size,
            k,
            prefix_owner_count,
            rank_in_S,
            W_prefix,
            top_owner_mass
        ) >= threshold;
    }}
}};

}} // namespace {namespace_name}
"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def save_val_metrics_json(
    y_val,
    p_val,
    w_val,
    threshold,
    out_path,
    max_predicted_fraction,
    label_key,
    weight_key,
    skip_ps_case,
    prefix_round,
    max_train_examples,
    train_mode,
):
    y_val = np.asarray(y_val, dtype=np.int64)
    p_val = np.asarray(p_val, dtype=np.float64)
    w_val = np.asarray(w_val, dtype=np.float64)
    y_pred = (p_val >= threshold).astype(np.int64)

    tp_mass = float(np.sum(w_val * y_val * y_pred))
    pred_mass = float(np.sum(w_val * y_pred))
    pos_mass = float(np.sum(w_val * y_val))

    metrics = {
        "threshold": float(threshold),
        "max_predicted_fraction": float(max_predicted_fraction),
        "max_train_examples": int(max_train_examples),
        "train_mode": train_mode,
        "n_examples": int(len(y_val)),
        "n_positives": int(np.sum(y_val)),
        "n_predicted_positives": int(np.sum(y_pred)),
        "accuracy": float(accuracy_score(y_val, y_pred)),
        "precision": float(precision_score(y_val, y_pred, zero_division=0)),
        "recall": float(recall_score(y_val, y_pred, zero_division=0)),
        "f1": float(f1_score(y_val, y_pred, zero_division=0)),
        "weighted_precision_mass": (tp_mass / pred_mass if pred_mass > 0.0 else 0.0),
        "weighted_positive_mass_recall": (tp_mass / pos_mass if pos_mass > 0.0 else 0.0),
        "pr_auc_weighted": safe_average_precision(y_val, p_val, w_val),
        "label_key": label_key,
        "weight_key": weight_key,
        "skip_ps_case": bool(skip_ps_case),
        "prefix_round": int(prefix_round),
    }

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"Wrote {out_path}")
    print(json.dumps(metrics, indent=2))


def fit_linear_or_constant(X_train_s, y_train, w_train):
    unique = np.unique(y_train)
    if len(unique) >= 2:
        clf = LogisticRegression(
            penalty="l2",
            C=0.5,
            solver="liblinear",
            max_iter=200,
            random_state=0,
        )
        clf.fit(X_train_s, y_train, sample_weight=w_train)
        return "logistic", clf.coef_[0].copy(), float(clf.intercept_[0])

    only_label = int(unique[0])
    bias = 30.0 if only_label == 1 else -30.0
    weights = np.zeros(X_train_s.shape[1], dtype=np.float64)
    return f"constant_{only_label}", weights, bias


def train_countmin(
    root_dir: str,
    graph_name: str,
    prefix_round: int = 2,
    label_key: str = "pivot_label",
    weight_key: str = "final_f_s",
    skip_ps_case: bool = True,
    max_predicted_fraction: float = 0.20,
    max_train_examples: int = 200000,
):
    X, y, w, groups = load_dataset(
        root_dir=root_dir,
        prefix_round=prefix_round,
        label_key=label_key,
        weight_key=weight_key,
        skip_ps_case=skip_ps_case,
    )

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
    w_train, w_val = w[train_idx], w[val_idx]

    X_train, y_train, w_train = subsample_train(
        X_train,
        y_train,
        w_train,
        max_train_examples=max_train_examples,
        seed=0,
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)

    train_mode, weights, bias = fit_linear_or_constant(X_train_s, y_train, w_train)

    z_val = X_val_s @ weights + bias
    p_val = 1.0 / (1.0 + np.exp(-np.clip(z_val, -60.0, 60.0)))
    threshold = choose_budget_threshold(
        p_pred=p_val,
        max_predicted_fraction=max_predicted_fraction,
    )

    model = {
        "feature_names": [
            "l",
            "log1p_S_size",
            "prefix_owner_share",
            "rank_frac",
            "W_prefix_frac",
            "top_owner_mass",
        ],
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist(),
        "weights": weights.tolist(),
        "bias": float(bias),
        "threshold": float(threshold),
        "prefix_round": int(prefix_round),
        "label_key": label_key,
        "weight_key": weight_key,
        "skip_ps_case": bool(skip_ps_case),
        "max_predicted_fraction": float(max_predicted_fraction),
        "max_train_examples": int(max_train_examples),
        "train_mode": train_mode,
    }

    json_path = OUTPUT_DIR / f"{graph_name}_countmin_offline.json"
    header_path = OUTPUT_DIR / f"{graph_name}_countmin_offline.hpp"

    save_val_metrics_json(
        y_val=y_val,
        p_val=p_val,
        w_val=w_val,
        threshold=threshold,
        out_path=json_path,
        max_predicted_fraction=max_predicted_fraction,
        label_key=label_key,
        weight_key=weight_key,
        skip_ps_case=skip_ps_case,
        prefix_round=prefix_round,
        max_train_examples=max_train_examples,
        train_mode=train_mode,
    )
    export_fixed_cpp_header(header_path, model, graph_name)
    print(f"Wrote {header_path}")


train_fixed = train_countmin


if __name__ == "__main__":
    for graph_name in GRAPH_NAMES:
        train_countmin(f"../../experiments/5k_ml/{graph_name}", graph_name)
