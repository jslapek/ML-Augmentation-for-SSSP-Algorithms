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


# Tiny online-feasible feature map.
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


def sigmoid(z):
    z = np.asarray(z, dtype=np.float64)
    out = np.empty_like(z)
    pos = z >= 0.0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


def simulate_online_updates(
    logits: np.ndarray,
    y_true: np.ndarray,
    threshold: float,
    eta: float,
    l2: float,
    bias_clip: float,
):
    logits = np.asarray(logits, dtype=np.float64)
    y_true = np.asarray(y_true, dtype=np.int64)

    delta_b = 0.0
    probs = []
    preds = []
    deltas = []

    for z_base, label in zip(logits, y_true):
        z = float(z_base + delta_b)
        p = float(sigmoid(np.array([z]))[0])
        probs.append(p)
        preds.append(int(p >= threshold))
        deltas.append(delta_b)

        grad = (p - float(label)) + float(l2) * delta_b
        delta_b -= float(eta) * grad
        delta_b = max(-float(bias_clip), min(float(bias_clip), delta_b))

    return {
        "prob": np.asarray(probs, dtype=np.float64),
        "pred": np.asarray(preds, dtype=np.int64),
        "delta": np.asarray(deltas, dtype=np.float64),
        "final_delta": float(delta_b),
    }


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

    namespace_name = f"countmin_online_{cpp_identifier(graph_name)}"

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
    static constexpr int kNumFeatures = {len(model_dict['feature_names'])};

    std::array<double, kNumFeatures> mean = {{{arr(model_dict['mean'])}}};
    std::array<double, kNumFeatures> scale = {{{arr(model_dict['scale'])}}};
    std::array<double, kNumFeatures> weights = {{{arr(model_dict['weights'])}}};
    double bias = {float(model_dict['bias']):.17g};
    double threshold = {float(model_dict['threshold']):.17g};

    int prefix_round = {int(model_dict['prefix_round'])};
    double eta = {float(model_dict['eta']):.17g};
    double l2 = {float(model_dict['l2']):.17g};
    double bias_clip = {float(model_dict['bias_clip']):.17g};

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
        return z;
    }}

    Decision infer(
        const State& st,
        int l,
        int S_size,
        int k,
        int prefix_owner_count,
        int rank_in_S,
        int W_prefix,
        double top_owner_mass
    ) const {{
        const double z = base_logit(
            l,
            S_size,
            k,
            prefix_owner_count,
            rank_in_S,
            W_prefix,
            top_owner_mass
        ) + st.delta_b;
        const double p = sigmoid(z);
        return Decision{{p, p >= threshold}};
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
        return infer(State{{}}, l, S_size, k, prefix_owner_count, rank_in_S, W_prefix, top_owner_mass).prob;
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
        return infer(State{{}}, l, S_size, k, prefix_owner_count, rank_in_S, W_prefix, top_owner_mass).pred;
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


def summarize_metrics(y_true, p_pred, w_true, threshold):
    y_true = np.asarray(y_true, dtype=np.int64)
    p_pred = np.asarray(p_pred, dtype=np.float64)
    w_true = np.asarray(w_true, dtype=np.float64)
    y_pred = (p_pred >= float(threshold)).astype(np.int64)

    positive_rate = float(np.mean(y_true)) if len(y_true) > 0 else 0.0
    predicted_fraction = float(np.mean(y_pred)) if len(y_pred) > 0 else 0.0
    tp_mass = float(np.sum(w_true * y_true * y_pred))
    pred_mass = float(np.sum(w_true * y_pred))
    pos_mass = float(np.sum(w_true * y_true))
    total_mass = float(np.sum(w_true))
    precision = float(precision_score(y_true, y_pred, zero_division=0)) if len(y_true) > 0 else 0.0

    return {
        "n_examples": int(len(y_true)),
        "n_positives": int(np.sum(y_true)),
        "n_negatives": int(np.sum(y_true == 0)),
        "n_predicted_positives": int(np.sum(y_pred)),
        "n_predicted_negatives": int(np.sum(y_pred == 0)),
        "accuracy": float(accuracy_score(y_true, y_pred)) if len(y_true) > 0 else 0.0,
        "precision": precision,
        "recall": float(recall_score(y_true, y_pred, zero_division=0)) if len(y_true) > 0 else 0.0,
        "f1": float(f1_score(y_true, y_pred, zero_division=0)) if len(y_true) > 0 else 0.0,
        "positive_rate": positive_rate,
        "predicted_fraction": predicted_fraction,
        "precision_lift_over_base_rate": (float(precision / positive_rate) if positive_rate > 0.0 else 0.0),
        "weighted_precision_mass": (tp_mass / pred_mass if pred_mass > 0.0 else 0.0),
        "weighted_positive_mass_recall": (tp_mass / pos_mass if pos_mass > 0.0 else 0.0),
        "captured_total_mass": (pred_mass / total_mass if total_mass > 0.0 else 0.0),
        "pr_auc_weighted": safe_average_precision(y_true, p_pred, sample_weight=w_true),
    }


def save_val_metrics_json(
    y_val,
    cold_prob,
    online_prob,
    w_val,
    threshold,
    out_path,
    model_meta,
    online_final_delta,
):
    metrics = {
        "threshold": float(threshold),
        "online_final_delta": float(online_final_delta),
        "cold": summarize_metrics(y_val, cold_prob, w_val, threshold),
        "online": summarize_metrics(y_val, online_prob, w_val, threshold),
    }
    metrics.update(model_meta)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"Wrote {out_path}")
    print(json.dumps(metrics, indent=2))


def train_online(
    root_dir: str,
    graph_name: str,
    prefix_round: int = 2,
    label_key: str = "pivot_label",
    weight_key: str = "final_f_s",
    skip_ps_case: bool = True,
    max_predicted_fraction: float = 0.20,
    eta: float = 0.02,
    l2: float = 1e-4,
    bias_clip: float = 2.0,
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

    scaler = StandardScaler()
    X_train_s_full = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)

    X_fit, y_fit, w_fit = subsample_train(
        X_train_s_full,
        y_train,
        w_train,
        max_train_examples=max_train_examples,
        seed=0,
    )

    fit_labels = np.unique(y_fit)
    if len(fit_labels) < 2:
        weights = np.zeros(X_train_s_full.shape[1], dtype=np.float64)
        bias = 12.0 if int(fit_labels[0]) == 1 else -12.0
    else:
        clf = LogisticRegression(
            penalty="l2",
            C=1.0,
            solver="liblinear",
            max_iter=200,
        )
        clf.fit(X_fit, y_fit, sample_weight=w_fit)
        weights = clf.coef_[0].astype(np.float64)
        bias = float(clf.intercept_[0])

    cold_logit = X_val_s @ weights + bias
    cold_prob = sigmoid(cold_logit)
    threshold = choose_budget_threshold(cold_prob, max_predicted_fraction=max_predicted_fraction)

    online = simulate_online_updates(
        logits=cold_logit,
        y_true=y_val,
        threshold=threshold,
        eta=eta,
        l2=l2,
        bias_clip=bias_clip,
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
        "eta": float(eta),
        "l2": float(l2),
        "bias_clip": float(bias_clip),
        "label_key": label_key,
        "weight_key": weight_key,
        "skip_ps_case": bool(skip_ps_case),
        "max_predicted_fraction": float(max_predicted_fraction),
        "max_train_examples": int(max_train_examples),
    }

    json_path = OUTPUT_DIR / f"{graph_name}_countmin_online.json"
    header_path = OUTPUT_DIR / f"{graph_name}_countmin_online.hpp"

    save_val_metrics_json(
        y_val=y_val,
        cold_prob=cold_prob,
        online_prob=online["prob"],
        w_val=w_val,
        threshold=threshold,
        out_path=json_path,
        model_meta={
            "prefix_round": int(prefix_round),
            "label_key": label_key,
            "weight_key": weight_key,
            "skip_ps_case": bool(skip_ps_case),
            "max_predicted_fraction": float(max_predicted_fraction),
            "max_train_examples": int(max_train_examples),
            "eta": float(eta),
            "l2": float(l2),
            "bias_clip": float(bias_clip),
        },
        online_final_delta=online["final_delta"],
    )
    export_adaptive_cpp_header(header_path, model, graph_name)
    print(f"Wrote {header_path}")


train_adaptive = train_online
train_countmin_online = train_online


if __name__ == "__main__":
    for graph_name in GRAPH_NAMES:
        train_online(f"../../experiments/5k_ml/{graph_name}", graph_name)
