#pragma once

#include <chrono>
#include <random>
#include <vector>
#include <optional>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <limits>

struct timerT {
  using clock = std::chrono::steady_clock;
  clock::time_point begin{}, end{};

  timerT() { start(); }

  void start() { begin = clock::now(); }
  void stop()  { end   = clock::now(); }

  double elapsed_ms() const {
    return std::chrono::duration<double, std::milli>(end - begin).count();
  }
};

struct GraphRow {
  int graph_id = 0;
  int n = 0;
  long long m = 0;
  double max_dist = 0.0;
};

struct BMSSPCallRow {
  int call_id = -1;
  int graph_id = 0;
  int parent_call_id = -1;
  int depth = 0;
  int l = 0;

  double B_in = 0.0;
  double B_out = 0.0;

  int S_size = 0;
  int U_size = 0;
  int P_size = 0;
  int W_size = 0;

  bool status = false; // true = successful, false = partial

  double dhat_S_min = 0.0;
  double dhat_S_mean = 0.0;
  double dhat_S_max = 0.0;
  double dhat_S_std = 0.0;

  long long edges_relaxed = 0;
  long long block_pulls = 0;
  long long block_inserts = 0;
  long long batch_prepends = 0;
  int findpivot_rounds = 0;

  std::optional<double> oracle_B_star = std::nullopt;
  double label_B = 0.0;
  bool label_P_eq_S = false;
};

struct FindPivotsRoundRow {
  int call_id = -1;
  int round_idx = 0;
  int W_i_size = 0;
  int W_cumulative = 0;
  long long relax_attempts = 0;
  long long relax_successes = 0;
  int active_owners = 0;
  double top_owner_mass = 0.0; // stored as a fraction in [0,1]
  double owner_entropy = 0.0;
  bool label_P_eq_S = false;
};

struct PivotSourceRow {
  int call_id = -1;
  int source_id = -1;
  double dhat_s = 0.0;
  int rank_in_S = 0;
  int prefix_owner_count = 0; // by default this is the count after round 1
  int final_f_s = 0;
  bool heavy_label = false;
  bool pivot_label = false;
};

struct BMSSPStats {
  int graph_id = 0;

  std::vector<GraphRow> graphs;
  std::vector<BMSSPCallRow> bmssp_calls;
  std::vector<FindPivotsRoundRow> findpivots_rounds;
  std::vector<PivotSourceRow> pivot_sources;

  void clear_tables() {
    graphs.clear();
    bmssp_calls.clear();
    findpivots_rounds.clear();
    pivot_sources.clear();
  }

  static double entropy_from_counts(const std::vector<int>& counts) {
    long long total = 0;
    for (int x : counts) total += x;
    if (total <= 0) return 0.0;

    double h = 0.0;
    for (int x : counts) {
      if (x <= 0) continue;
      const double p = static_cast<double>(x) / static_cast<double>(total);
      h -= p * std::log(p);
    }
    return h;
  }
};

inline int randomBit(double p) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::bernoulli_distribution dist(p);
  return dist(gen);
}
