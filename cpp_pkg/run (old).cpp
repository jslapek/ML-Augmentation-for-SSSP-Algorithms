// [[Rcpp::plugins(cpp20)]]
#include <Rcpp.h>
using namespace Rcpp;

#include "utils.hpp"
#include "common.hpp"
#include "algs/bmssp.hpp"
#include "algs/bmssp_timed.hpp"
#include "algs/bmssp_ml_theory.hpp"
#include "algs/bmssp_timed_ppred.hpp"
#include "algs/bmssp_timed_ppred_k.hpp"
#include "algs/bmssp_timed_cpred_dedup.hpp"
#include "algs/bmssp_timed_cpred_hybrid.hpp"
#include "algs/bmssp_timed_cpred_npf.hpp"
#include "algs/bmssp_timed_cpred_ob.hpp"
#include "algs/bmssp_timed_cpred_hybrid_k.hpp"
#include "algs/bmssp_timed_cpred_npf_k.hpp"
#include "algs/bmssp_timed_cpred_ob_k.hpp"
#include "algs/bmssp_timed_cpred_dedup_k.hpp"
#include "algs/bmssp_bounded (unoptimised).hpp"
#include "algs/bmssp_bounded (optimised).hpp"
#include "algs/bmssp_bounded_optimised_k.hpp"
#include "algs/dijkstra.hpp"

#include <fstream>
#include <iostream>
#include <vector>
#include <array>
#include <algorithm>
#include <json.hpp>
#include <filesystem>

using json = nlohmann::json;
namespace fs = std::filesystem;

using distT = long long;

NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(
  Stats,
  time_dijkstra,
  time_full,
  time_find_pivot,
  time_base_case,
  time_D_op,
  time_bmssp,
  time_batch_prepend,
  snip_split,
  snip_lower_bound,
  snip_block_insertion,
  snip_membership_check,
  snip_deletion,
  snip_tree_construction,
  snip_relaxation
)

static bool is_number(const std::string& s) {
  if (s.empty()) return false;
  for (char c : s) {
    if (c < '0' || c > '9') return false;
  }
  return true;
}

std::map<long long, std::vector<std::string>> list_graphs_in_order(const std::string& root) {
  fs::path root_path(root);
  std::map<long long, std::vector<std::string>> size_to_graphs;

  std::vector<std::pair<long long, fs::path>> size_dirs;
  for (const auto& entry : fs::directory_iterator(root_path)) {
    if (!entry.is_directory()) continue;

    std::string name = entry.path().filename().string();
    if (!is_number(name)) continue;

    size_dirs.push_back({std::stoll(name), entry.path()});
  }

  std::sort(
    size_dirs.begin(),
    size_dirs.end(),
    [](const auto& a, const auto& b) { return a.first < b.first; }
  );

  for (const auto& [n, dir] : size_dirs) {
    auto& bucket = size_to_graphs[n];
    for (const auto& f : fs::directory_iterator(dir)) {
      if (f.is_regular_file() && f.path().extension() == ".gr") {
        bucket.push_back(f.path().string());
      }
    }
    std::sort(bucket.begin(), bucket.end());
  }

  return size_to_graphs;
}

// [[Rcpp::export]]
std::string runSearch() {
  timerT timer;
  timerT total;
  total.start();

  std::ifstream f("run_class.json");
  json cfg;
  f >> cfg;

  std::string root =
      "C:/Users/Jakub/Documents/stuff/diss/project/graphs/" +
      cfg["graph"].get<std::string>();

  auto graphs = list_graphs_in_order(root);

  std::size_t dir_count = 0;
  json jout = json::object();

  for (const auto& [dir, gs] : graphs) {
    if (++dir_count >= 14) break;
    json& dir_bucket = jout[std::to_string(dir)];

    for (std::size_t gi = 0; gi < gs.size(); ++gi) {
      auto [adj, m] = readGraph<distT>(gs[gi]);

      double dijkstra_ms = 0.0;
      {
        spp::dijkstra<distT> dijkstra(adj);
        timer.start();
        auto [d_d, p_d] = dijkstra.execute(0);
        timer.stop();
        dijkstra_ms = timer.elapsed_ms();
      }

      spp_timed::bmssp<distT> bmssp(adj);
      bmssp.stats.time_dijkstra = dijkstra_ms;
      bmssp.prepare_graph(false);

      timer.start();
      auto [d, p] = bmssp.execute(0);
      timer.stop();

      bmssp.stats.update_time_full(timer.elapsed_ms());
      std::cout << "BMSSP time: " << bmssp.stats.time_full << "ms\n";

      dir_bucket[fs::path(gs[gi]).filename().string()] = bmssp.stats;
    }
  }

  std::ofstream out("experiments/run_stats.json");
  out << jout.dump(2);

  total.stop();
  std::cout << "Total time (" << cfg["graph"] << "): " << total.elapsed_ms() << "ms\n";
  return "done";
}

struct PpredConfig {
  const char* name;
  bool pred_c;
  bool pred_pivots;
  bool local_search;
  int BF_steps;
};

template <typename Algo, typename Graph>
Stats run_base_alg(const Graph& adj, double dijkstra_ms, distT /*max_dist*/ = 0) {
  timerT timer;

  Algo alg(adj);
  alg.stats.time_dijkstra = dijkstra_ms;
  alg.prepare_graph(false);

  timer.start();
  auto [d, p] = alg.execute(0);
  timer.stop();

  alg.stats.update_time_full(timer.elapsed_ms());
  return alg.stats;
}

template <typename Algo, typename Graph>
Stats run_bounded_alg(const Graph& adj, double dijkstra_ms, distT max_dist) {
  timerT timer;

  Algo alg(adj);
  alg.stats.time_dijkstra = dijkstra_ms;
  alg.set_threshold_schedule({max_dist + 1});
  alg.prepare_graph(false);

  timer.start();
  auto [d, p] = alg.execute(0);
  timer.stop();

  alg.stats.update_time_full(timer.elapsed_ms());
  return alg.stats;
}

template <typename Algo>
struct cpred_namespace_config;

// dedup
template <>
struct cpred_namespace_config<spp_timed_cpred_dedup::bmssp<distT>> {
  static void apply(bool pred_c, bool pred_pivots, bool local_search, int BF_steps) {
    spp_timed_cpred_dedup::set_pred_c(pred_c);
    spp_timed_cpred_dedup::set_pred_pivots(pred_pivots);
    spp_timed_cpred_dedup::set_local_search(local_search);
    spp_timed_cpred_dedup::set_BF_steps(BF_steps);
  }
};

// hybrid
template <>
struct cpred_namespace_config<spp_timed_cpred_hybrid::bmssp<distT>> {
  static void apply(bool pred_c, bool pred_pivots, bool local_search, int BF_steps) {
    spp_timed_cpred_hybrid::set_pred_c(pred_c);
    spp_timed_cpred_hybrid::set_pred_pivots(pred_pivots);
    spp_timed_cpred_hybrid::set_local_search(local_search);
    spp_timed_cpred_hybrid::set_BF_steps(BF_steps);
  }
};

// npf
template <>
struct cpred_namespace_config<spp_timed_cpred_npf::bmssp<distT>> {
  static void apply(bool pred_c, bool pred_pivots, bool local_search, int BF_steps) {
    spp_timed_cpred_npf::set_pred_c(pred_c);
    spp_timed_cpred_npf::set_pred_pivots(pred_pivots);
    spp_timed_cpred_npf::set_local_search(local_search);
    spp_timed_cpred_npf::set_BF_steps(BF_steps);
  }
};

// ob
template <>
struct cpred_namespace_config<spp_timed_cpred_ob::bmssp<distT>> {
  static void apply(bool pred_c, bool pred_pivots, bool local_search, int BF_steps) {
    spp_timed_cpred_ob::set_pred_c(pred_c);
    spp_timed_cpred_ob::set_pred_pivots(pred_pivots);
    spp_timed_cpred_ob::set_local_search(local_search);
    spp_timed_cpred_ob::set_BF_steps(BF_steps);
  }
};

// hybrid_k
template <>
struct cpred_namespace_config<spp_timed_cpred_hybrid_k::bmssp<distT>> {
  static void apply(bool pred_c, bool pred_pivots, bool local_search, int BF_steps) {
    spp_timed_cpred_hybrid_k::set_pred_c(pred_c);
    spp_timed_cpred_hybrid_k::set_pred_pivots(pred_pivots);
    spp_timed_cpred_hybrid_k::set_local_search(local_search);
    spp_timed_cpred_hybrid_k::set_BF_steps(BF_steps);
  }
};

// npf_k
template <>
struct cpred_namespace_config<spp_timed_cpred_npf_k::bmssp<distT>> {
  static void apply(bool pred_c, bool pred_pivots, bool local_search, int BF_steps) {
    spp_timed_cpred_npf_k::set_pred_c(pred_c);
    spp_timed_cpred_npf_k::set_pred_pivots(pred_pivots);
    spp_timed_cpred_npf_k::set_local_search(local_search);
    spp_timed_cpred_npf_k::set_BF_steps(BF_steps);
  }
};

// ob_k
template <>
struct cpred_namespace_config<spp_timed_cpred_ob_k::bmssp<distT>> {
  static void apply(bool pred_c, bool pred_pivots, bool local_search, int BF_steps) {
    spp_timed_cpred_ob_k::set_pred_c(pred_c);
    spp_timed_cpred_ob_k::set_pred_pivots(pred_pivots);
    spp_timed_cpred_ob_k::set_local_search(local_search);
    spp_timed_cpred_ob_k::set_BF_steps(BF_steps);
  }
};

// dedup_k
template <>
struct cpred_namespace_config<spp_timed_cpred_dedup_k::bmssp<distT>> {
  static void apply(bool pred_c, bool pred_pivots, bool local_search, int BF_steps) {
    spp_timed_cpred_dedup_k::set_pred_c(pred_c);
    spp_timed_cpred_dedup_k::set_pred_pivots(pred_pivots);
    spp_timed_cpred_dedup_k::set_local_search(local_search);
    spp_timed_cpred_dedup_k::set_BF_steps(BF_steps);
  }
};

template <typename Algo, typename Graph>
Stats run_cpred_alg(
    const Graph& adj,
    double dijkstra_ms,
    bool pred_c,
    bool pred_pivots,
    bool local_search,
    int BF_steps) {
  timerT timer;

  cpred_namespace_config<Algo>::apply(pred_c, pred_pivots, local_search, BF_steps);

  Algo alg(adj);
  alg.stats.time_dijkstra = dijkstra_ms;
  alg.prepare_graph(false);

  timer.start();
  auto [d, p] = alg.execute(0);
  timer.stop();

  alg.stats.update_time_full(timer.elapsed_ms());
  return alg.stats;
}

template <typename Algo, typename Graph, std::size_t N>
void run_cpred_family(
    const Graph& adj,
    json& family_bucket,
    double dijkstra_ms,
    const std::array<PpredConfig, N>& cases) {
  for (const auto& cfg : cases) {
    family_bucket[cfg.name] = run_cpred_alg<Algo>(
      adj,
      dijkstra_ms,
      cfg.pred_c,
      cfg.pred_pivots,
      cfg.local_search,
      cfg.BF_steps
    );
  }
}

template <typename Graph>
void run_standard_algorithms(
    const Graph& adj,
    json& graph_bucket,
    double dijkstra_ms,
    distT max_dist) {
  json& standard = graph_bucket["standard"];

  standard["bmssp_timed"] =
      run_base_alg<spp_timed::bmssp<distT>>(adj, dijkstra_ms, max_dist);

  standard["bmssp_timed_ppred"] =
      run_base_alg<spp_timed_ppred::bmssp<distT>>(adj, dijkstra_ms, max_dist);

  standard["bmssp_timed_ppred_k"] =
      run_base_alg<spp_timed_ppred_k::bmssp<distT>>(adj, dijkstra_ms, max_dist);

  standard["bmssp_bounded_opt"] =
      run_bounded_alg<spp_bounded_opt::bmssp<distT>>(adj, dijkstra_ms, max_dist);

  standard["bmssp_bounded_unopt"] =
      run_bounded_alg<spp_bounded_unopt::bmssp<distT>>(adj, dijkstra_ms, max_dist);
}

template <typename Graph>
void run_all_algorithms(
    const Graph& adj,
    json& graph_bucket,
    double dijkstra_ms,
    distT max_dist = 0) {
  run_standard_algorithms(adj, graph_bucket, dijkstra_ms, max_dist);

  // Shared parameter sets for:
  // [ssp_cpred_dedup, ssp_cpred_hybrid, ssp_cpred_npf, ssp_cpred_ob,
  //  ssp_cpred_hybrid_k, ssp_cpred_npf_k, ssp_cpred_ob_k, ssp_cpred_dedup_k]
  static const std::array<PpredConfig, 4> shared_cpred_cases{{
    { "ppred_c",        true,  false, false, -1 },
    { "ppred_pivots",   true,  true,  false, -1 },
    { "ppred_c_pivots", false, true,  false, -1 },
    { "pred_c_k1",      true,  true,  true,   1 }
  }};

  // Extra parameter sets only for [ssp_cpred_dedup_k]
  static const std::array<PpredConfig, 10> dedup_k_extra_cases{{
    { "pred_c_k0", true,  true, true, 0 },
    { "pred_c_k1", true,  true, true, 1 },
    { "pred_c_k2", true,  true, true, 2 },
    { "pred_c_k3", true,  true, true, 3 },
    { "pred_c_k4", true,  true, true, 4 },
    { "pred_k0",   false, true, true, 0 },
    { "pred_k1",   false, true, true, 1 },
    { "pred_k2",   false, true, true, 2 },
    { "pred_k3",   false, true, true, 3 },
    { "pred_k4",   false, true, true, 4 }
  }};

  json& cpred = graph_bucket["cpred"];

  run_cpred_family<spp_timed_cpred_dedup::bmssp<distT>>(
    adj, cpred["cpred_dedup"], dijkstra_ms, shared_cpred_cases
  );

  run_cpred_family<spp_timed_cpred_hybrid::bmssp<distT>>(
    adj, cpred["cpred_hybrid"], dijkstra_ms, shared_cpred_cases
  );

  run_cpred_family<spp_timed_cpred_npf::bmssp<distT>>(
    adj, cpred["cpred_npf"], dijkstra_ms, shared_cpred_cases
  );

  run_cpred_family<spp_timed_cpred_ob::bmssp<distT>>(
    adj, cpred["cpred_ob"], dijkstra_ms, shared_cpred_cases
  );

  run_cpred_family<spp_timed_cpred_hybrid_k::bmssp<distT>>(
    adj, cpred["cpred_hybrid_k"], dijkstra_ms, shared_cpred_cases
  );

  run_cpred_family<spp_timed_cpred_npf_k::bmssp<distT>>(
    adj, cpred["cpred_npf_k"], dijkstra_ms, shared_cpred_cases
  );

  run_cpred_family<spp_timed_cpred_ob_k::bmssp<distT>>(
    adj, cpred["cpred_ob_k"], dijkstra_ms, shared_cpred_cases
  );

  run_cpred_family<spp_timed_cpred_dedup_k::bmssp<distT>>(
    adj, cpred["cpred_dedup_k"], dijkstra_ms, shared_cpred_cases
  );

  run_cpred_family<spp_timed_cpred_dedup_k::bmssp<distT>>(
    adj, cpred["cpred_dedup_k_extra"], dijkstra_ms, dedup_k_extra_cases
  );
}

// [[Rcpp::export]]
std::string runGlobalSearch() {
  timerT timer;
  timerT total;
  total.start();

  std::ifstream f("run_class.json");
  json cfg;
  f >> cfg;

  std::string root =
      "C:/Users/Jakub/Documents/stuff/diss/project/graphs/" +
      cfg["graph"].get<std::string>();

  auto graphs = list_graphs_in_order(root);

  std::size_t dir_count = 0;
  json jout = json::object();

  for (const auto& [dir, gs] : graphs) {
    // if (dir != 33554432) continue;
    if (++dir_count >= 11) break;

    json& dir_bucket = jout[std::to_string(dir)];

    for (std::size_t gi = 0; gi < gs.size(); ++gi) {
      const std::string& graph_path = gs[gi];
      const std::string graph_name = fs::path(graph_path).filename().string();

      auto [adj, m] = readGraph<distT>(graph_path);

      double dijkstra_ms = 0.0;
      distT max_dist = 0;

      {
        spp::dijkstra<distT> dijkstra(adj);
        timer.start();
        auto [d_d, p_d] = dijkstra.execute(0);
        timer.stop();

        dijkstra_ms = timer.elapsed_ms();
        max_dist = *std::max_element(d_d.begin(), d_d.end());
      }

      json& graph_bucket = dir_bucket[graph_name];
      graph_bucket["dijkstra_ms"] = dijkstra_ms;
      graph_bucket["max_dist"] = max_dist;

      run_all_algorithms(adj, graph_bucket, dijkstra_ms, max_dist);
    }
  }

  std::ofstream out("experiments/run_stats.json");
  out << jout.dump(2);

  total.stop();
  std::cout << "Total time (" << cfg["graph"] << "): "
            << total.elapsed_ms() << "ms\n";

  return "done";
}
