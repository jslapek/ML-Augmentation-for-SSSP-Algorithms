// [[Rcpp::plugins(cpp20)]]
#include <Rcpp.h>
using namespace Rcpp;

#include "utils.hpp"
#include "common.hpp"
#include "algs/bmssp.hpp"
#include "algs/bmssp_timed.hpp"
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

/////

#include "algs/bmsspf.hpp"
// #include "algs/bmssp_learned_idx.hpp"
// #include "algs/bmssp_lrnd_idx.hpp"  <--- could not get to work
#include "algs/bmssp_lapq.hpp"

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
    if (++dir_count >= 10) break;
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

      // pscase_type = ["randomD", "randomE", "randomG", "randomH", "randomT", "RD", "RF", "mix_real", "mix_gen", "mix_all"]
      // pscase_predictor = ["false", "online", "offline", "blank"]
      // frontier = ["bpq", "lapq"]
      // countmin_predictor = ["false", "dedup", "ob", "npf", "hybrid"]
      // countmin_type = ["false", "online", "offline", "blank"]
      // BF_steps = int
      spp_bmsspf::bmssp<distT> bmssp(adj, "randomT", "offline", "bpq", "ob", "online", 0);
      // spp_timed::bmssp<distT> bmssp(adj);
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

  // Every cpred algorithm runs every one of these combinations.
  static const std::array<PpredConfig, 14> all_cpred_cases{{
    { "ppred_c",        true,  false, false, -1 },
    { "ppred_pivots",   true,  true,  false, -1 },
    { "ppred_c_pivots", false, true,  false, -1 },
    { "pred_c_k1",      true,  true,  true,   1 },

    { "pred_c_k0",      true,  true,  true,   0 },
    { "pred_c_k1",      true,  true,  true,   1 },
    { "pred_c_k2",      true,  true,  true,   2 },
    { "pred_c_k3",      true,  true,  true,   3 },
    { "pred_c_k4",      true,  true,  true,   4 },

    { "pred_k0",        false, true,  true,   0 },
    { "pred_k1",        false, true,  true,   1 },
    { "pred_k2",        false, true,  true,   2 },
    { "pred_k3",        false, true,  true,   3 },
    { "pred_k4",        false, true,  true,   4 }
  }};

  json& cpred = graph_bucket["cpred"];

  run_cpred_family<spp_timed_cpred_dedup::bmssp<distT>>(
    adj, cpred["cpred_dedup"], dijkstra_ms, all_cpred_cases
  );

  run_cpred_family<spp_timed_cpred_hybrid::bmssp<distT>>(
    adj, cpred["cpred_hybrid"], dijkstra_ms, all_cpred_cases
  );

  run_cpred_family<spp_timed_cpred_npf::bmssp<distT>>(
    adj, cpred["cpred_npf"], dijkstra_ms, all_cpred_cases
  );

  run_cpred_family<spp_timed_cpred_ob::bmssp<distT>>(
    adj, cpred["cpred_ob"], dijkstra_ms, all_cpred_cases
  );

  run_cpred_family<spp_timed_cpred_hybrid_k::bmssp<distT>>(
    adj, cpred["cpred_hybrid_k"], dijkstra_ms, all_cpred_cases
  );

  run_cpred_family<spp_timed_cpred_npf_k::bmssp<distT>>(
    adj, cpred["cpred_npf_k"], dijkstra_ms, all_cpred_cases
  );

  run_cpred_family<spp_timed_cpred_ob_k::bmssp<distT>>(
    adj, cpred["cpred_ob_k"], dijkstra_ms, all_cpred_cases
  );

  run_cpred_family<spp_timed_cpred_dedup_k::bmssp<distT>>(
    adj, cpred["cpred_dedup_k"], dijkstra_ms, all_cpred_cases
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
    if (++dir_count >= 10) break;

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


struct BMSSPFConfig {
  std::string pscase_type;
  std::string pscase_mode;
  std::string frontier;
  std::string countmin_search;
  std::string countmin_mode;
  int BF_steps = 0;

  std::string key() const {
    return std::string("pscase_type=") + pscase_type +
           "__pscase_mode=" + pscase_mode +
           "__frontier=" + frontier +
           "__countmin_search=" + countmin_search +
           "__countmin_mode=" + countmin_mode +
           "__BF_steps=" + std::to_string(BF_steps);
  }

  json to_json() const {
    return json{{
      {"pscase_type", pscase_type},
      {"pscase_mode", pscase_mode},
      {"frontier", frontier},
      {"countmin_search", countmin_search},
      {"countmin_mode", countmin_mode},
      {"BF_steps", BF_steps}
    }};
  }
};

static void push_bmsspf_config_unique(
    std::vector<BMSSPFConfig>& out,
    const BMSSPFConfig& cfg) {
  const std::string k = cfg.key();
  const bool exists = std::any_of(
    out.begin(),
    out.end(),
    [&](const BMSSPFConfig& x) { return x.key() == k; }
  );
  if (!exists) out.push_back(cfg);
}

static std::vector<std::string> allowed_bmsspf_pscase_types(
    const std::string& input_graph_family) {
  std::vector<std::string> out;
  auto add_unique = [&](const std::string& s) {
    if (s.empty()) return;
    if (std::find(out.begin(), out.end(), s) == out.end()) out.push_back(s);
  };

  add_unique(input_graph_family);
  add_unique("mix_gen");
  add_unique("mix_all");
  return out;
}

static std::vector<BMSSPFConfig> build_bmsspf_grid(
    const std::string& input_graph_family) {
  const std::vector<std::string> graph_types =
      allowed_bmsspf_pscase_types(input_graph_family);
  const std::array<std::string, 4> pscase_modes{{
      "false", "offline", "online", "blank"
  }};
  const std::array<std::string, 2> frontier_modes{{
      "bpq", "lapq"
  }};
  const std::array<std::string, 5> countmin_search_modes{{
      "false", "dedup", "npf", "hybrid", "ob"
  }};
  const std::array<std::string, 3> countmin_modes{{
      "offline", "online", "blank"
  }};
  const std::array<int, 4> bf_steps_values{{0, 1, 2, 3}};

  std::vector<BMSSPFConfig> out;

  for (const auto& frontier : frontier_modes) {
    for (const auto& pscase_mode : pscase_modes) {
      for (const auto& search_mode : countmin_search_modes) {
        const bool pscase_enabled = (pscase_mode != "false");
        const bool countmin_enabled = (search_mode != "false");

        std::vector<std::string> active_graph_types;
        if (!pscase_enabled && !countmin_enabled) {
          active_graph_types.push_back(input_graph_family);
        } else {
          active_graph_types = graph_types;
        }

        if (!countmin_enabled) {
          for (const auto& ptype : active_graph_types) {
            push_bmsspf_config_unique(out, BMSSPFConfig{
              ptype,
              pscase_mode,
              frontier,
              "false",
              "false",
              0
            });
          }
          continue;
        }

        for (const auto& ptype : active_graph_types) {
          for (const auto& cm_mode : countmin_modes) {
            for (int bf_steps : bf_steps_values) {
              push_bmsspf_config_unique(out, BMSSPFConfig{
                ptype,
                pscase_mode,
                frontier,
                search_mode,
                cm_mode,
                bf_steps
              });
            }
          }
        }
      }
    }
  }

  return out;
}

template <typename Graph>
Stats run_bmsspf_alg(
    const Graph& adj,
    double dijkstra_ms,
    const BMSSPFConfig& cfg) {
  timerT timer;

  spp_bmsspf::bmssp<distT> alg(
    adj,
    cfg.pscase_type,
    cfg.pscase_mode,
    cfg.frontier,
    cfg.countmin_search,
    cfg.countmin_mode,
    cfg.BF_steps
  );
  alg.stats.time_dijkstra = dijkstra_ms;
  alg.prepare_graph(false);

  timer.start();
  auto [d, p] = alg.execute(0);
  timer.stop();

  alg.stats.update_time_full(timer.elapsed_ms());
  return alg.stats;
}

// [[Rcpp::export]]
std::string runBMSSPFSearch() {
  timerT timer;
  timerT total;
  total.start();

  std::ifstream f("run_class.json");
  json cfg;
  f >> cfg;

  const std::string input_graph_family = cfg["graph"].get<std::string>();
  const std::string root =
      "C:/Users/Jakub/Documents/stuff/diss/project/graphs/" +
      input_graph_family;

  const auto graphs = list_graphs_in_order(root);
  const auto bmsspf_grid = build_bmsspf_grid(input_graph_family);

  std::size_t dir_count = 0;
  json jout = json::object();
  jout["meta"] = {
    {"input_graph_family", input_graph_family},
    {"n_bmsspf_configs", static_cast<int>(bmsspf_grid.size())}
  };

  for (const auto& [dir, gs] : graphs) {
    if (++dir_count >= 1) break;

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
      graph_bucket["bmsspf"] = json::object();

      for (const auto& bmsspf_cfg : bmsspf_grid) {
        json entry;
        entry["config"] = bmsspf_cfg.to_json();
        try {
          entry["stats"] = run_bmsspf_alg(adj, dijkstra_ms, bmsspf_cfg);
        } catch (const std::exception& e) {
          entry["error"] = e.what();
        } catch (...) {
          entry["error"] = "unknown error";
        }
        graph_bucket["bmsspf"][bmsspf_cfg.key()] = std::move(entry);
      }
    }
  }

  std::ofstream out("experiments/run_stats.json");
  out << jout.dump(2);

  total.stop();
  std::cout << "Total BMSSPF search time (" << input_graph_family << "): "
            << total.elapsed_ms() << "ms\n";

  return "done";
}
