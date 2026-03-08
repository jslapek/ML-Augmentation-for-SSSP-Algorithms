// [[Rcpp::plugins(cpp20)]]
#include <Rcpp.h>
using namespace Rcpp;

#include "utils.hpp"
#include "common.hpp"
#include "algs/bmssp.hpp"
#include "algs/bmssp_timed.hpp"
#include "algs/bmssp_ml_theory.hpp"

#include <fstream>
#include <iostream>
#include <json.hpp>
#include <filesystem>
using json = nlohmann::json;
namespace fs = std::filesystem;

using distT = long long;
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(Stats, time_full, time_find_pivot, time_base_case, time_D_op, time_bmssp, time_batch_prepend, snip_split, snip_lower_bound, snip_block_insertion, snip_membership_check, snip_deletion)

static bool is_number(const std::string& s) {
    if (s.empty()) return false;
    for (char c : s) if (c < '0' || c > '9') return false;
    return true;
}

std::map<long long, std::vector<std::string>> list_graphs_in_order(const std::string& root) {
    fs::path root_path(root);
    // std::vector<std::string> out;
    std::map<long long, std::vector<std::string>> size_to_graphs;

    // 1) collect numeric subdirectories (8,16,32,...)
    std::vector<std::pair<long long, fs::path>> size_dirs;
    for (const auto& entry : fs::directory_iterator(root_path)) {
        if (!entry.is_directory()) continue;

        std::string name = entry.path().filename().string();
        if (!is_number(name)) continue;

        size_dirs.push_back({std::stoll(name), entry.path()});
    }

    // 2) sort by numeric folder name
    std::sort(size_dirs.begin(), size_dirs.end(),
              [](const auto& a, const auto& b) { return a.first < b.first; });

    // 3) within each folder, collect .gr files and sort
    for (const auto& [n, dir] : size_dirs) {
        std::vector<fs::path> graphs;

        for (const auto& f : fs::directory_iterator(dir)) {
            if (f.is_regular_file() && f.path().extension() == ".gr") {
                size_to_graphs[n].push_back(f.path().string());
                graphs.push_back(f.path());
            }
        }
    }

    return size_to_graphs;
}

// [[Rcpp::export]]
std::string runSearch() {
  std::ifstream f("run_class.json");
  json cfg;
  f >> cfg;

  std::string root = "C:/Users/Jakub/Documents/stuff/diss/project/graphs/" + cfg["graph"].get<std::string>();
  auto graphs = list_graphs_in_order(root);

  size_t i = 0;
  json jout = json::object();
  for (auto [dir, gs] : graphs) {
    if (++i >= 10) break;
    json& dir_bucket = jout[std::to_string(dir)];

    std::cout << "Processing directory: " << dir << " with " << gs.size() << " graphs.\n";
    for (std::size_t i = 0; i < gs.size(); ++i) {
      std::cout << "Processing graph: " << fs::path(gs[i]).filename().string() << "\n";
      auto [adj, m] = readGraph<distT>(gs[i]);
      spp_timed::bmssp<distT> bmssp(adj);
      bmssp.prepare_graph(false);
      int source = 0;

      timerT timer;
      auto [d, p] = bmssp.execute(source);
      timer.stop();
      bmssp.stats.update_time_full(timer.elapsed_ms());

      dir_bucket[fs::path(gs[i]).filename().string()] = bmssp.stats;
      std::cout << bmssp.stats.time_full << "ms\n";
    }
  }
  std::ofstream out("experiments/run_stats.json");
  out << jout.dump(2);
  return "done";
}