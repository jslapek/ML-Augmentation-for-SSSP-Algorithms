// [[Rcpp::plugins(cpp20)]]
#include <Rcpp.h>
using namespace Rcpp;

#include "common.hpp"
#include "algs/bmssp.hpp"

#include <fstream>
#include <iostream>
#include <json.hpp>
using json = nlohmann::json;

using distT = long long;

// [[Rcpp::export]]
std::string runSearch() {
  std::ifstream f("run_class.json");
  json j;
  f >> j;

  std::string g_path = "C:/Users/Jakub/Documents/stuff/diss/project/graphs/random/random32D3.gr";
  auto [adj, m] = readGraph<distT>(g_path);

  spp_expected::bmssp<distT> bmssp(adj);
  bmssp.prepare_graph(false);
  int source = 0;
  auto [d, p] = bmssp.execute(source);

  json j;
  ///
  std::ofstream out("experiments/run_stats.json");
  out << j.dump(2);
  return "done";

}