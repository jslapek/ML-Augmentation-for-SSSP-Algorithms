#pragma once
#include <chrono>
#include <random>

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

struct Stats {
  double time_full = 0;
  double time_find_pivot = 0;
  double time_base_case = 0;
  double time_D_op = 0;
  double time_bmssp = 0;
  double time_batch_prepend = 0;

  // FindPivots
  double snip_tree_construction = 0.0;
  double snip_relaxation = 0.0;

  // BatchPQ insertion snippets
  double snip_split = 0;
  double snip_lower_bound = 0;
  double snip_block_insertion = 0;
  double snip_membership_check = 0;
  double snip_deletion = 0;

  void update_time_full(double new_time_full) {
    time_full += new_time_full;
    time_bmssp = time_full - time_find_pivot - time_base_case - time_D_op - time_batch_prepend;
  }
};

int randomBit(double p) {
  std::random_device rd;
  std::mt19937 gen(rd());          
  std::bernoulli_distribution dist(p);  
  return dist(gen);
}