#pragma once
#include <chrono>

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