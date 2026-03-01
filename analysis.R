# Download packages
if (!requireNamespace("reticulate", quietly = TRUE)) install.packages("reticulate")
if (!requireNamespace("ggplot2", quietly = TRUE))    install.packages("ggplot2")
if (!requireNamespace("dplyr", quietly = TRUE))      install.packages("dplyr")
if (!requireNamespace("Rcpp", quietly = TRUE))       install.packages("Rcpp")

# System variables
Sys.setenv(RETICULATE_PYTHON = "C:/Users/Jakub/anaconda3/python.exe")

# Packages
library(reticulate)
library(ggplot2)
library(dplyr)
library(Rcpp)
py_config()

# import python experimentation package
run_module <- import_from_path("run", "py_pkg")
Rcpp::sourceCpp("cpp_pkg/run.cpp")

default_cfg <- list(
  alg = "bmssp", 
  heap = "binary", 
  frontier = "block",
  graph = "random", 
  seed = 42L, 
  n = 1000L, 
  m = 4000L,
  transform = FALSE, 
  transform_delta = 4L,
  niters = 10L, 
  nsources = 1L
)

write_cfg <- function (path = "run_class.json", cfg = default_cfg) {
  jsonlite::write_json(cfg, path, auto_unbox = TRUE, pretty = TRUE, digits = NA)
}


get_ns <- function(from, to, by, type="seq") {
  if (type == "small") {
    return (seq(100, 1000, 100))
  } else if (type == "medium") {
    return (seq(1000, 10000, 1000))
  } else if (type == "large") {
    return (seq(10000, 100000, 10000))
  } 
  seq(from = from, to = to, by = by)
}

run_experiments <- function(cfg, ns) {
  experiments = list()
  for (i in ns) {
      cfg$n <- as.integer(i)
      cfg$m <- as.integer(i) * 4L
      experiments[[length(experiments) + 1]] <- run_module$runSearch(cfg)
      cat(sprintf("Completed experiment %d\n", i))
  }
  return (experiments)
}

get_mean_metric <- function(experiments, metric) {
  means <- sapply(experiments, function(x) {
      mean(sapply(x, function(iter) reticulate::py_to_r(reticulate::py_get_attr(iter[[2]], metric)))) 
  })
  return (means)
}

get_std_metric <- function(experiments, metric) {
  sds <- sapply(experiments, function(x) {
      sd(sapply(x, function(iter) reticulate::py_to_r(reticulate::py_get_attr(iter[[2]], metric))))
  })
  return (sds)
}

pipeline <- function(cfg=get_cfg(), ns=get_ns(type="medium")) {
  experiments <- run_experiments(cfg, ns) 
  mean_metric <- get_mean_metric(experiments, "edges_relaxed")
  std_metric <- get_std_metric(experiments, "edges_relaxed")
  return (list(mean=mean_metric, sd=std_metric))

}

# print(pipeline(ns=get_ns(type="small")))

write_cfg()
print(runSearch())