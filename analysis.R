# Download packages
if (!requireNamespace("reticulate", quietly = TRUE)) install.packages("reticulate")
if (!requireNamespace("scales", quietly = TRUE)) install.packages("scales")
if (!requireNamespace("ggplot2", quietly = TRUE))    install.packages("ggplot2")
if (!requireNamespace("dplyr", quietly = TRUE))      install.packages("dplyr")
if (!requireNamespace("tidyr", quietly = TRUE))      install.packages("tidyr")
if (!requireNamespace("Rcpp", quietly = TRUE))       install.packages("Rcpp")

# System variables
Sys.setenv(RETICULATE_PYTHON = "C:/Users/Jakub/anaconda3/python.exe")

# Packages
library(reticulate)
library(jsonlite)
library(scales)
library(ggplot2)
library(dplyr)
library(tidyr)
library(Rcpp)
py_config()

# import python experimentation package
# run_module <- import_from_path("run", "py_pkg")
Rcpp::sourceCpp("cpp_pkg/run.cpp")

default_cfg <- list(
  alg = "bmssp", 
  heap = "binary", 
  frontier = "block",
  graph = "randomD", 
  seed = 42L, 
  n = 1000L, 
  m = 4000L,
  transform = FALSE, 
  transform_delta = 4L,
  niters = 10L, 
  nsources = 1L
)

###### File I/O #######

write_path <- "run_class.json"
write_cfg <- function (path = write_path, cfg = default_cfg) {
  jsonlite::write_json(cfg, path, auto_unbox = TRUE, pretty = TRUE, digits = NA)
}

read_path <- "experiments/run_stats.json"
read_json <- function (path = read_path) {
  jsonlite::fromJSON(path, simplifyVector = TRUE)
}

######### Analysis ########

dir_stats <- function(x) { lapply(x, function(dir_block) {
  mat <- do.call(rbind, lapply(dir_block, function(stats) unlist(stats)))
  colnames(mat) <- names(dir_block[[1]])

  list(
    mean = colMeans(mat, na.rm = TRUE),
    sd   = apply(mat, 2, sd, na.rm = TRUE)
  )
})}

########### Plotting ##########

plot_mean_times <- function() {
  x <- read_json()
  stats <- dir_stats(x)
  
  means_df <- data.frame(
  directory = names(stats),
  do.call(rbind, lapply(stats, `[[`, "mean")),
  row.names = NULL,
  check.names = FALSE
) %>%
  mutate(directory = as.numeric(directory)) %>%
  arrange(directory)

  means_df <- means_df %>%
  mutate(across(
    c(time_find_pivot, time_base_case, time_D_op, time_batch_prepend, time_bmssp),
    ~ . / time_full
  ))

  parts_long <- means_df %>%
  select(directory, time_find_pivot, time_base_case, time_D_op, time_batch_prepend, time_bmssp) %>%
  pivot_longer(-directory, names_to = "part", values_to = "frac") %>%
  mutate(part = factor(
    part,
    levels = c("time_find_pivot", "time_base_case", "time_D_op", "time_batch_prepend", "time_bmssp")
  ))

  parts_long <- parts_long %>%
  mutate(part = recode(part,
    time_find_pivot    = "Find Pivots",
    time_base_case     = "Base Case",
    time_D_op          = "BatchPQ Insert/Erase",
    time_batch_prepend = "Batch Prepend",
    time_bmssp         = "BMSSP Recursion"
  ))

  total_df <- means_df %>% select(directory, time_full)

  # evenly spaced x positions
  x_levels <- sort(unique(parts_long$directory))
  parts_long <- parts_long %>%
    mutate(xpos = match(directory, x_levels))

  # labels as 2^k (if your x_levels are powers of 2)
  x_labs <- scales::label_math(expr = 2^.x)(log2(x_levels))
  # if you want integer exponents only:
  # x_labs <- scales::label_math(expr = 2^.x)(round(log2(x_levels)))

  plt <- ggplot(parts_long, aes(x = xpos, y = frac, fill = part)) +
    geom_area(position = "stack", alpha = 0.85) +
    scale_x_continuous(
      breaks = seq_along(x_levels),
      labels = x_labs
    ) +
    labs(x = "Graph Size (n)", y = "Mean Running Time (%)", fill = "Component") +
    scale_fill_manual(
      values = c(
        "Find Pivots" = "#6358a5",
        "Base Case" = "#0084b3",
        "BatchPQ Insert/Erase" = "#00b1b5",
        "Batch Prepend" = "grey",
        "BMSSP Recursion" = "#ffb300"
      ),
      name = "Component"
    ) +
    theme_minimal()
    

  ggsave(filename = "figures/plot.pdf", plot = plt, width = 8, height = 6)
}

plot_d_insert_times <- function() {
  x <- read_json()
  stats <- dir_stats(x)
  
  means_df <- data.frame(
  directory = names(stats),
  do.call(rbind, lapply(stats, `[[`, "mean")),
  row.names = NULL,
  check.names = FALSE
) %>%
  mutate(directory = as.numeric(directory)) %>%
  arrange(directory) %>%
  mutate(across(
    c(time_D_op, snip_split, snip_lower_bound, snip_block_insertion, snip_membership_check, snip_deletion),
    ~ . / time_D_op
  )) %>%
  mutate(
    time_D_op = time_D_op - snip_split - snip_lower_bound - snip_block_insertion - snip_membership_check - snip_deletion
  )
  

  parts_long <- means_df %>%
  select(directory, time_D_op, snip_split, snip_lower_bound, snip_block_insertion, snip_membership_check, snip_deletion) %>%
  pivot_longer(-directory, names_to = "part", values_to = "frac") %>%
  mutate(part = factor(
    part,
    levels = c("time_D_op", "snip_split", "snip_lower_bound", "snip_block_insertion", "snip_membership_check", "snip_deletion")
  ))

  parts_long <- parts_long %>%
  mutate(part = recode(part,
    time_D_op    = "Memory Access/Allocation",
    snip_split     = "Split",
    snip_lower_bound          = "Lower Bound Search",
    snip_block_insertion = "Intra-Block Insertion",
    snip_membership_check = "Membership Check",
    snip_deletion = "Deletion Overhead"
  ))

  total_df <- means_df %>% select(directory, time_D_op)

  # evenly spaced x positions
  x_levels <- sort(unique(parts_long$directory))
  parts_long <- parts_long %>%
    mutate(xpos = match(directory, x_levels))

  # labels as 2^k (if your x_levels are powers of 2)
  x_labs <- scales::label_math(expr = 2^.x)(log2(x_levels))
  # if you want integer exponents only:
  # x_labs <- scales::label_math(expr = 2^.x)(round(log2(x_levels)))

  plt <- ggplot(parts_long, aes(x = xpos, y = frac, fill = part)) +
    geom_area(position = "stack", alpha = 0.85) +
    scale_x_continuous(
      breaks = seq_along(x_levels),
      labels = x_labs
    ) +
    labs(x = "Graph Size (n)", y = "Mean Running Time (%)", fill = "Component") +
    scale_fill_manual(
      values = c(
        "Memory Access/Allocation" = "#0084b3",
        "Split" = "#00b1b5",
        "Lower Bound Search" = "grey",
        "Intra-Block Insertion" = "#ffb300",
        "Membership Check" = "#6358a5",
        "Deletion Overhead" = "#ff6f00"
      ),
      name = "Component"
    ) +
    theme_minimal()
    

  ggsave(filename = "figures/plot.pdf", plot = plt, width = 8, height = 6)
}

######### Main ##########

# write_cfg()
runSearch()
# plot_mean_times()
plot_d_insert_times()


######### OLD PYTHON ##########

# get_ns <- function(from, to, by, type="seq") {
#   if (type == "small") {
#     return (seq(100, 1000, 100))
#   } else if (type == "medium") {
#     return (seq(1000, 10000, 1000))
#   } else if (type == "large") {
#     return (seq(10000, 100000, 10000))
#   } 
#   seq(from = from, to = to, by = by)
# }

# run_experiments <- function(cfg, ns) {
#   experiments = list()
#   for (i in ns) {
#       cfg$n <- as.integer(i)
#       cfg$m <- as.integer(i) * 4L
#       experiments[[length(experiments) + 1]] <- run_module$runSearch(cfg)
#       cat(sprintf("Completed experiment %d\n", i))
#   }
#   return (experiments)
# }

# get_mean_metric <- function(experiments, metric) {
#   means <- sapply(experiments, function(x) {
#       mean(sapply(x, function(iter) reticulate::py_to_r(reticulate::py_get_attr(iter[[2]], metric)))) 
#   })
#   return (means)
# }

# get_std_metric <- function(experiments, metric) {
#   sds <- sapply(experiments, function(x) {
#       sd(sapply(x, function(iter) reticulate::py_to_r(reticulate::py_get_attr(iter[[2]], metric))))
#   })
#   return (sds)
# }

# pipeline <- function(cfg=get_cfg(), ns=get_ns(type="medium")) {
#   experiments <- run_experiments(cfg, ns) 
#   mean_metric <- get_mean_metric(experiments, "edges_relaxed")
#   std_metric <- get_std_metric(experiments, "edges_relaxed")
#   return (list(mean=mean_metric, sd=std_metric))

# }

# print(pipeline(ns=get_ns(type="small")))

######### OLD PYTHON ##########
