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
Rcpp::sourceCpp("cpp_pkg/run_stats.cpp")

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

plot_mean_times <- function(filename = "figures/time_components_bmssp.pdf") {
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
    

  ggsave(filename = filename, plot = plt, width = 8, height = 6)
}

plot_d_insert_times <- function(filename = "figures/time_components_insert.pdf") {
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
    

  ggsave(filename = filename, plot = plt, width = 8, height = 6)
}

plot_pivot <- function(filename="figures/time_components_pivot.pdf") {
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
    c(time_find_pivot, snip_tree_construction, snip_relaxation),
    ~ . / time_find_pivot
  )) %>%
  mutate(
    time_find_pivot = time_find_pivot - snip_tree_construction - snip_relaxation
  )
  

  parts_long <- means_df %>%
  select(directory, time_find_pivot, snip_tree_construction, snip_relaxation) %>%

  pivot_longer(-directory, names_to = "part", values_to = "frac") %>%
  mutate(part = factor(
    part,
    levels = c("time_find_pivot", "snip_tree_construction", "snip_relaxation")
  ))

  parts_long <- parts_long %>%
  mutate(part = recode(part,
    time_find_pivot    = "Bellman-Ford Exploration",
    snip_tree_construction = "Tree Construction",
    snip_relaxation = "Relaxation Overhead"
  ))

  total_df <- means_df %>% select(directory, time_find_pivot)

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
        "Bellman-Ford Exploration" = "#0084b3",
        "Relaxation Overhead" = "#ffb300",
        "Tree Construction" = "grey"
      ),
      name = "Component"
    ) +
    theme_minimal()
    

  ggsave(filename = filename, plot = plt, width = 8, height = 6)
}

######### Experiment Setups ##########


bmsspf_largest_table <- function(
  json_path = "experiments/run_stats_bmsspf.json",
  out_path = "tables/bmsspf_largest_table.tsv",
  graph_file = NULL,
  digits = 4,
  sort_by_mean = TRUE
) {
  if (!requireNamespace("jsonlite", quietly = TRUE)) install.packages("jsonlite")
  if (!requireNamespace("dplyr", quietly = TRUE)) install.packages("dplyr")

  library(jsonlite)
  library(dplyr)

  x <- jsonlite::fromJSON(json_path, simplifyVector = FALSE)

  if (length(x) == 0) {
    stop("JSON file is empty.")
  }

  get_numeric_bucket <- function(obj) {
    keys <- names(obj)
    vals <- suppressWarnings(as.numeric(keys))

    if (!is.null(keys) && any(!is.na(vals))) {
      return(obj)
    }

    if (length(obj) == 1 && is.list(obj[[1]])) {
      inner <- obj[[1]]
      inner_keys <- names(inner)
      inner_vals <- suppressWarnings(as.numeric(inner_keys))
      if (!is.null(inner_keys) && any(!is.na(inner_vals))) {
        return(inner)
      }
    }

    stop(
      paste0(
        "Top-level JSON keys are not numeric graph sizes. Found: ",
        paste(names(obj), collapse = ", ")
      )
    )
  }

  x <- get_numeric_bucket(x)

  size_keys <- names(x)
  size_vals <- suppressWarnings(as.numeric(size_keys))

  largest_idx <- which.max(size_vals)
  largest_size_key <- size_keys[largest_idx]
  largest_bucket <- x[[largest_size_key]]

  if (length(largest_bucket) == 0) {
    stop("Largest graph-size bucket is empty.")
  }

  graph_names <- names(largest_bucket)

  if (!is.null(graph_file)) {
    if (!(graph_file %in% graph_names)) {
      stop(sprintf(
        "Graph file '%s' not found in largest bucket. Available: %s",
        graph_file,
        paste(graph_names, collapse = ", ")
      ))
    }
    graph_names <- graph_file
  }

  rows <- list()

  for (g in graph_names) {
    graph_block <- largest_bucket[[g]]

    if (is.null(graph_block$bmsspf)) next

    runs <- graph_block$bmsspf
    run_keys <- names(runs)

    for (k in run_keys) {
      entry <- runs[[k]]

      cfg <- if (!is.null(entry$config) && length(entry$config) >= 1) {
        entry$config[[1]]
      } else {
        list()
      }

      stats <- if (!is.null(entry$stats)) entry$stats else list()

      rows[[length(rows) + 1]] <- data.frame(
        graph_size = as.numeric(largest_size_key),
        graph_file = g,
        config_key = k,
        pscase_type = if (!is.null(cfg$pscase_type)) cfg$pscase_type else NA_character_,
        pscase_mode = if (!is.null(cfg$pscase_mode)) cfg$pscase_mode else NA_character_,
        frontier = if (!is.null(cfg$frontier)) cfg$frontier else NA_character_,
        countmin_search = if (!is.null(cfg$countmin_search)) cfg$countmin_search else NA_character_,
        countmin_mode = if (!is.null(cfg$countmin_mode)) cfg$countmin_mode else NA_character_,
        BF_steps = if (!is.null(cfg$BF_steps)) as.integer(cfg$BF_steps) else NA_integer_,
        time_full = if (!is.null(stats$time_full)) as.numeric(stats$time_full) else NA_real_,
        stringsAsFactors = FALSE
      )
    }
  }

  if (length(rows) == 0) {
    stop("No BMSSPF rows found in the selected largest bucket.")
  }

  tab_raw <- bind_rows(rows)

  tab <- tab_raw %>%
    group_by(
      graph_size,
      pscase_type,
      pscase_mode,
      frontier,
      countmin_search,
      countmin_mode,
      BF_steps
    ) %>%
    summarise(
      n_graphs = sum(!is.na(time_full)),
      mean_time_full = mean(time_full, na.rm = TRUE),
      sd_time_full = ifelse(
        sum(!is.na(time_full)) > 1,
        sd(time_full, na.rm = TRUE),
        0
      ),
      runtime = paste0(
        format(round(mean_time_full, digits), nsmall = digits),
        " \u00B1 ",
        format(round(sd_time_full, digits), nsmall = digits)
      ),
      .groups = "drop"
    )

  if (sort_by_mean) {
    tab <- tab %>% arrange(mean_time_full)
  }

  tab_out <- tab %>%
    select(
      # graph_size,
      pscase_type,
      pscase_mode,
      frontier,
      countmin_search,
      countmin_mode,
      BF_steps,
      # n_graphs,
      runtime
    )

  write.table(
    tab_out,
    file = out_path,
    sep = "\t",
    row.names = FALSE,
    quote = FALSE
  )

  cat(sprintf(
    "Wrote %d aggregated rows for largest size bucket %s to %s\n",
    nrow(tab_out), largest_size_key, out_path
  ))

  return(tab_out)
}

# ---------- BMSSPF helpers ----------

read_bmsspf_json_tidy <- function(json_path) {
  if (!requireNamespace("jsonlite", quietly = TRUE)) install.packages("jsonlite")
  if (!requireNamespace("dplyr", quietly = TRUE)) install.packages("dplyr")

  library(jsonlite)
  library(dplyr)

  x <- jsonlite::fromJSON(json_path, simplifyVector = FALSE)

  if (length(x) == 0) stop("JSON file is empty.")

  get_numeric_bucket <- function(obj) {
    if (!is.list(obj)) stop("Top-level JSON is not a list.")

    keys <- names(obj)
    vals <- suppressWarnings(as.numeric(keys))

    if (!is.null(keys) && any(!is.na(vals))) return(obj)

    if (length(obj) == 1 && is.list(obj[[1]])) {
      inner <- obj[[1]]
      inner_keys <- names(inner)
      inner_vals <- suppressWarnings(as.numeric(inner_keys))
      if (!is.null(inner_keys) && any(!is.na(inner_vals))) return(inner)
    }

    stop(
      paste0(
        "Top-level JSON keys are not numeric graph sizes. Found: ",
        paste(names(obj), collapse = ", ")
      )
    )
  }

  x <- get_numeric_bucket(x)
  rows <- list()

  for (size_key in names(x)) {
    size_block <- x[[size_key]]
    if (!is.list(size_block)) next

    for (graph_file in names(size_block)) {
      graph_block <- size_block[[graph_file]]
      if (!is.list(graph_block)) next

      bmsspf_block <- graph_block[["bmsspf"]]
      if (is.null(bmsspf_block) || !is.list(bmsspf_block)) next

      for (run_key in names(bmsspf_block)) {
        entry <- bmsspf_block[[run_key]]
        if (!is.list(entry)) next

        cfg <- entry[["config"]]
        if (is.list(cfg) && length(cfg) >= 1) {
          cfg <- cfg[[1]]
        } else {
          cfg <- list()
        }

        stats <- entry[["stats"]]
        if (is.null(stats) || !is.list(stats)) {
          stats <- list()
        }

        rows[[length(rows) + 1]] <- data.frame(
          graph_size = as.numeric(size_key),
          graph_file = graph_file,
          config_key = run_key,
          pscase_type = if (!is.null(cfg[["pscase_type"]])) cfg[["pscase_type"]] else NA_character_,
          pscase_mode = if (!is.null(cfg[["pscase_mode"]])) cfg[["pscase_mode"]] else NA_character_,
          frontier = if (!is.null(cfg[["frontier"]])) cfg[["frontier"]] else NA_character_,
          countmin_search = if (!is.null(cfg[["countmin_search"]])) cfg[["countmin_search"]] else NA_character_,
          countmin_mode = if (!is.null(cfg[["countmin_mode"]])) cfg[["countmin_mode"]] else NA_character_,
          BF_steps = if (!is.null(cfg[["BF_steps"]])) as.integer(cfg[["BF_steps"]]) else NA_integer_,
          time_full = if (!is.null(stats[["time_full"]])) as.numeric(stats[["time_full"]]) else NA_real_,
          time_bmssp = if (!is.null(stats[["time_bmssp"]])) as.numeric(stats[["time_bmssp"]]) else NA_real_,
          time_find_pivot = if (!is.null(stats[["time_find_pivot"]])) as.numeric(stats[["time_find_pivot"]]) else NA_real_,
          time_base_case = if (!is.null(stats[["time_base_case"]])) as.numeric(stats[["time_base_case"]]) else NA_real_,
          time_D_op = if (!is.null(stats[["time_D_op"]])) as.numeric(stats[["time_D_op"]]) else NA_real_,
          time_batch_prepend = if (!is.null(stats[["time_batch_prepend"]])) as.numeric(stats[["time_batch_prepend"]]) else NA_real_,
          snip_split = if (!is.null(stats[["snip_split"]])) as.numeric(stats[["snip_split"]]) else NA_real_,
          snip_lower_bound = if (!is.null(stats[["snip_lower_bound"]])) as.numeric(stats[["snip_lower_bound"]]) else NA_real_,
          snip_block_insertion = if (!is.null(stats[["snip_block_insertion"]])) as.numeric(stats[["snip_block_insertion"]]) else NA_real_,
          snip_membership_check = if (!is.null(stats[["snip_membership_check"]])) as.numeric(stats[["snip_membership_check"]]) else NA_real_,
          snip_deletion = if (!is.null(stats[["snip_deletion"]])) as.numeric(stats[["snip_deletion"]]) else NA_real_,
          snip_tree_construction = if (!is.null(stats[["snip_tree_construction"]])) as.numeric(stats[["snip_tree_construction"]]) else NA_real_,
          snip_relaxation = if (!is.null(stats[["snip_relaxation"]])) as.numeric(stats[["snip_relaxation"]]) else NA_real_,
          stringsAsFactors = FALSE
        )
      }
    }
  }

  if (length(rows) == 0) stop("No BMSSPF rows found.")
  dplyr::bind_rows(rows)
}

bmsspf_largest_only <- function(df, graph_file = NULL) {
  library(dplyr)

  largest_size <- max(df$graph_size, na.rm = TRUE)
  out <- df %>% filter(graph_size == largest_size)

  if (!is.null(graph_file)) {
    out <- out %>% filter(graph_file == graph_file)
  }

  out
}

bmsspf_add_baseline_speedup <- function(df) {
  library(dplyr)

  baseline <- df %>%
    filter(
      frontier == "bpq",
      pscase_mode == "false",
      countmin_search == "false",
      countmin_mode == "false",
      BF_steps == 0
    ) %>%
    group_by(graph_size, graph_file) %>%
    summarise(baseline_time = mean(time_full, na.rm = TRUE), .groups = "drop")

  df %>%
    left_join(baseline, by = c("graph_size", "graph_file")) %>%
    mutate(speedup_vs_baseline = baseline_time / time_full)
}

bmsspf_ablation_label <- function(pscase_mode, countmin_search) {
  p_on <- !is.na(pscase_mode) && pscase_mode != "false"
  c_on <- !is.na(countmin_search) && countmin_search != "false"

  if (!p_on && !c_on) return("none")
  if (p_on && !c_on)  return("p=s only")
  if (!p_on && c_on)  return("countmin only")
  return("both")
}

# ---------- 1) Largest-instance heatmap ----------

plot_bmsspf_largest_heatmap <- function(
  json_path,
  filename = "figures/bmsspf_largest_heatmap.pdf",
  graph_file = NULL
) {
  library(dplyr)
  library(ggplot2)

  df <- read_bmsspf_json_tidy(json_path) %>%
    bmsspf_largest_only(graph_file = graph_file) %>%
    group_by(
      pscase_type, pscase_mode, frontier,
      countmin_search, countmin_mode, BF_steps
    ) %>%
    summarise(
      mean_time = mean(time_full, na.rm = TRUE),
      sd_time = ifelse(sum(!is.na(time_full)) > 1, sd(time_full, na.rm = TRUE), 0),
      .groups = "drop"
    ) %>%
    mutate(
      x_lab = paste(frontier, pscase_mode, pscase_type, sep = "\n"),
      y_lab = paste(countmin_search, countmin_mode, paste0("BF=", BF_steps), sep = "\n")
    )

  plt <- ggplot(df, aes(x = x_lab, y = y_lab, fill = mean_time)) +
    geom_tile() +
    geom_text(aes(label = sprintf("%.3f", mean_time)), size = 2.6) +
    labs(
      x = "Frontier / p=s mode / p=s type",
      y = "Count-Min search / Count-Min mode / BF steps",
      fill = "Mean runtime (ms)"
    ) +
    theme_minimal() +
    theme(
      axis.text.x = element_text(angle = 45, hjust = 1),
      panel.grid = element_blank()
    )

  ggsave(filename, plt, width = 14, height = 10)
  invisible(df)
}

# ---------- 2) Best speedup vs baseline across graph sizes ----------

plot_bmsspf_best_speedup <- function(
  json_paths,
  labels = NULL,
  filename = "figures/bmsspf_best_speedup.pdf"
) {
  library(dplyr)
  library(ggplot2)

  if (is.null(labels)) labels <- basename(tools::file_path_sans_ext(json_paths))
  if (length(labels) != length(json_paths)) stop("labels must match json_paths length")

  all_rows <- list()

  for (i in seq_along(json_paths)) {
    df <- read_bmsspf_json_tidy(json_paths[i]) %>%
      bmsspf_add_baseline_speedup()

    best_by_graph <- df %>%
      group_by(graph_size, graph_file) %>%
      slice_min(order_by = time_full, n = 1, with_ties = FALSE) %>%
      ungroup() %>%
      mutate(family = labels[i])

    all_rows[[i]] <- best_by_graph
  }

  plot_df <- bind_rows(all_rows) %>%
    group_by(family, graph_size) %>%
    summarise(
      mean_speedup = mean(speedup_vs_baseline, na.rm = TRUE),
      sd_speedup = ifelse(sum(!is.na(speedup_vs_baseline)) > 1, sd(speedup_vs_baseline, na.rm = TRUE), 0),
      .groups = "drop"
    )

  plt <- ggplot(plot_df, aes(x = graph_size, y = mean_speedup, group = family)) +
    geom_line() +
    geom_point() +
    geom_errorbar(aes(ymin = mean_speedup - sd_speedup, ymax = mean_speedup + sd_speedup), width = 0) +
    scale_x_log10(
      breaks = sort(unique(plot_df$graph_size)),
      labels = scales::label_math(expr = 2^.x)(round(log2(sort(unique(plot_df$graph_size)))))
    ) +
    labs(
      x = "Graph size (n)",
      y = "Best speedup over BPQ no-prediction baseline",
      color = "Family"
    ) +
    theme_minimal()

  ggsave(filename, plt, width = 8.5, height = 6)
  invisible(plot_df)
}

# ---------- 3) Ablation plot ----------

plot_bmsspf_ablation <- function(
  json_path,
  filename = "figures/bmsspf_ablation.pdf",
  graph_file = NULL
) {
  library(dplyr)
  library(ggplot2)

  df <- read_bmsspf_json_tidy(json_path) %>%
    bmsspf_largest_only(graph_file = graph_file) %>%
    rowwise() %>%
    mutate(ablation = bmsspf_ablation_label(pscase_mode, countmin_search)) %>%
    ungroup()

  best_per_group <- df %>%
    group_by(graph_file, frontier, ablation) %>%
    slice_min(order_by = time_full, n = 1, with_ties = FALSE) %>%
    ungroup()

  plot_df <- best_per_group %>%
    group_by(frontier, ablation) %>%
    summarise(
      mean_time = mean(time_full, na.rm = TRUE),
      sd_time = ifelse(sum(!is.na(time_full)) > 1, sd(time_full, na.rm = TRUE), 0),
      .groups = "drop"
    )

  plot_df$ablation <- factor(
    plot_df$ablation,
    levels = c("none", "p=s only", "countmin only", "both")
  )

  plt <- ggplot(plot_df, aes(x = ablation, y = mean_time, fill = frontier)) +
    geom_col(position = position_dodge(width = 0.8)) +
    geom_errorbar(
      aes(ymin = mean_time - sd_time, ymax = mean_time + sd_time),
      position = position_dodge(width = 0.8),
      width = 0.2
    ) +
    labs(
      x = NULL,
      y = "Mean runtime (ms)",
      fill = "Frontier"
    ) +
    theme_minimal()

  ggsave(filename, plt, width = 8, height = 5.5)
  invisible(plot_df)
}

# ---------- 4) BF_steps response plot ----------

plot_bmsspf_bf_steps <- function(
  json_path,
  filename = "figures/bmsspf_bf_steps.pdf",
  graph_file = NULL
) {
  library(dplyr)
  library(ggplot2)

  df <- read_bmsspf_json_tidy(json_path) %>%
    bmsspf_largest_only(graph_file = graph_file) %>%
    filter(countmin_search != "false")

  plot_df <- df %>%
    group_by(frontier, pscase_mode, countmin_search, countmin_mode, BF_steps) %>%
    summarise(
      mean_time = mean(time_full, na.rm = TRUE),
      sd_time = ifelse(sum(!is.na(time_full)) > 1, sd(time_full, na.rm = TRUE), 0),
      .groups = "drop"
    )

  plt <- ggplot(
    plot_df,
    aes(x = BF_steps, y = mean_time, color = frontier, linetype = pscase_mode, group = interaction(frontier, pscase_mode))
  ) +
    geom_line() +
    geom_point() +
    geom_errorbar(aes(ymin = mean_time - sd_time, ymax = mean_time + sd_time), width = 0.1) +
    facet_grid(countmin_search ~ countmin_mode, scales = "free_y") +
    labs(
      x = "BF steps",
      y = "Mean runtime (ms)",
      color = "Frontier",
      linetype = "p=s mode"
    ) +
    theme_minimal()

  ggsave(filename, plt, width = 11, height = 8)
  invisible(plot_df)
}

# ---------- 5) Runtime component breakdown for best configs ----------

plot_bmsspf_components <- function(
  json_path,
  filename = "figures/bmsspf_components.pdf",
  graph_file = NULL,
  top_n = 6
) {
  library(dplyr)
  library(tidyr)
  library(ggplot2)

  df <- read_bmsspf_json_tidy(json_path) %>%
    bmsspf_largest_only(graph_file = graph_file)

  ranked <- df %>%
    group_by(
      pscase_type, pscase_mode, frontier,
      countmin_search, countmin_mode, BF_steps
    ) %>%
    summarise(
      mean_time_full = mean(time_full, na.rm = TRUE),
      mean_find_pivot = mean(time_find_pivot, na.rm = TRUE),
      mean_base_case = mean(time_base_case, na.rm = TRUE),
      mean_D_op = mean(time_D_op, na.rm = TRUE),
      mean_batch_prepend = mean(time_batch_prepend, na.rm = TRUE),
      mean_bmssp = mean(time_bmssp, na.rm = TRUE),
      .groups = "drop"
    ) %>%
    arrange(mean_time_full) %>%
    slice_head(n = top_n) %>%
    mutate(
      config = paste(
        frontier,
        pscase_mode,
        countmin_search,
        countmin_mode,
        paste0("BF=", BF_steps),
        sep = "\n"
      )
    )

  plot_df <- ranked %>%
    select(
      config,
      mean_find_pivot,
      mean_base_case,
      mean_D_op,
      mean_batch_prepend,
      mean_bmssp
    ) %>%
    pivot_longer(
      -config,
      names_to = "component",
      values_to = "time_ms"
    ) %>%
    mutate(
      component = recode(
        component,
        mean_find_pivot = "Find Pivots",
        mean_base_case = "Base Case",
        mean_D_op = "Queue Ops",
        mean_batch_prepend = "Batch Prepend",
        mean_bmssp = "BMSSP Recursion"
      )
    )

  config_order <- ranked$config
  plot_df$config <- factor(plot_df$config, levels = config_order)

  plt <- ggplot(plot_df, aes(x = config, y = time_ms, fill = component)) +
    geom_col() +
    labs(
      x = NULL,
      y = "Mean runtime contribution (ms)",
      fill = "Component"
    ) +
    theme_minimal() +
    theme(axis.text.x = element_text(angle = 45, hjust = 1))

  ggsave(filename, plt, width = 10, height = 6.5)
  invisible(plot_df)
}

exp_random_timed <- function() {
  graphs <- c("randomD", "randomE", "randomG", "randomH", "randomT")
  for (g in graphs) {
    cfg_new <- default_cfg
    cfg_new$graph <- g
    write_cfg(cfg = cfg_new)
    runSearch()
    file.copy(from="experiments/run_stats.json", to=paste0("experiments/random_timed/", g, ".json"), overwrite = TRUE)
  }
}

plot_random_timed <- function() {
  graphs <- c("randomD", "randomE", "randomG", "randomH", "randomT")
  for (g in graphs) {
    read_path <<- paste0("experiments/random_timed/", g, ".json")
    plot_mean_times(filename = paste0("figures/random_timed/", g, "_mean.pdf"))
    plot_d_insert_times(filename = paste0("figures/random_timed/", g, "_insert.pdf"))
    plot_pivot(filename = paste0("figures/random_timed/", g, "_pivot.pdf"))
  }
}

exp_global_timed <- function() {
  graphs <- c("randomD", "randomE", "randomG", "randomH", "randomT")
  for (g in graphs) {
    cfg_new <- default_cfg
    cfg_new$graph <- g
    write_cfg(cfg = cfg_new)
    runGlobalSearch()
    file.copy(from="experiments/run_stats.json", to=paste0("experiments/global_timed/", g, ".json"), overwrite = TRUE)
  }
}

exp_5k_ml <- function() {
  graphs <- c("randomD", "randomE", "randomG", "randomH", "randomT", "RF", "RD", "mix_real", "mix_gen", "mix_all")
  for (g in graphs) {
    cfg_new <- default_cfg
    cfg_new$graph <- g
    write_cfg(cfg = cfg_new)
    run_5k()
    print("Completed experiment for graph.")
  }
}

exp_bmsspf_comb <- function() {
  graphs <- c("randomD", "randomE", "randomG", "randomH", "randomT")
  for (g in graphs) {
    cfg_new <- default_cfg
    cfg_new$graph <- g
    write_cfg(cfg = cfg_new)
    runBMSSPFSearch()
    print("completed experiment for graph")
    file.copy(from="experiments/run_stats_bmsspf.json", to=paste0("experiments/bmsspf_comb_small/", g, ".json"), overwrite = TRUE)
  }
}

table_bmsspf_comb <- function() {
  graphs <- c("randomD", "randomE", "randomG", "randomH", "randomT")
  for (g in graphs) {
    tab <- bmsspf_largest_table(
      json_path = paste0("experiments/bmsspf_comb/", g, ".json"),
      out_path = paste0("tables/bmsspf_comb/", g, ".tsv")
    )
  }
}


plot_all_bmsspf <- function() {
  graphs <- c("randomD", "randomE", "randomG", "randomH", "randomT")
  for (g in graphs) {
    plot_bmsspf_largest_heatmap(
      paste0("experiments/bmsspf_comb/", g, ".json"),
      paste0("figures/bmsspf_largest_heatmap/", g, "_heatmap.pdf")
    )

    plot_bmsspf_best_speedup(
      json_paths = paste0("experiments/bmsspf_comb/", c("randomD","randomE","randomG","randomH","randomT"), ".json"),
      labels = c("randomD","randomE","randomG","randomH","randomT"),
      paste0(filename = "figures/bmsspf_best_speedup/", g, "_best_speedup.pdf")
    )

    plot_bmsspf_ablation(
      paste0("experiments/bmsspf_comb/", g, ".json"),
      paste0("figures/bmsspf_ablation/", g, "_ablation.pdf")
    )

    plot_bmsspf_bf_steps(
      paste0("experiments/bmsspf_comb/", g, ".json"),
      paste0("figures/bmsspf_bf_steps/", g, "_bfsteps.pdf")
    )

    plot_bmsspf_components(
      paste0("experiments/bmsspf_comb/", g, ".json"),
      paste0("figures/bmsspf_components/", g, "_components.pdf")
    )
  }
}

######### Main ##########

# to run individual algorithms, run this:
runSearch()

########################

# exp_random_timed()
# plot_random_timed()

# runSearch()
# exp_bmsspf_comb()

# table_bmsspf_comb()
# plot_all_bmsspf()

# exp_global_timed()
# exp_5k_ml()


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
