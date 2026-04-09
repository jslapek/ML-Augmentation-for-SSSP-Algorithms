if (!requireNamespace("igraph", quietly = TRUE)) install.packages("igraph")
library(igraph)

plot_gr_file <- function(gr_path,
                         out_dir = "figures",
                         base_name = "graph_publication") {
  dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

  if (!file.exists(gr_path)) {
    warning(sprintf("File not found: %s", gr_path))
    return(invisible(NULL))
  }

  lines <- readLines(gr_path, warn = FALSE)
  edge_lines <- lines[grepl("^a\\s+", lines)]

  if (length(edge_lines) == 0) {
    warning(sprintf("No edges found in: %s", gr_path))
    return(invisible(NULL))
  }

  edges <- do.call(rbind, lapply(edge_lines, function(x) {
    p <- strsplit(trimws(x), "\\s+")[[1]]
    c(
      from = as.integer(p[2]),
      to = as.integer(p[3]),
      weight = as.numeric(p[4])
    )
  }))

  edges_df <- data.frame(
    from = edges[, "from"],
    to = edges[, "to"],
    weight = edges[, "weight"]
  )

  g <- graph_from_data_frame(edges_df, directed = TRUE)

  set.seed(42)
  lay <- layout_with_fr(as.undirected(g, mode = "collapse"))

  w <- E(g)$weight
  if (length(w) == 0 || max(w) == min(w)) {
    edge_w <- rep(1.4, ecount(g))
  } else {
    w_scaled <- (w - min(w)) / (max(w) - min(w))
    edge_w <- 0.3 + 2.8 * (w_scaled^2.2)
  }

  pdf(file.path(out_dir, paste0(base_name, ".pdf")), width = 6, height = 5)
  par(mar = c(0, 0, 0, 0), bg = "white")
  plot(
    g,
    layout = lay,
    vertex.size = 7,
    vertex.label = NA,
    vertex.color = "grey25",
    vertex.frame.color = "grey10",
    vertex.frame.width = 0.4,
    edge.color = adjustcolor("grey45", alpha.f = 0.7),
    edge.width = edge_w,
    edge.arrow.size = 0.4,
    edge.curved = 0.06
  )
  dev.off()

  invisible(file.path(out_dir, paste0(base_name, ".pdf")))
}

# graph_types <- c("randomD", "randomE", "randomG", "randomH", "randomT")
graph_types <- c("randomT")
# graph_types <- c("randomD", "randomG")

size <- "16"
for (graph_type in graph_types) {
  gr_path <- file.path("graphs", graph_type, size, "graph_2.gr")
  out_name <- paste0(graph_type, "_", size)
  plot_gr_file(gr_path, out_dir = "figures/graph_visualisations", base_name = out_name)
}

graph_types <- c("RD_5k_gr", "RF_5k_gr")

for (graph_type in graph_types) {
  gr_path <- file.path("graphs", graph_type, "graph_4.gr")
  out_name <- paste0(graph_type, "_", size)
  plot_gr_file(gr_path, out_dir = "figures/graph_visualisations", base_name = out_name)
}