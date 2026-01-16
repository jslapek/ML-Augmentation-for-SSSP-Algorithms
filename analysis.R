# Download packages
if (!requireNamespace("reticulate", quietly = TRUE)) install.packages("reticulate")
if (!requireNamespace("ggplot2", quietly = TRUE))    install.packages("ggplot2")

# System variables
Sys.setenv(RETICULATE_PYTHON = "C:/Users/Jakub/anaconda3/python.exe")

# Packages
library(reticulate)
library(ggplot2)
py_config()



run_module <- import_from_path("run", "pkg")

experiments = list()
ns <- c(50, 100, 200, 500, 750, 1000, 1500, 2000, 3000, 5000, 7500, 10000, 15000, 20000, 50000, 75000, 100000)



for (i in ns) {
    cfg <- run_module$RunConfig(
        alg = "bmssp",
        heap = "binary",
        frontier = "block",
        graph = "random",
        seed = 42L,
        n = as.integer(i),
        m = as.integer(i) * 4L,
        transform = FALSE,
        transform_delta = 4L,
        niters = 10L,
        nsources = 1L
    )
    experiments[[length(experiments) + 1]] <- run_module$runSearch(cfg)
    cat(sprintf("Completed experiment %d\n", i))
}
