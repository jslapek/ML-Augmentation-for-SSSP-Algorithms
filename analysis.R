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
experiments <- run_module$runSearch()