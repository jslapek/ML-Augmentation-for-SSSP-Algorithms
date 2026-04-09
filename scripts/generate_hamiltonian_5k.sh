#!/bin/bash

EXECUTABLE="C:/Users/Jakub/Documents/stuff/diss/project/scripts/hamiltonian-random-graph-generator.exe"
SOURCE_FILE="C:/Users/Jakub/Documents/stuff/diss/project/cpp_pkg/hamiltonian-random-graph-generator.cpp"

N=1000
OUTPUT_DIR="C:/Users/Jakub/Documents/stuff/diss/project/graphs/randomH_5k"

echo "Compiling..."
g++ -std=c++20 -O3 "$SOURCE_FILE" -o "$EXECUTABLE"

mkdir -p "$OUTPUT_DIR"

for M in {1..5000}; do
    OUTPUT_FILE="${OUTPUT_DIR}/graph_${M}.gr"
    "$EXECUTABLE" "$N" 100000 3 "$M" > "$OUTPUT_FILE"
done

echo "Done."