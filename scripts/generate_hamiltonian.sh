#!/bin/bash

EXECUTABLE="C:/Users/Jakub/Documents/stuff/diss/project/scripts/hamiltonian-random-graph-generator.exe"

echo "test"
g++ -std=c++20 -O3 "C:/Users/Jakub/Documents/stuff/diss/project/cpp_pkg/hamiltonian-random-graph-generator.cpp" -o "$EXECUTABLE"

N_VALUES=()
N=8
M=5
MAX_N=50000000

while (( N <= MAX_N )); do
    N_VALUES+=("$N")
    # Using arithmetic expansion to multiply N by 2
    N=$(( N * 2 ))
done

for N in "${N_VALUES[@]}"; do
    for M in {1,2,3,4,5}; do
        OUTPUT_FILE="C:/Users/Jakub/Documents/stuff/diss/project/graphs/randomH/${N}/graph_${M}.gr"
        mkdir -p "$(dirname "$OUTPUT_FILE")"
        "$EXECUTABLE" "$N" 100000 3 "$M" > "$OUTPUT_FILE"
    done
done
