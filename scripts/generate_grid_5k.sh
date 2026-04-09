#!/bin/bash

EXECUTABLE="C:/Users/Jakub/Documents/stuff/diss/project/scripts/grid-generator.exe"
SOURCE_FILE="C:/Users/Jakub/Documents/stuff/diss/project/cpp_pkg/grid-generator.cpp"

N=1000
ROWS=$(python -c "import math; print(max(1, math.isqrt(max(1, $N // 4))))" | tr -d '\r')
COLS=$(( (N + ROWS - 1) / ROWS ))

OUTPUT_DIR="C:/Users/Jakub/Documents/stuff/diss/project/graphs/randomG_5k"

echo "Compiling..."
g++ -std=c++20 -O3 "$SOURCE_FILE" -o "$EXECUTABLE"

mkdir -p "$OUTPUT_DIR"

for M in {1..5000}; do
    OUTPUT_FILE="${OUTPUT_DIR}/graph_${M}.gr"
    "$EXECUTABLE" "$ROWS" "$COLS" 0 "$M" > "$OUTPUT_FILE"
done

echo "Done."