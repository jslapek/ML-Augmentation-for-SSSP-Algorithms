#!/bin/bash

EXECUTABLE="C:/Users/Jakub/Documents/stuff/diss/project/scripts/strip-rrg-generator.exe"
SOURCE_FILE="C:/Users/Jakub/Documents/stuff/diss/project/cpp_pkg/strip-rrg-generator.cpp"

N=1000
STRIP_WIDTH=2.5
OUTPUT_DIR="C:/Users/Jakub/Documents/stuff/diss/project/graphs/randomE_5k"

echo "Compiling..."
g++ -std=c++20 -O3 "$SOURCE_FILE" -o "$EXECUTABLE"

mkdir -p "$OUTPUT_DIR"

for M in {1..5000}; do
    OUTPUT_FILE="${OUTPUT_DIR}/graph_${M}.gr"
    "$EXECUTABLE" "$N" 3 "$STRIP_WIDTH" "$M" > "$OUTPUT_FILE"
done

echo "Done."