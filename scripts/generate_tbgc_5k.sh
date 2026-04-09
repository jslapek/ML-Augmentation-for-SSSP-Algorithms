#!/bin/bash

EXECUTABLE="C:/Users/Jakub/Documents/stuff/diss/project/scripts/tbgc-generator.exe"
SOURCE_FILE="C:/Users/Jakub/Documents/stuff/diss/project/cpp_pkg/tbgc-generator.cpp"

N=1000
OUTPUT_DIR="C:/Users/Jakub/Documents/stuff/diss/project/graphs/randomT_5k"

echo "Compiling..."
g++ -std=c++20 -O3 "$SOURCE_FILE" -o "$EXECUTABLE"

mkdir -p "$OUTPUT_DIR"

for M in {1..5000}; do
    OUTPUT_FILE="${OUTPUT_DIR}/graph_${M}.gr"
    "$EXECUTABLE" "$N" 3 100000 "$M" > "$OUTPUT_FILE"

    if (( M % 100 == 0 )); then
        echo "Generated $M graphs..."
    fi
done

echo "Done."