#!/bin/bash

BASE_NAME="eml_trs_bundle_$(date +%Y-%m-%d)"
OUTPUT="${BASE_NAME}.txt"
COUNTER=1

while [ -f "$OUTPUT" ]; do
    OUTPUT="${BASE_NAME}_${COUNTER}.txt"
    COUNTER=$((COUNTER + 1))
done

# Wyczyść plik wynikowy
> "$OUTPUT"

echo "Łączenie plików projektu do $OUTPUT..."

# Znajdź pliki, omiń katalogi target, research, paper, i inne niepotrzebne dane
find . -type f -not -path "*/target/*" -not -path "*/\.*" -not -path "*/research/*" -not -path "*/venv/*" -not -path "*/models/*" -not -path "*/paper/*" -not -name "eml_trs_bundle*.txt" -not -name "test_gguf*" -not -name "*.gguf" | sort | while read -r file; do
    echo "======================================" >> "$OUTPUT"
    echo "Plik: $file" >> "$OUTPUT"
    echo "======================================" >> "$OUTPUT"
    cat "$file" >> "$OUTPUT"
    echo -e "\n" >> "$OUTPUT"
done

echo "Gotowe!"
