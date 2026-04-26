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

# Znajdź pliki, omiń katalogi target i ukryte (np. .git)
find . -type f -not -path "*/target/*" -not -path "*/\.*" -not -name "eml_trs_bundle*.txt" | sort | while read -r file; do
    echo "======================================" >> "$OUTPUT"
    echo "Plik: $file" >> "$OUTPUT"
    echo "======================================" >> "$OUTPUT"
    cat "$file" >> "$OUTPUT"
    echo -e "\n" >> "$OUTPUT"
done

echo "Gotowe!"
