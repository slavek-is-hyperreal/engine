#!/bin/bash

BASE_NAME="eml_code_bundle_$(date +%Y-%m-%d)"
OUTPUT="${BASE_NAME}.txt"
COUNTER=1

while [ -f "$OUTPUT" ]; do
    OUTPUT="${BASE_NAME}_${COUNTER}.txt"
    COUNTER=$((COUNTER + 1))
done

# Wyczyść plik wynikowy
> "$OUTPUT"

echo "Generowanie mapy projektu i łączenie kodu do $OUTPUT..."

# 1. Nagłówek i Mapa Projektu
echo "======================================" >> "$OUTPUT"
echo "PROJECT MAP (Tree depth 20)" >> "$OUTPUT"
echo "======================================" >> "$OUTPUT"
# Generujemy drzewo, omijając duże foldery binarne/dane
tree -L 20 -I "target|research|venv|models|paper|node_modules|.git" >> "$OUTPUT"
echo -e "\n\n" >> "$OUTPUT"

# 2. Łączenie plików (kod źródłowy + logi testów)
# Omijamy: target, ukryte, research, venv, modele, .md, .txt, .csv
find . -type f \
    \( -name "*.tests" -o \
       \( -not -path "*/target/*" \
          -not -path "*/target_zfs/*" \
          -not -path "*/.*" \
          -not -path "*/research/*" \
          -not -path "*/venv/*" \
          -not -path "*/models/*" \
          -not -path "*/paper/*" \
          -not -name "eml_code_bundle*.txt" \
          -not -name "eml_trs_bundle*.txt" \
          -not -name "test_gguf*" \
          -not -name "*.gguf" \
          -not -name "*.bin" \
          -not -name "*.md" \
          -not -name "*.txt" \
          -not -name "*.csv" \) \) | sort | while read -r file; do
    echo "======================================" >> "$OUTPUT"
    echo "File: $file" >> "$OUTPUT"
    echo "======================================" >> "$OUTPUT"
    cat "$file" >> "$OUTPUT"
    echo -e "\n" >> "$OUTPUT"
done

echo "Gotowe! Cały kod i logi .tests znajdują się w: $OUTPUT"
