# Podsumowanie Implementacji EML-TRS

## Dzień 1 (Fundament)
Zrealizowano:
- Utworzono szkielet projektu Cargo `eml-trs`.
- Wdrożono struktury AST (`EmlNode`) i podstawowe konstruktory (`ast.rs`).
- Zbudowano model kosztów na podstawie exhaustive search (`cost_model.rs`).
- Testy zaliczone z powodzeniem.

## Dzień 2 (Algorytmy)
Zrealizowano:
- Zaimplementowano Term Rewriting System (`trs.rs`) umożliwiający redukcję złożoności ciągów matematycznych operujących na grafach EML.
- Zaimplementowano reguły Constant Foldingu w `constant_fold.rs`, zwijające m.in. stałe mnożenia do postaci pre-komputowanej na wczesnym etapie.
- Stworzono funkcje budujące drzewa EML dla dot product metodą ASIS (`asis.rs`), redukując liczebność węzłów z asympotycznej granicy ~36K na ~28K.
- Pomyślnie zintegrowano poprawki, testy zakończyły się bez błędów.

## Dzień 3 (Infrastruktura i Wyniki)
Zrealizowano:
- Przekształcono model drzewiasty do DAG w `dag.rs`, umożliwiając optymalne współdzielenie węzłów (np. zmniejszając koszt przy RMSNorm).
- Napisano benchmarki dla logiki TinyLlama. Skompilowano je z sukcesem generując oczekiwany plik statystyk `paper/results/tinyllama_costs.csv`.
- Zaimplementowano benchmarki testujące mechanizmy TRS (zgodnie ze wskaźnikami - 100% redukcja zbędnych węzłów dla `ln(exp(x))`).

## Dzień 4 (Backendy i Dokumentacja)
Zrealizowano:
- Opracowano szablony WGSL implementujące szybkie operacje w oparciu o Minimax dla funkcji z bazy `exp` oraz `ln` (`backends/wgsl.rs`).
- Zaprojektowano prosty backend operujący decyzyjnie względem ALU i operacji kosztowych dla architektur klasycznych (`backends/alu.rs`).
- Utworzono plik z główną dokumentacją repozutorium `README.md` podsumowujący wyniki i odkrycia matematyczne.
- Ponownie wszystkie testy wykonane przy użyciu `cargo test` zakończyły się sukcesem. Projekt poprawnie się buduje i przechodzi kompletną ścieżkę optymalizacyjną.
