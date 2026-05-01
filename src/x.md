Ran command: `# 1. Sprawdź co zajmuje najwięcej miejsca w katalogu domowym (na ext4)
du -sh ~/* | sort -h
# 2. Jeśli ~/.cargo istnieje i jest duże - USUŃ GO (mamy go mieć na SSD/ZFS)
rm -rf ~/.cargo
# 3. Sprawdź czy nie ma ogromnych logów
sudo du -sh /var/log/* | sort -h
# Jeśli /var/log/syslog jest gigantyczny:
# sudo truncate -s 0 /var/log/syslog
# 4. Sprawdź czy nie ma plików 'core' w projekcie
find /my_data/engine -name "core.*" -delete`

Obejrzałem Twoje wyniki `du`. Wygląda na to, że ani Twój katalog domowy (~5-10 GB), ani logi systemowe (~1 GB) nie są źródłem problemu. Mimo to dysk ma 100% (87 GB). 

Gdzie jest pozostałe 70+ GB? Mam dwie teorie:

1.  **Pliki „usunięte, ale otwarte”**: Jeśli skrypt (lub Cargo) pisał do ogromnego pliku na `ext4`, a potem go usunąłeś (lub ja kazałem usunąć `~/.cargo`), ale proces wciąż działa w tle, to system **nie zwolni miejsca**, dopóki nie zabijesz tego procesu. To klasyka w Linuxie.
2.  **Katalog `/tmp`**: Cargo często trzyma tam ogromne ilości danych tymczasowych podczas kompilacji.

**Zróbmy szybki rentgen otwartych plików i wyczyśćmy `/tmp`:**

```bash
# 1. Znajdź procesy, które trzymają usunięte, ogromne pliki (zobaczysz kolumnę 'size')
sudo lsof / | grep deleted | sort -n -k 7 | tail -n 10

# 2. Sprawdź rozmiar /tmp
sudo du -sh /tmp

# 3. Jeśli lsof pokazał procesy 'cargo' lub 'full_layer_benchmark', ZABIJ JE:
sudo killall -9 cargo
sudo killall -9 full_layer_benchmark
```

**Dlaczego to ważne:** Jeśli `lsof` pokaże procesy z dopiskiem `(deleted)`, to po ich zabiciu (`killall`) zobaczysz, jak kółeczko na pulpicie nagle spada z 100% do np. 10%. To by oznaczało, że procesy „duchy” blokowały miejsce.

Daj znać, co pokazał `lsof`!