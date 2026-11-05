# Algorytm TBD - Track Before Detect


## Opis

Projekt do śledzenia pozycji obiektów na zaszumionym tle na podstawie wcześniej zarejestrowanych klatek.

Wykonano w ramach projektu Zachodniopomorskiego Uniwersytetu Technologicznego w Szczecinie pn. "Wsparcie studentów zespołu badawczo-wdrożeniowego realizującego wizyjny wielokamerowy system nadzoru przestrzeni powietrznej do śledzenia bezzałogowych statków powietrznych (BSP/UAV) z wykorzystaniem algorytmów śledzenia przed detekcją (TBD – Track-Before-Detect)" w ramach projektu niekonkurencyjnego pt. „Wsparcie studentów w zakresie podniesienia ich kompetencji i umiejętności” (Działanie 01.05 Umiejętności w szkolnictwie wyższym, Program Fundusze Europejskie dla Rozwoju Społecznego 2021-2027).

## Cechy

- Możliwość wykrywania bardzo małych zarejestrowanych obiektów 
- Algorytm nie używa cech obiektów (np. kształ, kolor czy rozmiar) tylko jego wyróżnianie się z szumu
- Skalowalność systemu ("horyzontalnie" i "wertykalnie")
- Działanie na rdzeniach CUDA


## Działanie

Program `main.cu` wczytuje wybrany folder (pierwszy argument uruchamiania) w którym znajdują się klatki obrazu w formacie .bmp ułożone kolejno alfabetycznie. Użytkownik może podać również głębokość z jaką ma działać algorytm (ile klatek w stecz ma być uwzględnianych) oraz nazwę pliku wyjściowego.

Po uruchomieniu program sam na podstawie pierwszej klatki ustawi wielkość macierzy dla wszystkich kolejnych klatek.

Wyniki są zapisywane do pliku `wykryte_pozycje.csv` lub do wskazanego pliku w formacie .csv jako kolejne wiersze <pozycja_x>, <pozycja_y> wykrytej pozycji w danej klatce.


- Przykład działania - Na pierwszym obrazie widać migająco losowo obiekt, na drugim jest on dołączony do losowego szumu typu "sól i pieprz" (dla łatwiejszej znalezienia go dla człowieka, porusza się on liniowo po przekątnej). Trzeci obraz ukazuje krzyżyk wygenerowany na podstawie pozycji z pliku `wykryte_pozycje.csv`. Głębokość działania algorytmu wynosiła 7.

![Przykładowy wynik działania programu](example/comparison.gif)


## Instalacja

Do działania programu potrzebne jest środowisko Linux wraz z zainstalowanym CUDA Toolkit firmy NVidia.
Dla kart z serii RTX 50xx (Blackwell, np. RTX 5090) wymagane jest CUDA **12.8 lub nowsze**.

```
make                  # RTX 5090 (sm_120)
make ARCH=sm_89       # np. RTX 40xx
make ARCH=native      # architektura karty zainstalowanej w komputerze
```

lub ręcznie: `nvcc -O3 -std=c++17 -arch=sm_120 src/main.cu -o main`.


## Użycie

Skompilowany program uruchamiamy podając jego ścieżkę w terminalu, podając obowiąkowo ścieżkę (relatywną lub globalną) do folderu w którym znajdują się klatki obrazu, oraz opcjonalnie podając głębokość warty z jaką ma działać algorytm i nazwę pliku wyjściowowego.

```
 ./main frames 7 wynik.csv

 >>>

Wykryty rozmiar: 100x100, klatek: 100, glebokosc: 7, promien: 1
GPU: NVIDIA GeForce RTX 5090 (sm_120, 170 SM)
Czas calkowity: ... s (... FPS, z wczytywaniem plikow)
Wykryte punkty przelotu zostaly zapisane do pliku: wynik.csv

 ```

Dodatkowe opcje:

- `--verify` – każde okno jest dodatkowo liczone pierwotnym algorytmem (warstwa po warstwie + `thrust::max_element`) i porównywane z wynikiem szybkiej ścieżki (wolne, tylko do sprawdzenia poprawności),
- `--threads N` – liczba wątków wczytujących pliki BMP (domyślnie min(8, liczba rdzeni)),
- `--prog K`, `--wynik`, `--tlo A`, `--polaryzacja P` – próg wykrycia i model tła (opis niżej),
- `--obiekty K` – liczba obiektów wykrywanych w każdej klatce (domyślnie 1, maksymalnie 256); dla `K > 1` zamiast pojedynczego maksimum wyszukiwane jest `K` najlepszych maksimów lokalnych końcowej mapy kosztu,
- `--nms R` – promień tłumienia niemaksymalnego (NMS) w pikselach dla `--obiekty K > 1` (domyślnie 8): punkt jest brany pod uwagę tylko wtedy, gdy ma największy koszt w swoim otoczeniu `(2R+1)×(2R+1)` (przy remisie wygrywa mniejszy indeks piksela, tak jak dla `K = 1`), więc dwa wykryte punkty są zawsze oddalone o więcej niż `R` w poziomie lub w pionie,
- `--promien P` – maksymalne przesunięcie obiektu między kolejnymi klatkami w pikselach: poprzednik punktu jest szukany w otoczeniu (2P+1)×(2P+1) zamiast 3×3 (domyślnie `P = 1`, czyli dotychczasowe działanie). Działa w trybie folderu i w `--bench` (również z `--verify`),
- `--predkosc V` – tylko dla `--bench`: prędkość syntetycznego obiektu w px/klatkę (domyślnie równa `P`),
- `--bench <SZER>x<WYS> [klatek] [glebokosc]` – benchmark na syntetycznych klatkach generowanych na GPU (bez dysku), np.:

```
 ./main --bench 2560x1440 300 7
 ./main --bench 2560x1440 50 7 --verify
 ./main --bench 2560x1440 300 7 --obiekty 2 --nms 8   # dwa obiekty syntetyczne
 ./main --bench 2560x1440 300 7 --promien 3          # obiekt porusza się o 3 px/klatkę
 ./main frames 7 wynik.csv --promien 2
```

### Wiele obiektów (`--obiekty K`)

Dla domyślnego `K = 1` plik wyjściowy ma format `<x>,<y>` (bez zmian względem poprzednich wersji). Dla `K > 1` każdy wiersz (jedna klatka) zawiera do `K` trójek posortowanych malejąco według kosztu:

```
x1,y1,w1,x2,y2,w2,...,xK,yK,wK
```

gdzie `x`, `y` to pozycja (w tym samym układzie co dla `K = 1`), a `w` to *wynik* punktu, czyli (koszt − średnia) / odchylenie standardowe końcowej mapy kosztu (patrz „Próg wykrycia”). Z `--prog` punkty o wyniku poniżej progu są pomijane. Jeśli obraz ma mniej maksimów lokalnych niż `K` (albo mniej punktów powyżej progu), wiersz zawiera mniej trójek. Pozycja pierwszej trójki jest zawsze taka sama jak wynik dla `K = 1`.

```
 ./main frames 7 wynik.csv --obiekty 3 --nms 10
```

Wyszukiwanie odbywa się w całości na GPU, asynchronicznie (bez synchronizacji na klatkę): ostatnia warstwa zapisuje mapę kosztu do bufora roboczego, kernel na kafelkach 32×32 (pamięć współdzielona z otoczką `R`, maksimum rozdzielne: poziome, potem pionowe) wyznacza maksima lokalne i zachowuje co najwyżej `K` najlepszych z każdego kafelka, a jednoblokowy kernel wybiera z nich `K` najlepszych (radix select + sortowanie). Promień `R` jest ograniczony rozmiarem pamięci współdzielonej (ok. 180 pikseli na RTX 5090). Z `--verify` wszystkie `K` punktów każdego okna są dodatkowo porównywane z referencją na CPU.


## Próg wykrycia i model tła

**Próg wykrycia (`--prog K`, `--wynik`).** Dla każdej klatki liczony jest *wynik* = (maksimum − średnia) / odchylenie standardowe mapy kosztu ostatniej warstwy, czyli o ile odchyleń standardowych najlepsza ścieżka wyróżnia się z tła. Z `--prog K` klatki z wynikiem < K dostają w CSV pozycję `-1,-1` (brak obiektu). `--wynik` dopisuje wynik jako trzecią kolumnę. Bez tych opcji format CSV jest taki jak wcześniej.

Na końcu program wypisuje rozkład wyniku (min / mediana / p95 / max). Próg najłatwiej dobrać, uruchamiając program na nagraniu **bez obiektu** i ustawiając `K` nieco powyżej maksymalnego wyniku.

**Model tła (`--tlo A`, `--polaryzacja jasny|ciemny|oba`).** Dla każdego piksela utrzymywana jest wykładniczo ważona średnia i wariancja jasności (współczynnik `A`, np. `0.05` ≈ pamięć ~40 klatek). Do algorytmu zamiast jasności trafia znormalizowane odchylenie od tła `z = (jasność − średnia) / σ` (σ uśrednione w oknie 3×3). Dzięki temu:

- statyczne jasne elementy sceny (budynki, słońce, gradient nieba) przestają „wygrywać” z obiektem,
- szum ma tę samą skalę w całym obrazie, więc próg `--prog` nie zależy od jasności tła,
- `--polaryzacja ciemny` wykrywa obiekty ciemniejsze od tła (np. dron na tle jasnego nieba), a `oba` – jaśniejsze i ciemniejsze.

Model tła wymaga nieruchomej kamery i przez pierwsze ~1/`A` klatek się „uczy”. Na sztucznych danych z bardzo silnym szumem, gdzie obiekt ma po prostu maksymalną jasność (jak w `example/frames`), lepiej działa tryb domyślny bez `--tlo`.

```
 ./main nagranie 7 wynik.csv --tlo 0.05 --polaryzacja oba --prog 8 --wynik
```


## Wydajność (2K, głębokość ≥ 7)

Program przetwarza klatki strumieniowo, więc działa w czasie rzeczywistym niezależnie od długości sekwencji:

- na GPU trzymane jest tylko `L` ostatnich klatek (bufor pierścieniowy), a nie cała sekwencja – wcześniej 2K × 1000 klatek wymagało ~30 GB VRAM i tyle samo RAM, a indeksowanie przepełniało `int` po ~580 klatkach 2K,
- pliki BMP są czytane równolegle przez kilka wątków do pamięci *pinned*, kopiowane asynchronicznie i konwertowane do skali szarości na GPU,
- wszystkie `L-1` warstw programowania dynamicznego liczone są jednym kernelem na kafelkach 64×32 w pamięci współdzielonej (z otoczką `L-1` pikseli), a wyszukiwanie maksimum jest połączone z ostatnią warstwą (`atomicMax` na kluczu 64-bit) – wcześniej było `L-1` osobnych kerneli, każdy z `cudaMalloc/cudaFree`, zapis nieużywanej tablicy indeksów i synchroniczne `thrust::max_element` + `cudaMemcpy` na każdą klatkę,
- w pętli nie ma żadnych alokacji ani synchronizacji CPU↔GPU; wyniki są pobierane raz na końcu,
- głębokości większe niż 12 są dzielone na kilka (możliwie równych) przebiegów kernela, więc działa dowolna głębokość,
- dla `--promien P > 1` otoczka kafelka rośnie o `P` pikseli na warstwę, a maksimum z otoczenia (2P+1)×(2P+1) liczone jest separowalnie (najpierw w poziomie, potem w pionie, koszt O(P) zamiast O(P²) na piksel). Liczba warstw w jednym przebiegu jest dobierana tak, aby kafelek mieścił się w połowie pamięci współdzielonej SM (min. 2 bloki na SM) – na RTX 5090: P=2 → 8, P=3 → 5, P=4 → 4, P=5 → 3, P=6…8 → 2, P≥9 → 1 warstwa; maksymalny promień ograniczony jest pamięcią współdzieloną bloku (na RTX 5090 ok. 32). Dla `P = 1` używana jest dotychczasowa ścieżka 3×3.

Koszt obliczeń GPU na klatkę 2560×1440 przy `L = 7` to rząd ułamka milisekundy, więc przy odczycie z dysku wąskim gardłem jest przepustowość dysku (klatka BMP 24-bit 2560×1440 ≈ 11 MB, czyli ~170 MB/s dla 15 FPS).
Wyniki są bit w bit identyczne z pierwotną wersją (dla obrazów kwadratowych; dla niekwadratowych poprawiono błędną interpretację wymiarów).


## Wymagania

- System operacyjny Linux (np. Ubuntu 22.04)
- CUDA Toolkit 12.8+ (dla RTX 50xx; dla starszych kart wystarczy 12.0 z odpowiednim `ARCH`)
- Kompilator nvcc
