# Algorytm TBD - Track Before Detect


## Opis

Projekt do śledzenia pozycji obiektów na zaszumionym tle na podstawie wcześniej zarejestrowanych klatek.

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

Do działania programu potrzebne jest środowisko Linux wraz z zainstalowanymi biliotekami CUDA od firmy NVidia.
Program należy skompilować komendą `nvcc main.cu -o main` znajdując się w tym samym folderze.


## Użycie

Skompilowany program uruchamiamy podając jego ścieżkę w terminalu, podając obowiąkowo ścieżkę (relatywną lub globalną) do folderu w którym znajdują się klatki obrazu, oraz opcjonalnie podając głębokość warty z jaką ma działać algorytm i nazwę pliku wyjściowowego.

```
 ./main frames 7 wynik.csv

 >>>

Wykryty rozmiar: 100x100
Czas calkowity: 0.176129 s
Wykryte punkty przelotu zostały zapisane do pliku: wynik.csv

 ```

## Wymagania

- System operacyjny Linux (np. Ubuntu 22.04)
- Sterowniki CUDA firmy Nvidia w wersji V12.0.140
- Kompilator nvcc