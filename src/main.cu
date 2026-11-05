// TBD (Track-Before-Detect) - wersja strumieniowa zoptymalizowana pod duze
// rozdzielczosci (2K i wiecej) oraz glebokosc >= 7.
//
// Najwazniejsze zmiany wzgledem pierwotnej wersji:
//  * brak cudaMalloc/cudaFree/thrust/cudaMemcpy synchronicznych w petli -
//    wszystkie bufory sa alokowane raz, a cala praca na klatke jest kolejkowana
//    asynchronicznie w jednym strumieniu CUDA (zero synchronizacji na klatke),
//  * wszystkie warstwy programowania dynamicznego (L-1 warstw) liczone sa w
//    jednym kernelu na kafelkach w pamieci wspoldzielonej (fuzja warstw),
//    a argmax (max_element) jest zlaczony z ostatnia warstwa (atomicMax na
//    upakowanym kluczu 64-bit) - brak posrednich buforow w pamieci globalnej,
//  * nie jest zapisywana nieuzywana tablica indeksow (BestIndexSlice) ani
//    nieuzywana objetosc d_Y,
//  * klatki nie sa juz wszystkie trzymane w RAM/VRAM: na GPU jest tylko bufor
//    pierscieniowy L klatek, a pliki BMP sa wczytywane rownolegle przez kilka
//    watkow do pamieci "pinned" i konwertowane do skali szarosci na GPU,
//  * poprawiona obsluga obrazow nieprostokatnych (W != H) oraz przepelnienie
//    int przy indeksowaniu dlugich sekwencji.
//  * opcjonalnie wiele obiektow na klatke (--obiekty K, --nms R): K najlepszych
//    maksimow lokalnych mapy kosztu z tlumieniem niemaksymalnym w promieniu R.
//
// Wynik (plik CSV) jest identyczny z wersja pierwotna dla obrazow kwadratowych
// (dla domyslnego K = 1).

#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/extrema.h>
#include <thrust/execution_policy.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace fs = std::filesystem;

#define CUDA_CHECK(err)                                                              \
    do {                                                                             \
        cudaError_t e_ = (err);                                                      \
        if (e_ != cudaSuccess) {                                                     \
            std::cerr << "CUDA error: " << cudaGetErrorString(e_) << " at " << __FILE__ \
                      << ":" << __LINE__ << "\n";                                    \
            std::exit(1);                                                            \
        }                                                                            \
    } while (0)

// ---------------------------------------------------------------------------
// Parametry kerneli
// ---------------------------------------------------------------------------
constexpr int TILE_X = 64;   // szerokosc kafelka wyjsciowego
constexpr int TILE_Y = 32;   // wysokosc kafelka wyjsciowego
constexpr int BLOCK_X = 32;  // = rozmiar warpa
constexpr int BLOCK_Y = 8;
constexpr int BLOCK_THREADS = BLOCK_X * BLOCK_Y;
// Maksymalna liczba warstw liczonych w jednym przebiegu kernela. Wieksze
// glebokosci sa dzielone na kilka przebiegow (posrednie koszty w VRAM).
constexpr int MAX_FUSED_LAYERS = 12;

struct SegmentFrames {
    const float* f[MAX_FUSED_LAYERS];
};

// ---------------------------------------------------------------------------
// BMP
// ---------------------------------------------------------------------------
#pragma pack(push, 1)
struct BMPHeader {
    uint16_t bfType;
    uint32_t bfSize;
    uint16_t bfReserved1;
    uint16_t bfReserved2;
    uint32_t bfOffBits;
};

struct BMPInfoHeader {
    uint32_t biSize;
    int32_t biWidth;
    int32_t biHeight;
    uint16_t biPlanes;
    uint16_t biBitCount;
    uint32_t biCompression;
    uint32_t biSizeImage;
    int32_t biXPelsPerMeter;
    int32_t biYPelsPerMeter;
    uint32_t biClrUsed;
    uint32_t biClrImportant;
};
#pragma pack(pop)

struct vector2 {
    int x;
    int y;
};

struct BmpLayout {
    int W = 0, H = 0;
    int channels = 0;       // bajty na piksel (1, 3 lub 4)
    size_t rowBytes = 0;    // dlugosc wiersza z wyrownaniem do 4 bajtow
    uint32_t offBits = 0;   // przesuniecie danych pikseli w pliku
    size_t dataBytes() const { return rowBytes * (size_t)H; }
};

static bool readBmpLayout(std::FILE* f, BmpLayout& out, std::string& err) {
    BMPHeader h;
    BMPInfoHeader ih;
    if (std::fread(&h, sizeof(h), 1, f) != 1 || std::fread(&ih, sizeof(ih), 1, f) != 1) {
        err = "za krotki naglowek";
        return false;
    }
    if (h.bfType != 0x4D42) {
        err = "to nie jest plik BMP";
        return false;
    }
    if (ih.biCompression != 0 && ih.biCompression != 3) {
        err = "skompresowane BMP nie sa obslugiwane";
        return false;
    }
    if (ih.biBitCount != 8 && ih.biBitCount != 24 && ih.biBitCount != 32) {
        err = "obslugiwane sa tylko BMP 8/24/32 bit";
        return false;
    }
    out.W = ih.biWidth;
    out.H = std::abs(ih.biHeight);
    out.channels = ih.biBitCount / 8;
    out.rowBytes = ((size_t)out.W * out.channels + 3) & ~(size_t)3;
    out.offBits = h.bfOffBits;
    if (out.W <= 0 || out.H <= 0) {
        err = "niepoprawny rozmiar obrazu";
        return false;
    }
    return true;
}

// Wczytuje surowe dane pikseli pliku BMP (bez dekodowania) do bufora dst.
static bool loadBmpRaw(const std::string& path, const BmpLayout& expected, unsigned char* dst,
                       std::string& err) {
    std::FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) {
        err = "nie mozna otworzyc pliku";
        return false;
    }
    std::setvbuf(f, nullptr, _IONBF, 0);
    BmpLayout l;
    bool ok = readBmpLayout(f, l, err);
    if (ok && (l.W != expected.W || l.H != expected.H || l.channels != expected.channels)) {
        err = "inny rozmiar/format niz pierwsza klatka";
        ok = false;
    }
    if (ok && std::fseek(f, (long)l.offBits, SEEK_SET) != 0) {
        err = "blad seek";
        ok = false;
    }
    if (ok && std::fread(dst, 1, l.dataBytes(), f) != l.dataBytes()) {
        err = "plik jest uciety";
        ok = false;
    }
    std::fclose(f);
    return ok;
}

// ---------------------------------------------------------------------------
// Kernele
// ---------------------------------------------------------------------------

// Klucz do argmax: starsze 32 bity = wartosc float zamieniona na liczbe
// zachowujaca porzadek, mlodsze 32 bity = ~indeks (przy remisie wygrywa
// mniejszy indeks - tak samo jak thrust::max_element).
__device__ __forceinline__ unsigned long long makeKey(float v, unsigned int idx) {
    unsigned int u = __float_as_uint(v);
    u = (u & 0x80000000u) ? ~u : (u | 0x80000000u);
    return ((unsigned long long)u << 32) | (unsigned long long)(0xFFFFFFFFu - idx);
}

static float keyValue(unsigned long long key) {
    unsigned int u = (unsigned int)(key >> 32);
    u = (u & 0x80000000u) ? (u & 0x7FFFFFFFu) : ~u;
    float v;
    std::memcpy(&v, &u, sizeof(v));
    return v;
}
static unsigned int keyIndex(unsigned long long key) {
    return 0xFFFFFFFFu - (unsigned int)(key & 0xFFFFFFFFull);
}

// Konwersja surowych danych BMP -> skala szarosci (float).
// Odwzorowanie wierszy identyczne z pierwotnym loadBMP (wiersz z pliku r ->
// wiersz H-1-r), wartosci: (B+G+R)/3.0f dla 24/32 bit, bajt dla 8 bit.
__global__ void bmp_to_gray(const unsigned char* __restrict__ raw, size_t rowBytes, int channels,
                            int W, int H, float* __restrict__ out) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;
    const unsigned char* p = raw + (size_t)(H - 1 - y) * rowBytes + (size_t)x * channels;
    float v;
    if (channels >= 3)
        v = (float)((int)p[0] + (int)p[1] + (int)p[2]) / 3.0f;
    else
        v = (float)p[0];
    out[(size_t)y * W + x] = v;
}

// Jeden przebieg programowania dynamicznego dla R kolejnych warstw:
//   C_0 = prev
//   C_j(p) = max_{q in 3x3(p)} C_{j-1}(q) + frame_j(p)     (poza obrazem C = 0)
// Kafelek wyjsciowy TILE_X x TILE_Y jest liczony wraz z otoczka R pikseli,
// ktora z kazda warstwa zmniejsza sie o 1. Cale obliczenia odbywaja sie w
// pamieci wspoldzielonej. FINAL=true: zamiast zapisu kosztu liczony jest
// argmax calego obrazu (atomicMax na kluczu 64-bit).
//
// max(a_i) + c daje bit w bit ten sam wynik co max(a_i + c), bo zaokraglenie
// float jest monotoniczne - wyniki sa identyczne z wersja pierwotna.
template <bool FINAL>
__global__ void __launch_bounds__(BLOCK_THREADS)
tbd_fused(const float* __restrict__ prev, const __grid_constant__ SegmentFrames frames, int R, int W, int H,
          float* __restrict__ outCost, unsigned long long* __restrict__ bestKey) {
    extern __shared__ float smem[];
    const int SX = TILE_X + 2 * R;
    const int SY = TILE_Y + 2 * R;
    float* A = smem;
    float* B = smem + SX * SY;

    const int x0 = blockIdx.x * TILE_X - R;
    const int y0 = blockIdx.y * TILE_Y - R;
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    // Wczytanie kosztu poczatkowego (z otoczka); poza obrazem = 0.
    for (int ly = ty; ly < SY; ly += BLOCK_Y) {
        const int gy = y0 + ly;
        const bool rowIn = (gy >= 0) && (gy < H);
        const float* src = prev + (size_t)(rowIn ? gy : 0) * W;
        for (int lx = tx; lx < SX; lx += BLOCK_X) {
            const int gx = x0 + lx;
            A[ly * SX + lx] = (rowIn && gx >= 0 && gx < W) ? __ldg(src + gx) : 0.0f;
        }
    }
    __syncthreads();

    for (int j = 1; j <= R; ++j) {
        const float* __restrict__ cur = frames.f[j - 1];
        for (int ly = j + ty; ly < SY - j; ly += BLOCK_Y) {
            const int gy = y0 + ly;
            const bool rowIn = (gy >= 0) && (gy < H);
            const float* curRow = cur + (size_t)(rowIn ? gy : 0) * W;
            for (int lx = j + tx; lx < SX - j; lx += BLOCK_X) {
                const int gx = x0 + lx;
                float v = 0.0f;
                if (rowIn && gx >= 0 && gx < W) {
                    const float* a = A + (ly - 1) * SX + lx;
                    float m = fmaxf(fmaxf(a[-1], a[0]), a[1]);
                    a += SX;
                    m = fmaxf(m, fmaxf(fmaxf(a[-1], a[0]), a[1]));
                    a += SX;
                    m = fmaxf(m, fmaxf(fmaxf(a[-1], a[0]), a[1]));
                    v = m + __ldg(curRow + gx);
                }
                B[ly * SX + lx] = v;
            }
        }
        __syncthreads();
        float* t = A;
        A = B;
        B = t;
    }

    // A zawiera teraz koszt koncowy; poprawny jest obszar [R, R+TILE).
    if (!FINAL) {
        for (int ly = ty; ly < TILE_Y; ly += BLOCK_Y) {
            const int gy = y0 + R + ly;
            if (gy >= H) break;
            for (int lx = tx; lx < TILE_X; lx += BLOCK_X) {
                const int gx = x0 + R + lx;
                if (gx < W) outCost[(size_t)gy * W + gx] = A[(ly + R) * SX + lx + R];
            }
        }
        return;
    }

    unsigned long long best = 0ull;
    for (int ly = ty; ly < TILE_Y; ly += BLOCK_Y) {
        const int gy = y0 + R + ly;
        if (gy >= H) break;
        for (int lx = tx; lx < TILE_X; lx += BLOCK_X) {
            const int gx = x0 + R + lx;
            if (gx < W) {
                unsigned long long k =
                    makeKey(A[(ly + R) * SX + lx + R], (unsigned int)((size_t)gy * W + gx));
                best = k > best ? k : best;
            }
        }
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        unsigned long long o = __shfl_down_sync(0xFFFFFFFFu, best, off);
        best = o > best ? o : best;
    }
    __shared__ unsigned long long warpBest[BLOCK_Y];
    if (tx == 0) warpBest[ty] = best;
    __syncthreads();
    if (ty == 0 && tx == 0) {
        unsigned long long b = warpBest[0];
        for (int i = 1; i < BLOCK_Y; ++i) b = warpBest[i] > b ? warpBest[i] : b;
        atomicMax(bestKey, b);
    }
}

// ---------------------------------------------------------------------------
// Wiele obiektow na klatke (--obiekty K > 1): maksima lokalne z tlumieniem
// niemaksymalnym (NMS, promien R) + wybor K najlepszych.
//
// Punkt p jest maksimum lokalnym, gdy jego klucz (makeKey) jest najwiekszy w
// oknie (2R+1)x(2R+1) wokol p (obcietym do obrazu). Klucze sa unikalne, wiec
// remisy wartosci rozstrzyga mniejszy indeks - tak samo jak dla K == 1.
// Dwa maksima lokalne leza zawsze dalej niz R od siebie (w metryce
// Czebyszewa), wiec w kafelku NMS_TX x NMS_TY jest ich co najwyzej
// ceil(NMS_TX/(R+1)) * ceil(NMS_TY/(R+1)) - stad dokladne ograniczenie
// rozmiaru bufora kandydatow (bez mozliwosci przepelnienia).
// ---------------------------------------------------------------------------
constexpr int NMS_TX = BLOCK_X;  // szerokosc kafelka NMS (1 kolumna na watek)
constexpr int NMS_TY = 32;       // wysokosc kafelka NMS
constexpr int NMS_PER_THREAD = NMS_TY / BLOCK_Y;
constexpr int MAX_OBJECTS = 256;  // gorny limit --obiekty
constexpr int SELECT_THREADS = 1024;

// Kernel 1: test maksimum lokalnego (rozdzielnie: max w poziomie do pamieci
// wspoldzielonej z otoczka R wierszy, potem max w pionie) i wybor co najwyzej
// K najlepszych maksimow kafelka. Wynik dopisywany do bufora kandydatow.
__global__ void __launch_bounds__(BLOCK_THREADS)
nms_tile_peaks(const float* __restrict__ cost, int W, int H, int R, int K,
               unsigned long long* __restrict__ cand, unsigned int* __restrict__ candCount,
               unsigned int candCap) {
    extern __shared__ unsigned long long hmax[];  // (NMS_TY + 2R) x NMS_TX
    __shared__ unsigned long long warpBest[BLOCK_Y];
    __shared__ unsigned int sTotal, sBase, sOff;

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tid = ty * BLOCK_X + tx;
    const int x0 = blockIdx.x * NMS_TX;
    const int y0 = blockIdx.y * NMS_TY;
    const int gx = x0 + tx;
    const int SYH = NMS_TY + 2 * R;
    if (tid == 0) {
        sTotal = 0;
        sOff = 0;
    }

    // 1) maksimum klucza w poziomie (okno 2R+1) dla wierszy kafelka z otoczka
    for (int ly = ty; ly < SYH; ly += BLOCK_Y) {
        const int gy = y0 - R + ly;
        unsigned long long m = 0ull;
        if (gy >= 0 && gy < H && gx < W) {
            const size_t rowOff = (size_t)gy * W;
            const int xa = max(gx - R, 0);
            const int xb = min(gx + R, W - 1);
            for (int x = xa; x <= xb; ++x) {
                const unsigned long long k = makeKey(__ldg(cost + rowOff + x), (unsigned int)(rowOff + x));
                m = k > m ? k : m;
            }
        }
        hmax[ly * NMS_TX + tx] = m;
    }
    __syncthreads();

    // 2) maksimum w pionie; punkt jest maksimum lokalnym, gdy max okna == jego klucz
    unsigned long long pk[NMS_PER_THREAD];
    unsigned int myCount = 0;
#pragma unroll
    for (int i = 0; i < NMS_PER_THREAD; ++i) {
        const int ly = ty + i * BLOCK_Y;
        const int gy = y0 + ly;
        pk[i] = 0ull;
        if (gy < H && gx < W) {
            const size_t idx = (size_t)gy * W + gx;
            const unsigned long long own = makeKey(__ldg(cost + idx), (unsigned int)idx);
            unsigned long long m = 0ull;
            for (int r = ly; r <= ly + 2 * R; ++r) {
                const unsigned long long k = hmax[r * NMS_TX + tx];
                m = k > m ? k : m;
            }
            if (m == own) {
                pk[i] = own;
                ++myCount;
            }
        }
    }
    if (myCount) atomicAdd(&sTotal, myCount);
    __syncthreads();

    // 3) zapis min(K, liczba maksimow) najlepszych maksimow kafelka
    const unsigned int total = sTotal;
    const unsigned int nOut = min(total, (unsigned int)K);
    if (nOut == 0) return;  // warunek jednakowy dla calego bloku
    if (tid == 0) sBase = atomicAdd(candCount, nOut);
    __syncthreads();
    const unsigned int base = sBase;

    if (total <= (unsigned int)K) {
        // wszystkie maksima kafelka (kolejnosc nieistotna)
        if (myCount) {
            unsigned int o = base + atomicAdd(&sOff, myCount);
#pragma unroll
            for (int i = 0; i < NMS_PER_THREAD; ++i) {
                if (pk[i]) {
                    if (o < candCap) cand[o] = pk[i];
                    ++o;
                }
            }
        }
        return;
    }

    // wiecej maksimow niz K: K rund wyboru maksimum w bloku
    for (int r = 0; r < K; ++r) {
        unsigned long long m = pk[0];
#pragma unroll
        for (int i = 1; i < NMS_PER_THREAD; ++i) m = pk[i] > m ? pk[i] : m;
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            const unsigned long long o = __shfl_xor_sync(0xFFFFFFFFu, m, off);
            m = o > m ? o : m;
        }
        if (tx == 0) warpBest[ty] = m;
        __syncthreads();
        unsigned long long b = warpBest[0];
        for (int i = 1; i < BLOCK_Y; ++i) b = warpBest[i] > b ? warpBest[i] : b;
        if (tid == 0 && base + r < candCap) cand[base + r] = b;
#pragma unroll
        for (int i = 0; i < NMS_PER_THREAD; ++i)
            if (pk[i] == b) pk[i] = 0ull;
        __syncthreads();
    }
}

// Kernel 2 (jeden blok): K najwiekszych kluczy z bufora kandydatow,
// posortowanych malejaco -> out[0..K-1] (0 = brak). Wybor progu (K-ty
// najwiekszy klucz) przez radix select po 8 bitow, potem sortowanie przez
// zliczanie rang. Na koniec zeruje licznik kandydatow dla nastepnego okna.
__global__ void __launch_bounds__(SELECT_THREADS)
nms_select_topk(const unsigned long long* __restrict__ cand, unsigned int* __restrict__ candCount,
                unsigned int candCap, int K, unsigned long long* __restrict__ out) {
    __shared__ unsigned int hist[256];
    __shared__ unsigned long long sTop[MAX_OBJECTS];
    __shared__ unsigned long long sPrefix, sMask;
    __shared__ unsigned int sNeed, sNTop;

    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const unsigned int n = min(*candCount, candCap);

    unsigned long long threshold = 1ull;  // n <= K: wszystkie kandydaty (klucze != 0)
    if (n > (unsigned int)K) {
        if (tid == 0) {
            sPrefix = 0ull;
            sMask = 0ull;
            sNeed = (unsigned int)K;
        }
        for (int pass = 0; pass < 8; ++pass) {
            const int shift = 56 - 8 * pass;
            for (int b = tid; b < 256; b += blockDim.x) hist[b] = 0;
            __syncthreads();
            const unsigned long long prefix = sPrefix, mask = sMask;
            for (unsigned int base = 0; base < n; base += blockDim.x) {  // petla jednakowa w warpie
                const unsigned int i = base + tid;
                unsigned int bin = 256;
                if (i < n) {
                    const unsigned long long k = cand[i];
                    if ((k & mask) == prefix) bin = (unsigned int)(k >> shift) & 255u;
                }
                const unsigned int peers = __match_any_sync(0xFFFFFFFFu, bin);
                if (bin < 256 && lane == __ffs(peers) - 1) atomicAdd(&hist[bin], (unsigned int)__popc(peers));
            }
            __syncthreads();
            if (tid == 0) {
                const unsigned int need = sNeed;
                unsigned int cum = 0;
                int b = 255;
                for (; b > 0; --b) {
                    if (cum + hist[b] >= need) break;
                    cum += hist[b];
                }
                sPrefix = prefix | ((unsigned long long)b << shift);
                sMask = mask | (0xFFull << shift);
                sNeed = need - cum;
            }
            __syncthreads();
        }
        threshold = sPrefix;  // dokladnie K-ty najwiekszy klucz
    }

    if (tid == 0) sNTop = 0;
    __syncthreads();
    for (unsigned int i = tid; i < n; i += blockDim.x) {
        const unsigned long long k = cand[i];
        if (k >= threshold) {
            const unsigned int p = atomicAdd(&sNTop, 1u);
            if (p < (unsigned int)MAX_OBJECTS) sTop[p] = k;
        }
    }
    __syncthreads();
    const unsigned int m = min(sNTop, (unsigned int)K);
    for (unsigned int i = tid; i < m; i += blockDim.x) {
        const unsigned long long k = sTop[i];
        unsigned int rank = 0;
        for (unsigned int j = 0; j < m; ++j) rank += sTop[j] > k ? 1u : 0u;
        out[rank] = k;
    }
    for (unsigned int r = m + tid; r < (unsigned int)K; r += blockDim.x) out[r] = 0ull;
    if (tid == 0) *candCount = 0;  // wszystkie watki odczytaly n przed pierwszym __syncthreads
}

// ---- wersja referencyjna (pierwotny algorytm), uzywana tylko z --verify ----
__global__ void ref_layer(const float* __restrict__ prev, const float* __restrict__ cur,
                          float* __restrict__ out, int W, int H) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;
    const int off[9][2] = {{1, 1}, {1, 0}, {1, -1}, {0, 1}, {0, 0}, {0, -1}, {-1, 1}, {-1, 0}, {-1, -1}};
    float best = -1e30f;
    for (int n = 0; n < 9; ++n) {
        int yy = y + off[n][0];
        int xx = x + off[n][1];
        float val = 0.0f;
        if (yy >= 0 && yy < H && xx >= 0 && xx < W) val = prev[(size_t)yy * W + xx];
        val += cur[(size_t)y * W + x];
        if (val > best) best = val;
    }
    out[(size_t)y * W + x] = best;
}

// ---- generator syntetycznych klatek (tryb --bench) ----
__device__ __forceinline__ uint32_t hash3(uint32_t x, uint32_t y, uint32_t t) {
    uint32_t h = x * 0x8da6b343u ^ y * 0xd8163841u ^ t * 0xcb1ab31fu;
    h ^= h >> 16;
    h *= 0x7feb352du;
    h ^= h >> 15;
    h *= 0x846ca68bu;
    h ^= h >> 16;
    return h;
}

// Szum (0..127, 1% "soli" o jasnosci 200) + obiekt 2x2 o jasnosci 255
// poruszajacy sie po przekatnej o 1 piksel na klatke (opcjonalnie drugi
// obiekt (objX2, objY2); wspolrzedne ujemne = brak). Zapis w formacie
// surowych danych BMP 24-bit (jak z pliku).
__global__ void synth_bmp24(unsigned char* __restrict__ raw, size_t rowBytes, int W, int H, int t,
                            int objX, int objY, int objX2 = -10, int objY2 = -10) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int r = blockIdx.y * blockDim.y + threadIdx.y;  // wiersz w pliku
    if (x >= W || r >= H) return;
    int y = H - 1 - r;  // wiersz w pamieci po konwersji
    unsigned char v;
    if ((x >= objX && x <= objX + 1 && y >= objY && y <= objY + 1) ||
        (x >= objX2 && x <= objX2 + 1 && y >= objY2 && y <= objY2 + 1)) {
        v = 255;
    } else {
        uint32_t h = hash3((uint32_t)x, (uint32_t)y, (uint32_t)t);
        v = (h % 100u == 0u) ? 200 : (unsigned char)((h >> 8) & 127u);
    }
    unsigned char* p = raw + (size_t)r * rowBytes + (size_t)x * 3;
    p[0] = v;
    p[1] = v;
    p[2] = v;
}

// ---- referencja CPU dla --obiekty K > 1 (tylko z --verify) ----
static unsigned long long makeKeyHost(float v, unsigned int idx) {
    unsigned int u;
    std::memcpy(&u, &v, sizeof(u));
    u = (u & 0x80000000u) ? ~u : (u | 0x80000000u);
    return ((unsigned long long)u << 32) | (unsigned long long)(0xFFFFFFFFu - idx);
}

// Brute force: wszystkie maksima lokalne w oknie (2R+1)x(2R+1), K najlepszych
// malejaco, dopelnione zerami do K.
static void hostTopKNms(const std::vector<float>& c, int W, int H, int K, int R,
                        std::vector<unsigned long long>& out) {
    std::vector<unsigned long long> peaks;
    auto key = [&](int x, int y) {
        const size_t i = (size_t)y * W + x;
        return makeKeyHost(c[i], (unsigned int)i);
    };
    auto isMax = [&](int x, int y, int r) {
        const unsigned long long k = key(x, y);
        for (int yy = std::max(0, y - r); yy <= std::min(H - 1, y + r); ++yy)
            for (int xx = std::max(0, x - r); xx <= std::min(W - 1, x + r); ++xx)
                if (key(xx, yy) > k) return false;
        return true;
    };
    for (int y = 0; y < H; ++y)
        for (int x = 0; x < W; ++x)
            if (isMax(x, y, std::min(R, 1)) && isMax(x, y, R)) peaks.push_back(key(x, y));
    const size_t m = std::min(peaks.size(), (size_t)K);
    std::partial_sort(peaks.begin(), peaks.begin() + m, peaks.end(), std::greater<unsigned long long>());
    out.assign(peaks.begin(), peaks.begin() + m);
    out.resize(K, 0ull);
}

// ---------------------------------------------------------------------------
// Silnik TBD na GPU: pierscien L klatek + kolejkowanie okien
// ---------------------------------------------------------------------------
class TbdEngine {
public:
    // K - liczba obiektow na klatke (--obiekty), nmsR - promien NMS (--nms).
    TbdEngine(int W, int H, int L, const BmpLayout& layout, int maxWindows, bool verify, int K = 1,
              int nmsR = 0)
        : W_(W), H_(H), L_(L), layout_(layout), verify_(verify), K_(K), nmsR_(nmsR) {
        const size_t frameBytes = (size_t)W * H * sizeof(float);
        ring_.resize(L);
        for (int i = 0; i < L; ++i) CUDA_CHECK(cudaMalloc(&ring_[i], frameBytes));
        CUDA_CHECK(cudaMalloc(&tmp_[0], frameBytes));
        CUDA_CHECK(cudaMalloc(&tmp_[1], frameBytes));
        CUDA_CHECK(cudaMalloc(&d_raw_, layout.dataBytes()));
        CUDA_CHECK(cudaMalloc(&d_keys_, (size_t)std::max(1, maxWindows) * sizeof(unsigned long long)));
        CUDA_CHECK(cudaMemset(d_keys_, 0, (size_t)std::max(1, maxWindows) * sizeof(unsigned long long)));
        CUDA_CHECK(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking));
        if (verify_) {
            CUDA_CHECK(cudaMalloc(&ref_[0], frameBytes));
            CUDA_CHECK(cudaMalloc(&ref_[1], frameBytes));
        }
        const int maxSmem = smemBytes(MAX_FUSED_LAYERS);
        CUDA_CHECK(cudaFuncSetAttribute(tbd_fused<true>, cudaFuncAttributeMaxDynamicSharedMemorySize, maxSmem));
        CUDA_CHECK(cudaFuncSetAttribute(tbd_fused<false>, cudaFuncAttributeMaxDynamicSharedMemorySize, maxSmem));
        if (K_ > 1) initPeaks(maxWindows);
    }

    ~TbdEngine() {
        for (float* p : ring_) cudaFree(p);
        cudaFree(tmp_[0]);
        cudaFree(tmp_[1]);
        cudaFree(ref_[0]);
        cudaFree(ref_[1]);
        cudaFree(d_raw_);
        cudaFree(d_keys_);
        cudaFree(d_peaks_);
        cudaFree(d_cand_);
        cudaFree(d_candCount_);
        cudaStreamDestroy(stream_);
    }

    cudaStream_t stream() const { return stream_; }
    unsigned char* rawBuffer() const { return d_raw_; }

    // Konwersja surowej klatki (w d_raw_) do pierscienia jako klatka nr t.
    void pushFrame(long long t) {
        dim3 block(32, 8);
        dim3 grid((W_ + 31) / 32, (H_ + 7) / 8);
        bmp_to_gray<<<grid, block, 0, stream_>>>(d_raw_, layout_.rowBytes, layout_.channels, W_, H_,
                                                 ring_[t % L_]);
        CUDA_CHECK(cudaGetLastError());
    }

    // Okno zaczynajace sie od klatki t0, obejmujace D klatek (t0..t0+D-1).
    void runWindow(long long t0, int D, int windowIdx) {
        unsigned long long* key = d_keys_ + windowIdx;
        const int total = D - 1;
        const float* src = frame(t0);
        int done = 0, ping = 0;
        dim3 block(BLOCK_X, BLOCK_Y);
        dim3 grid((W_ + TILE_X - 1) / TILE_X, (H_ + TILE_Y - 1) / TILE_Y);
        do {
            const int r = std::min(MAX_FUSED_LAYERS, total - done);
            SegmentFrames sf{};
            for (int k = 0; k < r; ++k) sf.f[k] = frame(t0 + done + 1 + k);
            const bool last = (done + r == total);
            if (last && K_ == 1) {
                tbd_fused<true><<<grid, block, smemBytes(r), stream_>>>(src, sf, r, W_, H_, nullptr, key);
            } else {
                tbd_fused<false><<<grid, block, smemBytes(r), stream_>>>(src, sf, r, W_, H_, tmp_[ping], nullptr);
                src = tmp_[ping];
                ping ^= 1;
            }
            CUDA_CHECK(cudaGetLastError());
            done += r;
        } while (done < total);
        // K > 1: koszt koncowy jest w src (bufor tymczasowy) -> NMS + top-K.
        if (K_ > 1) findPeaks(src, windowIdx);

        if (verify_) verifyWindow(t0, D, windowIdx);
    }

    // Wyniki n okien: K == 1 - n kluczy argmax; K > 1 - n*K kluczy maksimow
    // (okno i: out[i*K .. i*K+K-1], malejaco, 0 = brak).
    void downloadKeys(std::vector<unsigned long long>& out, int n) {
        const size_t cnt = (size_t)n * K_;
        out.resize(cnt);
        if (cnt > 0)
            CUDA_CHECK(cudaMemcpyAsync(out.data(), K_ == 1 ? d_keys_ : d_peaks_,
                                       cnt * sizeof(unsigned long long), cudaMemcpyDeviceToHost, stream_));
        CUDA_CHECK(cudaStreamSynchronize(stream_));
    }

    long long verifyErrors() const { return verifyErrors_; }

private:
    static int smemBytes(int R) {
        return 2 * (TILE_X + 2 * R) * (TILE_Y + 2 * R) * (int)sizeof(float);
    }
    const float* frame(long long t) const { return ring_[t % L_]; }

    void initPeaks(int maxWindows) {
        const int tilesX = (W_ + NMS_TX - 1) / NMS_TX;
        const int tilesY = (H_ + NMS_TY - 1) / NMS_TY;
        // maksymalna liczba maksimow lokalnych w kafelku (patrz opis nms_tile_peaks)
        const long long perTile = (long long)((NMS_TX + nmsR_) / (nmsR_ + 1)) * ((NMS_TY + nmsR_) / (nmsR_ + 1));
        const long long cap = (long long)tilesX * tilesY * std::min<long long>(K_, perTile);
        if (cap > 0xFFFFFFFFll) {
            std::cerr << "Za duzy bufor kandydatow (zmniejsz --obiekty lub zwieksz --nms)\n";
            std::exit(1);
        }
        candCap_ = (unsigned int)cap;
        nmsSmem_ = (size_t)(NMS_TY + 2 * nmsR_) * NMS_TX * sizeof(unsigned long long);
        int dev = 0, maxOptin = 0;
        CUDA_CHECK(cudaGetDevice(&dev));
        CUDA_CHECK(cudaDeviceGetAttribute(&maxOptin, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev));
        const size_t staticSmem = 128;  // warpBest + zmienne __shared__ kernela
        if (nmsSmem_ + staticSmem > (size_t)maxOptin) {
            const int maxR = (int)(((size_t)maxOptin - staticSmem) / (NMS_TX * sizeof(unsigned long long)) - NMS_TY) / 2;
            std::cerr << "Za duzy promien --nms " << nmsR_ << " (maksymalnie " << maxR << " na tym GPU)\n";
            std::exit(1);
        }
        CUDA_CHECK(cudaFuncSetAttribute(nms_tile_peaks, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)nmsSmem_));
        const size_t peaksBytes = (size_t)std::max(1, maxWindows) * K_ * sizeof(unsigned long long);
        CUDA_CHECK(cudaMalloc(&d_peaks_, peaksBytes));
        CUDA_CHECK(cudaMemset(d_peaks_, 0, peaksBytes));
        CUDA_CHECK(cudaMalloc(&d_cand_, (size_t)std::max(1u, candCap_) * sizeof(unsigned long long)));
        CUDA_CHECK(cudaMalloc(&d_candCount_, sizeof(unsigned int)));
        CUDA_CHECK(cudaMemset(d_candCount_, 0, sizeof(unsigned int)));
    }

    // Maksima lokalne (NMS) mapy kosztu -> K najlepszych do d_peaks_[okno*K + k].
    // Asynchronicznie w strumieniu; licznik kandydatow zeruje nms_select_topk.
    void findPeaks(const float* cost, int windowIdx) {
        dim3 block(BLOCK_X, BLOCK_Y);
        dim3 grid((W_ + NMS_TX - 1) / NMS_TX, (H_ + NMS_TY - 1) / NMS_TY);
        nms_tile_peaks<<<grid, block, nmsSmem_, stream_>>>(cost, W_, H_, nmsR_, K_, d_cand_, d_candCount_,
                                                            candCap_);
        CUDA_CHECK(cudaGetLastError());
        nms_select_topk<<<1, SELECT_THREADS, 0, stream_>>>(d_cand_, d_candCount_, candCap_, K_,
                                                            d_peaks_ + (size_t)windowIdx * K_);
        CUDA_CHECK(cudaGetLastError());
    }

    // Pierwotny algorytm warstwa po warstwie + thrust::max_element; porownanie.
    void verifyWindow(long long t0, int D, int windowIdx) {
        dim3 block(16, 16);
        dim3 grid((W_ + 15) / 16, (H_ + 15) / 16);
        const size_t n = (size_t)W_ * H_;
        const float* prev = frame(t0);
        for (int d = 1; d < D; ++d) {
            float* out = ref_[d & 1];
            ref_layer<<<grid, block, 0, stream_>>>(prev, frame(t0 + d), out, W_, H_);
            CUDA_CHECK(cudaGetLastError());
            prev = out;
        }
        thrust::device_ptr<const float> p(prev);
        const size_t refIdx = thrust::max_element(thrust::cuda::par.on(stream_), p, p + n) - p;
        float refVal = 0.0f;
        unsigned long long key = 0;
        CUDA_CHECK(cudaMemcpyAsync(&refVal, prev + refIdx, sizeof(float), cudaMemcpyDeviceToHost, stream_));
        // K > 1: porownywany jest najlepszy z wykrytych punktow (zawsze = argmax).
        const unsigned long long* dKey = K_ == 1 ? d_keys_ + windowIdx : d_peaks_ + (size_t)windowIdx * K_;
        CUDA_CHECK(cudaMemcpyAsync(&key, dKey, sizeof(key), cudaMemcpyDeviceToHost, stream_));
        CUDA_CHECK(cudaStreamSynchronize(stream_));
        if (keyIndex(key) != refIdx || keyValue(key) != refVal) {
            if (verifyErrors_ < 10)
                std::cerr << "[verify] okno " << windowIdx << ": fused idx=" << keyIndex(key)
                          << " val=" << keyValue(key) << " | ref idx=" << refIdx << " val=" << refVal << "\n";
            ++verifyErrors_;
        }
        if (K_ > 1) {
            // Pelne porownanie K maksimow z referencja CPU (brute force NMS).
            std::vector<float> hMap(n);
            std::vector<unsigned long long> gpu(K_), ref;
            CUDA_CHECK(cudaMemcpyAsync(hMap.data(), prev, n * sizeof(float), cudaMemcpyDeviceToHost, stream_));
            CUDA_CHECK(cudaMemcpyAsync(gpu.data(), dKey, (size_t)K_ * sizeof(unsigned long long),
                                       cudaMemcpyDeviceToHost, stream_));
            CUDA_CHECK(cudaStreamSynchronize(stream_));
            hostTopKNms(hMap, W_, H_, K_, nmsR_, ref);
            if (gpu != ref) {
                if (verifyErrors_ < 10) {
                    int k = 0;
                    while (gpu[k] == ref[k]) ++k;
                    std::cerr << "[verify] okno " << windowIdx << ": maksimum nr " << k + 1 << " fused idx="
                              << keyIndex(gpu[k]) << " val=" << keyValue(gpu[k]) << " | ref idx=" << keyIndex(ref[k])
                              << " val=" << keyValue(ref[k]) << "\n";
                }
                ++verifyErrors_;
            }
        }
    }

    int W_, H_, L_;
    BmpLayout layout_;
    bool verify_;
    std::vector<float*> ring_;
    float* tmp_[2] = {nullptr, nullptr};
    float* ref_[2] = {nullptr, nullptr};
    unsigned char* d_raw_ = nullptr;
    unsigned long long* d_keys_ = nullptr;
    // --obiekty K > 1
    int K_ = 1;
    int nmsR_ = 0;
    unsigned long long* d_peaks_ = nullptr;  // [okno * K + k]
    unsigned long long* d_cand_ = nullptr;   // kandydaci z kafelkow (bufor roboczy)
    unsigned int* d_candCount_ = nullptr;
    unsigned int candCap_ = 0;
    size_t nmsSmem_ = 0;
    cudaStream_t stream_ = nullptr;
    long long verifyErrors_ = 0;
};

// ---------------------------------------------------------------------------
// Rownolegle wczytywanie plikow do pamieci pinned
// ---------------------------------------------------------------------------
class FramePrefetcher {
public:
    FramePrefetcher(const std::vector<std::string>& files, const BmpLayout& layout, int threads)
        : files_(files), layout_(layout) {
        const int nSlots = std::max(2, threads * 2);
        slots_.resize(nSlots);
        for (int i = 0; i < nSlots; ++i) {
            CUDA_CHECK(cudaHostAlloc(&slots_[i].host, layout.dataBytes(), cudaHostAllocDefault));
            CUDA_CHECK(cudaEventCreateWithFlags(&slots_[i].ev, cudaEventDisableTiming));
            slots_[i].nextFrame = i;
        }
        for (int i = 0; i < threads; ++i) workers_.emplace_back([this] { worker(); });
    }

    ~FramePrefetcher() {
        {
            std::lock_guard<std::mutex> lk(m_);
            abort_ = true;
        }
        cv_.notify_all();
        for (auto& t : workers_) t.join();
        for (auto& s : slots_) {
            if (s.needSync) cudaEventSynchronize(s.ev);  // kopia H2D moze byc w toku
            cudaFreeHost(s.host);
            cudaEventDestroy(s.ev);
        }
    }

    // Czeka az klatka k bedzie w pamieci; zwraca wskaznik do danych.
    const unsigned char* acquire(int k) {
        Slot& s = slots_[k % slots_.size()];
        std::unique_lock<std::mutex> lk(m_);
        cv_.wait(lk, [&] { return s.state == Slot::Ready && s.nextFrame == k; });
        if (!s.ok) {
            std::cerr << "Blad wczytywania " << files_[k] << ": " << s.err << "\n";
            std::exit(1);
        }
        return s.host;
    }

    // Zwalnia slot klatki k po zakolejkowaniu kopii H2D; kopia musi zakonczyc
    // sie (zdarzenie ev) zanim slot zostanie nadpisany.
    void release(int k, cudaStream_t stream) {
        Slot& s = slots_[k % slots_.size()];
        CUDA_CHECK(cudaEventRecord(s.ev, stream));
        {
            std::lock_guard<std::mutex> lk(m_);
            s.state = Slot::Free;
            s.needSync = true;
            s.nextFrame = k + (int)slots_.size();
        }
        cv_.notify_all();
    }

private:
    struct Slot {
        enum State { Free, Loading, Ready } state = Free;
        int nextFrame = 0;
        bool needSync = false;
        bool ok = true;
        std::string err;
        unsigned char* host = nullptr;
        cudaEvent_t ev = nullptr;
    };

    void worker() {
        for (;;) {
            const int k = next_.fetch_add(1);
            if (k >= (int)files_.size()) return;
            Slot& s = slots_[k % slots_.size()];
            bool needSync;
            {
                std::unique_lock<std::mutex> lk(m_);
                cv_.wait(lk, [&] { return abort_ || (s.state == Slot::Free && s.nextFrame == k); });
                if (abort_) return;
                s.state = Slot::Loading;
                needSync = s.needSync;
            }
            if (needSync) CUDA_CHECK(cudaEventSynchronize(s.ev));
            std::string err;
            const bool ok = loadBmpRaw(files_[k], layout_, s.host, err);
            {
                std::lock_guard<std::mutex> lk(m_);
                s.ok = ok;
                s.err = err;
                s.state = Slot::Ready;
            }
            cv_.notify_all();
        }
    }

    const std::vector<std::string>& files_;
    BmpLayout layout_;
    std::vector<Slot> slots_;
    std::vector<std::thread> workers_;
    std::atomic<int> next_{0};
    std::mutex m_;
    std::condition_variable cv_;
    bool abort_ = false;
};

// ---------------------------------------------------------------------------
// Tryby uruchomienia
// ---------------------------------------------------------------------------
static void printDevice() {
    int dev = 0;
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDevice(&dev));
    CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
    std::cout << "GPU: " << prop.name << " (sm_" << prop.major << prop.minor << ", "
              << prop.multiProcessorCount << " SM)\n";
}

static int runFolder(const std::string& folder, int L, const std::string& outputFile, bool verify,
                     int threads, int K, int nmsR) {
    std::vector<std::string> images;
    for (auto& p : fs::directory_iterator(folder))
        if (p.path().extension() == ".bmp") images.push_back(p.path().string());
    if (images.empty()) {
        std::cerr << "Brak bitmap w folderze\n";
        return 1;
    }
    std::sort(images.begin(), images.end());
    const int N = (int)images.size();

    BmpLayout layout;
    {
        std::string err;
        std::FILE* f = std::fopen(images[0].c_str(), "rb");
        if (!f || !readBmpLayout(f, layout, err)) {
            std::cerr << "Nie mozna wczytac pierwszej bitmapy " << err << "\n";
            if (f) std::fclose(f);
            return 1;
        }
        std::fclose(f);
    }
    const int W = layout.W, H = layout.H;
    std::cout << "Wykryty rozmiar: " << W << "x" << H << ", klatek: " << N << ", glebokosc: " << L << "\n";
    if (K > 1) std::cout << "Obiektow na klatke: " << K << ", promien NMS: " << nmsR << "\n";
    printDevice();

    TbdEngine engine(W, H, L, layout, N, verify, K, nmsR);

    auto t0 = std::chrono::high_resolution_clock::now();
    {
        FramePrefetcher prefetch(images, layout, threads);
        for (int k = 0; k < N; ++k) {
            const unsigned char* host = prefetch.acquire(k);
            CUDA_CHECK(cudaMemcpyAsync(engine.rawBuffer(), host, layout.dataBytes(), cudaMemcpyHostToDevice,
                                       engine.stream()));
            prefetch.release(k, engine.stream());
            engine.pushFrame(k);
            if (k >= L - 1) engine.runWindow(k - L + 1, L, k - L + 1);
        }
        // Koncowka sekwencji: okna krotsze niz L (jak w wersji pierwotnej).
        for (int i = std::max(0, N - L + 1); i < N; ++i) engine.runWindow(i, N - i, i);
    }
    std::vector<unsigned long long> keys;
    engine.downloadKeys(keys, N);
    auto t1 = std::chrono::high_resolution_clock::now();
    const double elapsed = std::chrono::duration<double>(t1 - t0).count();

    std::ofstream fout(outputFile);
    if (K == 1) {
        for (int i = 0; i < N; ++i) {
            const unsigned int idx = keyIndex(keys[i]);
            const int row = (int)(idx / (unsigned int)W);
            const int col = (int)(idx % (unsigned int)W);
            vector2 pos{col, H - row};
            fout << pos.x << "," << pos.y << "\n";
        }
    } else {
        // wiersz: x1,y1,v1,x2,y2,v2,... (malejaco wg kosztu v; mniej trojek,
        // gdy obraz ma mniej maksimow lokalnych niz K)
        for (int i = 0; i < N; ++i) {
            for (int k = 0; k < K; ++k) {
                const unsigned long long key = keys[(size_t)i * K + k];
                if (key == 0) break;
                const unsigned int idx = keyIndex(key);
                const int row = (int)(idx / (unsigned int)W);
                const int col = (int)(idx % (unsigned int)W);
                fout << (k ? "," : "") << col << "," << H - row << "," << keyValue(key);
            }
            fout << "\n";
        }
    }
    fout.close();

    std::cout << "Czas calkowity: " << elapsed << " s (" << N / elapsed << " FPS, z wczytywaniem plikow)\n";
    if (verify) {
        std::cout << "Weryfikacja z algorytmem referencyjnym: "
                  << (engine.verifyErrors() == 0 ? "OK" : "BLEDY: " + std::to_string(engine.verifyErrors()))
                  << "\n";
    }
    std::cout << "Wykryte punkty przelotu zostaly zapisane do pliku: " << outputFile << "\n";
    return verify && engine.verifyErrors() ? 2 : 0;
}

static int runBench(int W, int H, int N, int L, bool verify, int K, int nmsR) {
    BmpLayout layout;
    layout.W = W;
    layout.H = H;
    layout.channels = 3;
    layout.rowBytes = ((size_t)W * 3 + 3) & ~(size_t)3;
    std::cout << "Benchmark syntetyczny: " << W << "x" << H << ", klatek: " << N << ", glebokosc: " << L
              << "\n";
    if (K > 1)
        std::cout << "Obiektow na klatke: " << K << ", promien NMS: " << nmsR
                  << " (drugi obiekt syntetyczny przesuniety o W/2)\n";
    printDevice();

    TbdEngine engine(W, H, L, layout, N, verify, K, nmsR);
    auto objPos = [&](long long t, int& x, int& y) {
        x = (int)((W / 4 + t) % W);
        y = (int)((H / 4 + t) % H);
    };
    // Drugi obiekt (tylko dla K > 1): rownolegly tor przesuniety o W/2.
    auto objPos2 = [&](long long t, int& x, int& y) {
        objPos(t, x, y);
        x = (x + W / 2) % W;
    };
    dim3 block(32, 8);
    dim3 grid((W + 31) / 32, (H + 7) / 8);

    // Rozgrzewka (JIT, alokacje wewnetrzne sterownika).
    synth_bmp24<<<grid, block, 0, engine.stream()>>>(engine.rawBuffer(), layout.rowBytes, W, H, 0, 0, 0);
    engine.pushFrame(0);
    CUDA_CHECK(cudaStreamSynchronize(engine.stream()));

    cudaEvent_t e0, e1;
    CUDA_CHECK(cudaEventCreate(&e0));
    CUDA_CHECK(cudaEventCreate(&e1));
    CUDA_CHECK(cudaEventRecord(e0, engine.stream()));
    for (int k = 0; k < N; ++k) {
        int ox, oy, ox2 = -10, oy2 = -10;
        objPos(k, ox, oy);
        if (K > 1) objPos2(k, ox2, oy2);
        synth_bmp24<<<grid, block, 0, engine.stream()>>>(engine.rawBuffer(), layout.rowBytes, W, H, k, ox, oy,
                                                          ox2, oy2);
        engine.pushFrame(k);
        if (k >= L - 1) engine.runWindow(k - L + 1, L, k - L + 1);
    }
    for (int i = std::max(0, N - L + 1); i < N; ++i) engine.runWindow(i, N - i, i);
    CUDA_CHECK(cudaEventRecord(e1, engine.stream()));
    std::vector<unsigned long long> keys;
    engine.downloadKeys(keys, N);
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, e0, e1));

    // Sprawdzenie trafien: argmax ostatniej warstwy = pozycja obiektu w
    // ostatniej klatce okna (i + D - 1).
    // K > 1: obiekt jest trafiony, gdy ktorykolwiek z K punktow okna lezy obok.
    int hits = 0, hits2 = 0;
    auto near = [&](int i, int ox, int oy) {
        for (int k = 0; k < K; ++k) {
            const unsigned long long key = keys[(size_t)i * K + k];
            if (key == 0) break;
            const unsigned int idx = keyIndex(key);
            const int row = (int)(idx / (unsigned int)W), col = (int)(idx % (unsigned int)W);
            if (std::abs(col - ox) <= 2 && std::abs(row - oy) <= 2) return true;
        }
        return false;
    };
    for (int i = 0; i < N; ++i) {
        const int D = std::min(L, N - i);
        int ox, oy;
        objPos(i + D - 1, ox, oy);
        if (near(i, ox, oy)) ++hits;
        if (K > 1) {
            objPos2(i + D - 1, ox, oy);
            if (near(i, ox, oy)) ++hits2;
        }
    }
    std::cout << "Czas GPU: " << ms / 1000.0 << " s, " << ms / N << " ms/klatke -> " << 1000.0 * N / ms
              << " FPS" << (verify ? " (z --verify czas nie jest miarodajny)" : "") << "\n";
    if (K == 1)
        std::cout << "Trafienia w syntetyczny obiekt: " << hits << "/" << N << "\n";
    else
        std::cout << "Trafienia w syntetyczne obiekty: " << hits << "/" << N << " oraz " << hits2 << "/" << N
                  << "\n";
    if (verify) {
        std::cout << "Weryfikacja z algorytmem referencyjnym: "
                  << (engine.verifyErrors() == 0 ? "OK" : "BLEDY: " + std::to_string(engine.verifyErrors()))
                  << "\n";
    }
    cudaEventDestroy(e0);
    cudaEventDestroy(e1);
    return verify && engine.verifyErrors() ? 2 : 0;
}

static void usage() {
    std::cerr << "Uzycie:\n"
                 "  ./main <folder_z_bitmapami> [glebokosc (domyslnie 5)] [plik_wyjsciowy (domyslnie "
                 "wykryte_pozycje.csv)] [--verify] [--threads N]\n"
                 "         [--obiekty K (domyslnie 1, max "
              << MAX_OBJECTS
              << ")] [--nms R (domyslnie 8)]\n"
                 "  ./main --bench <SZER>x<WYS> [klatek (domyslnie 300)] [glebokosc (domyslnie 7)] [--verify]\n"
                 "         [--obiekty K] [--nms R]\n";
}

int main(int argc, char** argv) {
    std::vector<std::string> pos;
    bool verify = false, bench = false;
    int threads = (int)std::min(8u, std::max(2u, std::thread::hardware_concurrency()));
    int K = 1, nmsR = 8;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--verify") verify = true;
        else if (a == "--bench") bench = true;
        else if (a == "--threads" && i + 1 < argc) threads = std::max(1, std::atoi(argv[++i]));
        else if (a == "--obiekty" && i + 1 < argc) K = std::atoi(argv[++i]);
        else if (a == "--nms" && i + 1 < argc) nmsR = std::atoi(argv[++i]);
        else pos.push_back(a);
    }
    if (K < 1 || K > MAX_OBJECTS || nmsR < 0 || nmsR > 100000) {
        std::cerr << "--obiekty musi byc w zakresie 1.." << MAX_OBJECTS << ", --nms w zakresie 0..100000\n";
        return 1;
    }

    try {
        if (bench) {
            if (pos.empty()) {
                usage();
                return 1;
            }
            int W = 0, H = 0;
            if (std::sscanf(pos[0].c_str(), "%dx%d", &W, &H) != 2 || W <= 0 || H <= 0) {
                usage();
                return 1;
            }
            const int N = pos.size() >= 2 ? std::stoi(pos[1]) : 300;
            const int L = pos.size() >= 3 ? std::stoi(pos[2]) : 7;
            if (N < 1 || L < 1) {
                usage();
                return 1;
            }
            return runBench(W, H, N, L, verify, K, nmsR);
        }
        if (pos.empty()) {
            usage();
            return 1;
        }
        const int L = pos.size() >= 2 ? std::stoi(pos[1]) : 5;
        if (L < 1) {
            std::cerr << "Glebokosc musi byc >= 1\n";
            return 1;
        }
        const std::string out = pos.size() >= 3 ? pos[2] : "wykryte_pozycje.csv";
        return runFolder(pos[0], L, out, verify, threads, K, nmsR);
    } catch (const std::exception& e) {
        std::cerr << "Blad: " << e.what() << "\n";
        return 1;
    }
}
