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
//
// Wynik (plik CSV) jest identyczny z wersja pierwotna dla obrazow kwadratowych.

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
__device__ __forceinline__ float decodeGray(const unsigned char* __restrict__ raw, size_t rowBytes,
                                            int channels, int H, int x, int y) {
    const unsigned char* p = raw + (size_t)(H - 1 - y) * rowBytes + (size_t)x * channels;
    if (channels >= 3) return (float)((int)p[0] + (int)p[1] + (int)p[2]) / 3.0f;
    return (float)p[0];
}

__global__ void bmp_to_gray(const unsigned char* __restrict__ raw, size_t rowBytes, int channels,
                            int W, int H, float* __restrict__ out) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;
    out[(size_t)y * W + x] = decodeGray(raw, rowBytes, channels, H, x, y);
}

// ---- model tla (opcja --tlo) ----
// Dla kazdego piksela utrzymywana jest wykladniczo wazona srednia mu i
// wariancja var jasnosci. Do algorytmu TBD trafia znormalizowane odchylenie
//   z = (g - mu) / max(sqrt(var_3x3), SIGMA_MIN)
// gdzie var_3x3 to wariancja usredniona w oknie 3x3 (estymata z ~2/alfa
// klatek jest zaszumiona; usrednienie zmniejsza jej rozrzut ~3x)
// (dla polaryzacji "ciemny" -z, dla "oba" |z|). Usuwa to statyczne jasne
// elementy sceny (budynki, slonce, gradient nieba), a szum w kazdym miejscu
// obrazu ma podobna skale - prog wykrycia nie zalezy od jasnosci tla.
constexpr float SIGMA_MIN = 1.0f;
enum Polarity { POL_BRIGHT = 0, POL_DARK = 1, POL_BOTH = 2 };

__device__ __forceinline__ void blockReduceAdd2(double a, double b, double* __restrict__ acc) {
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        a += __shfl_down_sync(0xFFFFFFFFu, a, off);
        b += __shfl_down_sync(0xFFFFFFFFu, b, off);
    }
    __shared__ double sa[32], sb[32];
    const int tid = threadIdx.y * blockDim.x + threadIdx.x;
    const int nWarps = (blockDim.x * blockDim.y + 31) / 32;
    if ((tid & 31) == 0) {
        sa[tid >> 5] = a;
        sb[tid >> 5] = b;
    }
    __syncthreads();
    if (tid == 0) {
        for (int i = 1; i < nWarps; ++i) {
            a += sa[i];
            b += sb[i];
        }
        atomicAdd(acc, a);
        atomicAdd(acc + 1, b);
    }
}

// Suma i suma kwadratow jasnosci klatki (do inicjalizacji wariancji tla).
__global__ void frame_sums(const float* __restrict__ g, size_t n, double* __restrict__ acc) {
    double s = 0.0, s2 = 0.0;
    const size_t stride = (size_t)gridDim.x * blockDim.x * blockDim.y;
    for (size_t i = (size_t)blockIdx.x * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x; i < n;
         i += stride) {
        const double v = g[i];
        s += v;
        s2 += v * v;
    }
    blockReduceAdd2(s, s2, acc);
}

// Pierwsza klatka: mu = g, var = wariancja przestrzenna klatki (zawyzona
// wzgledem szumu czasowego, wiec na starcie model jest ostrozny), z = 0.
__global__ void bg_init(float* __restrict__ g, float* __restrict__ mu, float* __restrict__ var,
                        const double* __restrict__ acc, size_t n) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const double mean = acc[0] / (double)n;
    const double v0 = fmax(acc[1] / (double)n - mean * mean, (double)(SIGMA_MIN * SIGMA_MIN));
    mu[i] = g[i];
    var[i] = (float)v0;
    g[i] = 0.0f;
}

__global__ void bg_feature(const unsigned char* __restrict__ raw, size_t rowBytes, int channels, int W, int H,
                           float* __restrict__ mu, const float* __restrict__ varIn, float* __restrict__ varOut,
                           float alpha, int polarity, float* __restrict__ out) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;
    const size_t i = (size_t)y * W + x;
    const float g = decodeGray(raw, rowBytes, channels, H, x, y);
    const float m = mu[i];
    const float v = varIn[i];
    float vs = 0.0f;
#pragma unroll
    for (int dy = -1; dy <= 1; ++dy) {
        const int yy = min(max(y + dy, 0), H - 1);
#pragma unroll
        for (int dx = -1; dx <= 1; ++dx) {
            const int xx = min(max(x + dx, 0), W - 1);
            vs += __ldg(varIn + (size_t)yy * W + xx);
        }
    }
    vs *= (1.0f / 9.0f);
    const float d = g - m;
    float z = d / fmaxf(sqrtf(vs), SIGMA_MIN);
    if (polarity == POL_DARK) z = -z;
    else if (polarity == POL_BOTH) z = fabsf(z);
    out[i] = z;
    // aktualizacja modelu (EMA sredniej i wariancji)
    mu[i] = m + alpha * d;
    varOut[i] = (1.0f - alpha) * (v + alpha * d * d);
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
          float* __restrict__ outCost, unsigned long long* __restrict__ bestKey,
          double* __restrict__ stats) {
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

    // argmax + suma i suma kwadratow mapy kosztu (statystyki do progu wykrycia)
    unsigned long long best = 0ull;
    double sum = 0.0, sum2 = 0.0;
    for (int ly = ty; ly < TILE_Y; ly += BLOCK_Y) {
        const int gy = y0 + R + ly;
        if (gy >= H) break;
        for (int lx = tx; lx < TILE_X; lx += BLOCK_X) {
            const int gx = x0 + R + lx;
            if (gx < W) {
                const float v = A[(ly + R) * SX + lx + R];
                unsigned long long k = makeKey(v, (unsigned int)((size_t)gy * W + gx));
                best = k > best ? k : best;
                sum += v;
                sum2 += (double)v * v;
            }
        }
    }
    blockReduceAdd2(sum, sum2, stats);
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
// poruszajacy sie po przekatnej o 1 piksel na klatke. Zapis w formacie
// surowych danych BMP 24-bit (jak z pliku).
__global__ void synth_bmp24(unsigned char* __restrict__ raw, size_t rowBytes, int W, int H, int t,
                            int objX, int objY) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int r = blockIdx.y * blockDim.y + threadIdx.y;  // wiersz w pliku
    if (x >= W || r >= H) return;
    int y = H - 1 - r;  // wiersz w pamieci po konwersji
    unsigned char v;
    if (x >= objX && x <= objX + 1 && y >= objY && y <= objY + 1) {
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

// ---------------------------------------------------------------------------
// Silnik TBD na GPU: pierscien L klatek + kolejkowanie okien
// ---------------------------------------------------------------------------
struct Options {
    bool verify = false;
    float bgAlpha = 0.0f;          // --tlo: 0 = bez modelu tla (surowa jasnosc)
    int polarity = POL_BRIGHT;     // --polaryzacja
    double threshold = -1.0;       // --prog: < 0 = zawsze zwracaj pozycje
    bool printScore = false;       // --wynik: trzecia kolumna CSV
};

// Wynik okna: pozycja maksimum i "wynik" = (max - srednia) / odchylenie
// standardowe mapy kosztu ostatniej warstwy.
struct WindowResult {
    int col, row;
    float value;
    double score;
};

class TbdEngine {
public:
    TbdEngine(int W, int H, int L, const BmpLayout& layout, int maxWindows, const Options& opt)
        : W_(W), H_(H), L_(L), layout_(layout), verify_(opt.verify), opt_(opt) {
        const size_t frameBytes = (size_t)W * H * sizeof(float);
        ring_.resize(L);
        for (int i = 0; i < L; ++i) CUDA_CHECK(cudaMalloc(&ring_[i], frameBytes));
        CUDA_CHECK(cudaMalloc(&tmp_[0], frameBytes));
        CUDA_CHECK(cudaMalloc(&tmp_[1], frameBytes));
        CUDA_CHECK(cudaMalloc(&d_raw_, layout.dataBytes()));
        CUDA_CHECK(cudaMalloc(&d_keys_, (size_t)std::max(1, maxWindows) * sizeof(unsigned long long)));
        CUDA_CHECK(cudaMemset(d_keys_, 0, (size_t)std::max(1, maxWindows) * sizeof(unsigned long long)));
        CUDA_CHECK(cudaMalloc(&d_stats_, (size_t)std::max(1, maxWindows) * 2 * sizeof(double)));
        CUDA_CHECK(cudaMemset(d_stats_, 0, (size_t)std::max(1, maxWindows) * 2 * sizeof(double)));
        if (opt_.bgAlpha > 0.0f) {
            CUDA_CHECK(cudaMalloc(&d_mu_, frameBytes));
            CUDA_CHECK(cudaMalloc(&d_var_[0], frameBytes));
            CUDA_CHECK(cudaMalloc(&d_var_[1], frameBytes));
            CUDA_CHECK(cudaMalloc(&d_frameAcc_, 2 * sizeof(double)));
            CUDA_CHECK(cudaMemset(d_frameAcc_, 0, 2 * sizeof(double)));
        }
        CUDA_CHECK(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking));
        if (verify_) {
            CUDA_CHECK(cudaMalloc(&ref_[0], frameBytes));
            CUDA_CHECK(cudaMalloc(&ref_[1], frameBytes));
        }
        const int maxSmem = smemBytes(MAX_FUSED_LAYERS);
        CUDA_CHECK(cudaFuncSetAttribute(tbd_fused<true>, cudaFuncAttributeMaxDynamicSharedMemorySize, maxSmem));
        CUDA_CHECK(cudaFuncSetAttribute(tbd_fused<false>, cudaFuncAttributeMaxDynamicSharedMemorySize, maxSmem));
    }

    ~TbdEngine() {
        for (float* p : ring_) cudaFree(p);
        cudaFree(tmp_[0]);
        cudaFree(tmp_[1]);
        cudaFree(ref_[0]);
        cudaFree(ref_[1]);
        cudaFree(d_raw_);
        cudaFree(d_keys_);
        cudaFree(d_stats_);
        cudaFree(d_mu_);
        cudaFree(d_var_[0]);
        cudaFree(d_var_[1]);
        cudaFree(d_frameAcc_);
        cudaStreamDestroy(stream_);
    }

    cudaStream_t stream() const { return stream_; }
    // Nastepna klatka ponownie zainicjalizuje model tla.
    void resetBackground() { bgInitialized_ = false; }
    unsigned char* rawBuffer() const { return d_raw_; }

    // Konwersja surowej klatki (w d_raw_) do pierscienia jako klatka nr t:
    // jasnosc albo (z --tlo) znormalizowane odchylenie od modelu tla.
    void pushFrame(long long t) {
        dim3 block(32, 8);
        dim3 grid((W_ + 31) / 32, (H_ + 7) / 8);
        float* out = ring_[t % L_];
        const size_t n = (size_t)W_ * H_;
        if (opt_.bgAlpha <= 0.0f) {
            bmp_to_gray<<<grid, block, 0, stream_>>>(d_raw_, layout_.rowBytes, layout_.channels, W_, H_, out);
        } else if (!bgInitialized_) {
            bmp_to_gray<<<grid, block, 0, stream_>>>(d_raw_, layout_.rowBytes, layout_.channels, W_, H_, out);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaMemsetAsync(d_frameAcc_, 0, 2 * sizeof(double), stream_));
            frame_sums<<<512, dim3(32, 8), 0, stream_>>>(out, n, d_frameAcc_);
            CUDA_CHECK(cudaGetLastError());
            bg_init<<<(unsigned)((n + 255) / 256), 256, 0, stream_>>>(out, d_mu_, d_var_[varIdx_], d_frameAcc_, n);
            bgInitialized_ = true;
        } else {
            bg_feature<<<grid, block, 0, stream_>>>(d_raw_, layout_.rowBytes, layout_.channels, W_, H_, d_mu_,
                                                    d_var_[varIdx_], d_var_[varIdx_ ^ 1], opt_.bgAlpha,
                                                    opt_.polarity, out);
            varIdx_ ^= 1;
        }
        CUDA_CHECK(cudaGetLastError());
    }

    // Okno zaczynajace sie od klatki t0, obejmujace D klatek (t0..t0+D-1).
    void runWindow(long long t0, int D, int windowIdx) {
        unsigned long long* key = d_keys_ + windowIdx;
        double* stats = d_stats_ + 2 * (size_t)windowIdx;
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
            if (last) {
                tbd_fused<true><<<grid, block, smemBytes(r), stream_>>>(src, sf, r, W_, H_, nullptr, key, stats);
            } else {
                tbd_fused<false><<<grid, block, smemBytes(r), stream_>>>(src, sf, r, W_, H_, tmp_[ping], nullptr,
                                                                    nullptr);
                src = tmp_[ping];
                ping ^= 1;
            }
            CUDA_CHECK(cudaGetLastError());
            done += r;
        } while (done < total);

        if (verify_) verifyWindow(t0, D, windowIdx);
    }

    // Pobiera wyniki wszystkich okien (jedyna synchronizacja z GPU).
    std::vector<WindowResult> downloadResults(int n) {
        std::vector<unsigned long long> keys(n);
        std::vector<double> stats(2 * (size_t)n);
        if (n > 0) {
            CUDA_CHECK(cudaMemcpyAsync(keys.data(), d_keys_, (size_t)n * sizeof(unsigned long long),
                                       cudaMemcpyDeviceToHost, stream_));
            CUDA_CHECK(cudaMemcpyAsync(stats.data(), d_stats_, 2 * (size_t)n * sizeof(double),
                                       cudaMemcpyDeviceToHost, stream_));
        }
        CUDA_CHECK(cudaStreamSynchronize(stream_));
        std::vector<WindowResult> res(n);
        const double cnt = (double)W_ * H_;
        for (int i = 0; i < n; ++i) {
            const unsigned int idx = keyIndex(keys[i]);
            res[i].col = (int)(idx % (unsigned int)W_);
            res[i].row = (int)(idx / (unsigned int)W_);
            res[i].value = keyValue(keys[i]);
            const double mean = stats[2 * i] / cnt;
            const double sd = std::sqrt(std::max(0.0, stats[2 * i + 1] / cnt - mean * mean));
            res[i].score = sd > 1e-12 ? (res[i].value - mean) / sd : 0.0;
        }
        return res;
    }

    long long verifyErrors() const { return verifyErrors_; }

private:
    static int smemBytes(int R) {
        return 2 * (TILE_X + 2 * R) * (TILE_Y + 2 * R) * (int)sizeof(float);
    }
    const float* frame(long long t) const { return ring_[t % L_]; }

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
        CUDA_CHECK(cudaMemcpyAsync(&key, d_keys_ + windowIdx, sizeof(key), cudaMemcpyDeviceToHost, stream_));
        CUDA_CHECK(cudaStreamSynchronize(stream_));
        if (keyIndex(key) != refIdx || keyValue(key) != refVal) {
            if (verifyErrors_ < 10)
                std::cerr << "[verify] okno " << windowIdx << ": fused idx=" << keyIndex(key)
                          << " val=" << keyValue(key) << " | ref idx=" << refIdx << " val=" << refVal << "\n";
            ++verifyErrors_;
        }
    }

    int W_, H_, L_;
    BmpLayout layout_;
    bool verify_;
    Options opt_;
    bool bgInitialized_ = false;
    std::vector<float*> ring_;
    float* tmp_[2] = {nullptr, nullptr};
    float* ref_[2] = {nullptr, nullptr};
    unsigned char* d_raw_ = nullptr;
    unsigned long long* d_keys_ = nullptr;
    double* d_stats_ = nullptr;
    float* d_mu_ = nullptr;
    float* d_var_[2] = {nullptr, nullptr};  // wariancja tla (ping-pong, bo czytane jest okno 3x3)
    int varIdx_ = 0;
    double* d_frameAcc_ = nullptr;
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

static bool isDetected(const WindowResult& r, const Options& opt) {
    return opt.threshold < 0.0 || r.score >= opt.threshold;
}

// Podsumowanie rozkladu wyniku - pomaga dobrac prog --prog (np. uruchamiajac
// program na nagraniu bez obiektu i biorac prog troche powyzej maksimum).
static void printScoreSummary(const std::vector<WindowResult>& res, const Options& opt) {
    if (res.empty()) return;
    std::vector<double> sc;
    for (auto& r : res) sc.push_back(r.score);
    std::sort(sc.begin(), sc.end());
    auto q = [&](double p) { return sc[(size_t)std::min<double>(sc.size() - 1, std::floor(p * (sc.size() - 1)))]; };
    std::cout << "Wynik (max-srednia)/odch.std: min " << sc.front() << ", mediana " << q(0.5) << ", p95 " << q(0.95)
              << ", max " << sc.back() << "\n";
    if (opt.threshold >= 0.0) {
        size_t det = 0;
        for (auto& r : res) det += isDetected(r, opt);
        std::cout << "Prog " << opt.threshold << ": wykryto obiekt w " << det << "/" << res.size() << " klatkach\n";
    }
}

static void writeCsv(const std::string& path, const std::vector<WindowResult>& res, int H, const Options& opt) {
    std::ofstream fout(path);
    for (auto& r : res) {
        vector2 pos{r.col, H - r.row};
        if (!isDetected(r, opt)) pos = {-1, -1};
        fout << pos.x << "," << pos.y;
        if (opt.printScore) fout << "," << r.score;
        fout << "\n";
    }
}

static int runFolder(const std::string& folder, int L, const std::string& outputFile, const Options& opt,
                     int threads) {
    const bool verify = opt.verify;
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
    printDevice();

    TbdEngine engine(W, H, L, layout, N, opt);

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
    const std::vector<WindowResult> res = engine.downloadResults(N);
    auto t1 = std::chrono::high_resolution_clock::now();
    const double elapsed = std::chrono::duration<double>(t1 - t0).count();

    writeCsv(outputFile, res, H, opt);

    std::cout << "Czas calkowity: " << elapsed << " s (" << N / elapsed << " FPS, z wczytywaniem plikow)\n";
    printScoreSummary(res, opt);
    if (verify) {
        std::cout << "Weryfikacja z algorytmem referencyjnym: "
                  << (engine.verifyErrors() == 0 ? "OK" : "BLEDY: " + std::to_string(engine.verifyErrors()))
                  << "\n";
    }
    std::cout << "Wykryte punkty przelotu zostaly zapisane do pliku: " << outputFile << "\n";
    return verify && engine.verifyErrors() ? 2 : 0;
}

static int runBench(int W, int H, int N, int L, const Options& opt) {
    const bool verify = opt.verify;
    BmpLayout layout;
    layout.W = W;
    layout.H = H;
    layout.channels = 3;
    layout.rowBytes = ((size_t)W * 3 + 3) & ~(size_t)3;
    std::cout << "Benchmark syntetyczny: " << W << "x" << H << ", klatek: " << N << ", glebokosc: " << L
              << "\n";
    printDevice();

    TbdEngine engine(W, H, L, layout, N, opt);
    auto objPos = [&](long long t, int& x, int& y) {
        x = (int)((W / 4 + t) % W);
        y = (int)((H / 4 + t) % H);
    };
    dim3 block(32, 8);
    dim3 grid((W + 31) / 32, (H + 7) / 8);

    // Rozgrzewka (JIT, alokacje wewnetrzne sterownika).
    synth_bmp24<<<grid, block, 0, engine.stream()>>>(engine.rawBuffer(), layout.rowBytes, W, H, 0, 0, 0);
    engine.pushFrame(0);
    CUDA_CHECK(cudaStreamSynchronize(engine.stream()));
    engine.resetBackground();

    cudaEvent_t e0, e1;
    CUDA_CHECK(cudaEventCreate(&e0));
    CUDA_CHECK(cudaEventCreate(&e1));
    CUDA_CHECK(cudaEventRecord(e0, engine.stream()));
    for (int k = 0; k < N; ++k) {
        int ox, oy;
        objPos(k, ox, oy);
        synth_bmp24<<<grid, block, 0, engine.stream()>>>(engine.rawBuffer(), layout.rowBytes, W, H, k, ox, oy);
        engine.pushFrame(k);
        if (k >= L - 1) engine.runWindow(k - L + 1, L, k - L + 1);
    }
    for (int i = std::max(0, N - L + 1); i < N; ++i) engine.runWindow(i, N - i, i);
    CUDA_CHECK(cudaEventRecord(e1, engine.stream()));
    const std::vector<WindowResult> res = engine.downloadResults(N);
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, e0, e1));

    // Sprawdzenie trafien: argmax ostatniej warstwy = pozycja obiektu w
    // ostatniej klatce okna (i + D - 1).
    int hits = 0;
    for (int i = 0; i < N; ++i) {
        const int D = std::min(L, N - i);
        int ox, oy;
        objPos(i + D - 1, ox, oy);
        if (isDetected(res[i], opt) && std::abs(res[i].col - ox) <= 2 && std::abs(res[i].row - oy) <= 2) ++hits;
    }
    std::cout << "Czas GPU: " << ms / 1000.0 << " s, " << ms / N << " ms/klatke -> " << 1000.0 * N / ms
              << " FPS" << (verify ? " (z --verify czas nie jest miarodajny)" : "") << "\n";
    std::cout << "Trafienia w syntetyczny obiekt: " << hits << "/" << N << "\n";
    printScoreSummary(res, opt);
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
                 "wykryte_pozycje.csv)] [opcje]\n"
                 "  ./main --bench <SZER>x<WYS> [klatek (domyslnie 300)] [glebokosc (domyslnie 7)] [opcje]\n"
                 "Opcje:\n"
                 "  --tlo A               model tla (srednia/wariancja EMA, wspolczynnik A, np. 0.05)\n"
                 "  --polaryzacja P       jasny | ciemny | oba (z --tlo; domyslnie jasny)\n"
                 "  --prog K              prog wykrycia: (max-srednia)/odch.std >= K, inaczej -1,-1\n"
                 "  --wynik               dopisz wynik jako trzecia kolumne CSV\n"
                 "  --verify              porownanie z algorytmem referencyjnym\n"
                 "  --threads N           liczba watkow wczytujacych pliki\n";
}

int main(int argc, char** argv) {
    std::vector<std::string> pos;
    Options opt;
    bool bench = false, polaritySet = false;
    int threads = (int)std::min(8u, std::max(2u, std::thread::hardware_concurrency()));
    try {
        for (int i = 1; i < argc; ++i) {
            std::string a = argv[i];
            const bool hasArg = i + 1 < argc;
            if (a == "--verify") opt.verify = true;
            else if (a == "--bench") bench = true;
            else if (a == "--wynik") opt.printScore = true;
            else if (a == "--threads" && hasArg) threads = std::max(1, std::atoi(argv[++i]));
            else if (a == "--tlo" && hasArg) opt.bgAlpha = std::stof(argv[++i]);
            else if (a == "--prog" && hasArg) opt.threshold = std::stod(argv[++i]);
            else if (a == "--polaryzacja" && hasArg) {
                const std::string v = argv[++i];
                polaritySet = true;
                if (v == "jasny") opt.polarity = POL_BRIGHT;
                else if (v == "ciemny") opt.polarity = POL_DARK;
                else if (v == "oba") opt.polarity = POL_BOTH;
                else {
                    std::cerr << "Nieznana polaryzacja: " << v << "\n";
                    return 1;
                }
            } else if (a.size() > 2 && a.compare(0, 2, "--") == 0) {
                std::cerr << "Nieznana opcja: " << a << "\n";
                usage();
                return 1;
            } else pos.push_back(a);
        }
        if (opt.bgAlpha < 0.0f || opt.bgAlpha >= 1.0f) {
            std::cerr << "--tlo musi byc w przedziale (0, 1)\n";
            return 1;
        }
        if (polaritySet && opt.bgAlpha <= 0.0f) {
            std::cerr << "--polaryzacja wymaga --tlo\n";
            return 1;
        }

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
            return runBench(W, H, N, L, opt);
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
        return runFolder(pos[0], L, out, opt, threads);
    } catch (const std::exception& e) {
        std::cerr << "Blad: " << e.what() << "\n";
        return 1;
    }
}
