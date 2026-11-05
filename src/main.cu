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
#include <stdexcept>
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

// Jeden przebieg programowania dynamicznego dla nLayers kolejnych warstw:
//   C_0 = prev
//   C_j(p) = max_{q w otoczeniu (2P+1)x(2P+1) punktu p} C_{j-1}(q) + frame_j(p)
//   (poza obrazem C = 0, czyli sasiad spoza obrazu daje 0)
// P = promien ruchu (o ile pikseli obiekt moze sie przesunac miedzy klatkami).
// Kafelek wyjsciowy TILE_X x TILE_Y jest liczony wraz z otoczka R = P*nLayers
// pikseli, ktora z kazda warstwa zmniejsza sie o P. Cale obliczenia odbywaja
// sie w pamieci wspoldzielonej. FINAL=true: zamiast zapisu kosztu liczony jest
// argmax calego obrazu (atomicMax na kluczu 64-bit).
//
// GEN=false: szybka sciezka dla P = 1 (bezposrednio 3x3, jedna synchronizacja
// na warstwe, bufory A/B zamieniane co warstwe).
// GEN=true: dowolne P, maksimum separowalne (najpierw poziomo do bufora B,
// potem pionowo z powrotem do A) - koszt O(P) zamiast O(P^2) na piksel.
//
// max(a_i) + c daje bit w bit ten sam wynik co max(a_i + c), bo zaokraglenie
// float jest monotoniczne, a fmaxf jest dokladne (kolejnosc i rozbicie na
// max poziomy/pionowy nie zmienia wyniku) - wyniki sa identyczne z wersja
// pierwotna.
template <bool FINAL, bool GEN>
__global__ void __launch_bounds__(BLOCK_THREADS)
tbd_fused(const float* __restrict__ prev, const __grid_constant__ SegmentFrames frames, int nLayers, int P,
          int W, int H, float* __restrict__ outCost, unsigned long long* __restrict__ bestKey) {
    extern __shared__ float smem[];
    const int R = GEN ? P * nLayers : nLayers;  // otoczka kafelka
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

    if (!GEN) {
        for (int j = 1; j <= nLayers; ++j) {
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
    } else {
        // Warstwa j: A (warstwa j-1) jest poprawne w obszarze [lo-P, S-lo+P),
        // wynik warstwy j powstaje w obszarze [lo, S-lo), lo = P*j.
        // 1) B = max poziome z A: wiersze [lo-P, SY-lo+P) (wszystkie potrzebne
        //    w kroku pionowym), kolumny [lo, SX-lo).
        // 2) A = max pionowe z B + klatka: obszar [lo, S-lo). Stare A nie jest
        //    juz potrzebne po kroku 1, wiec po warstwie wynik jest znowu w A.
        const int span = 2 * P;
        for (int j = 1; j <= nLayers; ++j) {
            const float* __restrict__ cur = frames.f[j - 1];
            const int lo = P * j;
            const int loIn = lo - P;
            for (int ly = loIn + ty; ly < SY - loIn; ly += BLOCK_Y) {
                const float* aRow = A + ly * SX - P;
                for (int lx = lo + tx; lx < SX - lo; lx += BLOCK_X) {
                    const float* a = aRow + lx;
                    float m = a[0];
                    for (int d = 1; d <= span; ++d) m = fmaxf(m, a[d]);
                    B[ly * SX + lx] = m;
                }
            }
            __syncthreads();
            for (int ly = lo + ty; ly < SY - lo; ly += BLOCK_Y) {
                const int gy = y0 + ly;
                const bool rowIn = (gy >= 0) && (gy < H);
                const float* curRow = cur + (size_t)(rowIn ? gy : 0) * W;
                for (int lx = lo + tx; lx < SX - lo; lx += BLOCK_X) {
                    const int gx = x0 + lx;
                    float v = 0.0f;
                    if (rowIn && gx >= 0 && gx < W) {
                        const float* b = B + (ly - P) * SX + lx;
                        float m = b[0];
                        for (int d = 1; d <= span; ++d) m = fmaxf(m, b[d * SX]);
                        v = m + __ldg(curRow + gx);
                    }
                    A[ly * SX + lx] = v;
                }
            }
            __syncthreads();
        }
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

// ---- wersja referencyjna (pierwotny algorytm), uzywana tylko z --verify ----
// Uogolniona na promien P: wszystkie (2P+1)^2 przesuniecia, dla P = 1 jest to
// dokladnie pierwotne 9 sasiadow.
__global__ void ref_layer(const float* __restrict__ prev, const float* __restrict__ cur,
                          float* __restrict__ out, int W, int H, int P) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;
    float best = -1e30f;
    for (int dy = P; dy >= -P; --dy) {
        for (int dx = P; dx >= -P; --dx) {
            int yy = y + dy;
            int xx = x + dx;
            float val = 0.0f;
            if (yy >= 0 && yy < H && xx >= 0 && xx < W) val = prev[(size_t)yy * W + xx];
            val += cur[(size_t)y * W + x];
            if (val > best) best = val;
        }
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
// poruszajacy sie po przekatnej (pozycje liczy runBench, domyslnie o 1 piksel
// na klatke, z --promien P o P pikseli na klatke). Zapis w formacie
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
class TbdEngine {
public:
    TbdEngine(int W, int H, int L, const BmpLayout& layout, int maxWindows, bool verify, int P)
        : W_(W), H_(H), L_(L), P_(P), layout_(layout), verify_(verify) {
        configureLayers();
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
    }

    ~TbdEngine() {
        for (float* p : ring_) cudaFree(p);
        cudaFree(tmp_[0]);
        cudaFree(tmp_[1]);
        cudaFree(ref_[0]);
        cudaFree(ref_[1]);
        cudaFree(d_raw_);
        cudaFree(d_keys_);
        cudaStreamDestroy(stream_);
    }

    cudaStream_t stream() const { return stream_; }
    unsigned char* rawBuffer() const { return d_raw_; }
    int layersPerPass() const { return maxLayers_; }

    // Konwersja surowej klatki (w d_raw_) do pierscienia jako klatka nr t.
    void pushFrame(long long t) {
        dim3 block(32, 8);
        dim3 grid((W_ + 31) / 32, (H_ + 7) / 8);
        bmp_to_gray<<<grid, block, 0, stream_>>>(d_raw_, layout_.rowBytes, layout_.channels, W_, H_,
                                                 ring_[t % L_]);
        CUDA_CHECK(cudaGetLastError());
    }

    // Okno zaczynajace sie od klatki t0, obejmujace D klatek (t0..t0+D-1).
    // D-1 warstw dzielonych jest na mozliwie rowne przebiegi po co najwyzej
    // maxLayers_ warstw (np. 19 warstw przy limicie 12 -> 10 + 9).
    void runWindow(long long t0, int D, int windowIdx) {
        unsigned long long* key = d_keys_ + windowIdx;
        const int total = D - 1;
        const int passes = std::max(1, (total + maxLayers_ - 1) / maxLayers_);
        const float* src = frame(t0);
        int done = 0, ping = 0, pass = 0;
        do {
            ++pass;
            const int r = (int)((long long)total * pass / passes) - done;
            SegmentFrames sf{};
            for (int k = 0; k < r; ++k) sf.f[k] = frame(t0 + done + 1 + k);
            const bool last = (done + r == total);
            if (last) {
                launchFused<true>(src, sf, r, nullptr, key);
            } else {
                launchFused<false>(src, sf, r, tmp_[ping], nullptr);
                src = tmp_[ping];
                ping ^= 1;
            }
            done += r;
        } while (done < total);

        if (verify_) verifyWindow(t0, D, windowIdx);
    }

    void downloadKeys(std::vector<unsigned long long>& out, int n) {
        out.resize(n);
        if (n > 0)
            CUDA_CHECK(cudaMemcpyAsync(out.data(), d_keys_, (size_t)n * sizeof(unsigned long long),
                                       cudaMemcpyDeviceToHost, stream_));
        CUDA_CHECK(cudaStreamSynchronize(stream_));
    }

    long long verifyErrors() const { return verifyErrors_; }

private:
    // Pamiec wspoldzielona dla otoczki R (dwa bufory kafelka z otoczka).
    static size_t smemBytes(long long R) {
        return 2 * (size_t)(TILE_X + 2 * R) * (size_t)(TILE_Y + 2 * R) * sizeof(float);
    }

    // Wybor liczby warstw na przebieg (maxLayers_) dla promienia P_.
    // P = 1: jak dotychczas MAX_FUSED_LAYERS (39 KB pamieci wspoldzielonej).
    // P > 1: otoczka P*warstwy rosnie szybciej, wiec bierzemy najwiecej warstw
    // (<= MAX_FUSED_LAYERS), dla ktorych kafelek miesci sie w budzecie: polowa
    // pamieci wspoldzielonej SM (>= 2 bloki na SM), ale nie mniej niz dla P = 1
    // i nie wiecej niz limit opt-in na blok. Jesli nawet jedna warstwa nie
    // miesci sie w budzecie, uzywamy jednej warstwy na przebieg do limitu
    // opt-in (mniej blokow na SM); powyzej - blad.
    void configureLayers() {
        int dev = 0, optin = 0, perSM = 0;
        CUDA_CHECK(cudaGetDevice(&dev));
        CUDA_CHECK(cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev));
        CUDA_CHECK(cudaDeviceGetAttribute(&perSM, cudaDevAttrMaxSharedMemoryPerMultiprocessor, dev));
        cudaFuncAttributes fa;
        CUDA_CHECK(cudaFuncGetAttributes(&fa, tbd_fused<true, true>));
        const size_t staticSmem = fa.sharedSizeBytes;  // warpBest (FINAL)
        const size_t limit = (size_t)optin > staticSmem ? (size_t)optin - staticSmem : 0;
        const size_t reservedPerBlock = 1024;  // pamiec zarezerwowana przez system na blok
        const size_t halfSM = (size_t)perSM / 2 > reservedPerBlock + staticSmem
                                  ? (size_t)perSM / 2 - reservedPerBlock - staticSmem
                                  : 0;
        const size_t budget = std::min(limit, std::max(smemBytes(MAX_FUSED_LAYERS), halfSM));

        if (P_ == 1) {
            maxLayers_ = MAX_FUSED_LAYERS;
        } else {
            maxLayers_ = 1;
            while (maxLayers_ < MAX_FUSED_LAYERS &&
                   smemBytes((long long)P_ * (maxLayers_ + 1)) <= budget)
                ++maxLayers_;
        }
        const size_t need = smemBytes((long long)(P_ == 1 ? 1 : P_) * maxLayers_);
        if (need > limit) {
            int maxP = 1;
            while (smemBytes(maxP + 1) <= limit) ++maxP;
            throw std::runtime_error("promien " + std::to_string(P_) +
                                     " jest za duzy dla tego GPU (pamiec wspoldzielona: potrzeba " +
                                     std::to_string(need) + " B, dostepne " + std::to_string(limit) +
                                     " B), maksymalny promien: " + std::to_string(maxP));
        }
        auto allowSmem = [need](auto* kernel) {
            CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)need));
        };
        if (P_ == 1) {
            allowSmem(tbd_fused<true, false>);
            allowSmem(tbd_fused<false, false>);
        } else {
            allowSmem(tbd_fused<true, true>);
            allowSmem(tbd_fused<false, true>);
        }
    }

    template <bool FINAL>
    void launchFused(const float* src, const SegmentFrames& sf, int r, float* out, unsigned long long* key) {
        dim3 block(BLOCK_X, BLOCK_Y);
        dim3 grid((W_ + TILE_X - 1) / TILE_X, (H_ + TILE_Y - 1) / TILE_Y);
        if (P_ == 1)
            tbd_fused<FINAL, false><<<grid, block, smemBytes(r), stream_>>>(src, sf, r, 1, W_, H_, out, key);
        else
            tbd_fused<FINAL, true><<<grid, block, smemBytes((long long)P_ * r), stream_>>>(src, sf, r, P_, W_, H_,
                                                                                          out, key);
        CUDA_CHECK(cudaGetLastError());
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
            ref_layer<<<grid, block, 0, stream_>>>(prev, frame(t0 + d), out, W_, H_, P_);
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

    int W_, H_, L_, P_;
    int maxLayers_ = MAX_FUSED_LAYERS;
    BmpLayout layout_;
    bool verify_;
    std::vector<float*> ring_;
    float* tmp_[2] = {nullptr, nullptr};
    float* ref_[2] = {nullptr, nullptr};
    unsigned char* d_raw_ = nullptr;
    unsigned long long* d_keys_ = nullptr;
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
                     int threads, int P) {
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
    std::cout << "Wykryty rozmiar: " << W << "x" << H << ", klatek: " << N << ", glebokosc: " << L
              << ", promien: " << P << "\n";
    printDevice();

    TbdEngine engine(W, H, L, layout, N, verify, P);

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
    for (int i = 0; i < N; ++i) {
        const unsigned int idx = keyIndex(keys[i]);
        const int row = (int)(idx / (unsigned int)W);
        const int col = (int)(idx % (unsigned int)W);
        vector2 pos{col, H - row};
        fout << pos.x << "," << pos.y << "\n";
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

static int runBench(int W, int H, int N, int L, bool verify, int P, int speed) {
    BmpLayout layout;
    layout.W = W;
    layout.H = H;
    layout.channels = 3;
    layout.rowBytes = ((size_t)W * 3 + 3) & ~(size_t)3;
    std::cout << "Benchmark syntetyczny: " << W << "x" << H << ", klatek: " << N << ", glebokosc: " << L
              << ", promien: " << P << ", predkosc obiektu: " << speed << " px/klatke\n";
    printDevice();

    TbdEngine engine(W, H, L, layout, N, verify, P);
    std::cout << "Warstw na przebieg kernela: " << engine.layersPerPass() << "\n";
    auto objPos = [&](long long t, int& x, int& y) {
        x = (int)((W / 4 + t * speed) % W);
        y = (int)((H / 4 + t * speed) % H);
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
        int ox, oy;
        objPos(k, ox, oy);
        synth_bmp24<<<grid, block, 0, engine.stream()>>>(engine.rawBuffer(), layout.rowBytes, W, H, k, ox, oy);
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
    int hits = 0;
    for (int i = 0; i < N; ++i) {
        const int D = std::min(L, N - i);
        int ox, oy;
        objPos(i + D - 1, ox, oy);
        const unsigned int idx = keyIndex(keys[i]);
        const int row = (int)(idx / (unsigned int)W), col = (int)(idx % (unsigned int)W);
        if (std::abs(col - ox) <= 2 && std::abs(row - oy) <= 2) ++hits;
    }
    std::cout << "Czas GPU: " << ms / 1000.0 << " s, " << ms / N << " ms/klatke -> " << 1000.0 * N / ms
              << " FPS" << (verify ? " (z --verify czas nie jest miarodajny)" : "") << "\n";
    std::cout << "Trafienia w syntetyczny obiekt: " << hits << "/" << N << "\n";
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
                 "wykryte_pozycje.csv)] [--verify] [--threads N] [--promien P]\n"
                 "  ./main --bench <SZER>x<WYS> [klatek (domyslnie 300)] [glebokosc (domyslnie 7)] [--verify]\n"
                 "         [--promien P] [--predkosc V]\n"
                 "  --promien P   maksymalne przesuniecie obiektu miedzy klatkami w pikselach\n"
                 "                (otoczenie (2P+1)x(2P+1), domyslnie 1 = 3x3)\n"
                 "  --predkosc V  predkosc obiektu syntetycznego w --bench (domyslnie = P)\n";
}

int main(int argc, char** argv) {
    std::vector<std::string> pos;
    bool verify = false, bench = false;
    int threads = (int)std::min(8u, std::max(2u, std::thread::hardware_concurrency()));
    int P = 1, speed = -1;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--verify") verify = true;
        else if (a == "--bench") bench = true;
        else if (a == "--threads" && i + 1 < argc) threads = std::max(1, std::atoi(argv[++i]));
        else if (a == "--promien" && i + 1 < argc) P = std::atoi(argv[++i]);
        else if (a == "--predkosc" && i + 1 < argc) speed = std::atoi(argv[++i]);
        else pos.push_back(a);
    }
    if (P < 1 || P > 1000) {
        std::cerr << "Promien musi byc w zakresie 1..1000\n";
        return 1;
    }
    if (speed < 0) speed = P;  // domyslnie obiekt syntetyczny porusza sie o P px/klatke

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
            return runBench(W, H, N, L, verify, P, speed);
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
        return runFolder(pos[0], L, out, verify, threads, P);
    } catch (const std::exception& e) {
        std::cerr << "Blad: " << e.what() << "\n";
        return 1;
    }
}
