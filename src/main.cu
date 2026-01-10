#include <cuda_runtime.h>
#include <curand.h>
#include <thrust/device_ptr.h>
#include <thrust/extrema.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <fstream>
#include <string>
#include <filesystem>



namespace fs = std::filesystem;
#define CUDA_CHECK(err) do { cudaError_t e = (err); if (e != cudaSuccess) { std::cerr<<"CUDA error: "<<cudaGetErrorString(e)<<" at "<<__FILE__<<":"<<__LINE__<<"\n"; exit(1);} } while(0)
#define CURAND_CHECK(err) do { curandStatus_t e = (err); if (e != CURAND_STATUS_SUCCESS) { std::cerr<<"cuRAND error at "<<__FILE__<<":"<<__LINE__<<" code "<<e<<"\n"; exit(1);} } while(0)



#pragma pack(push,1)
struct BMPHeader {
    uint16_t bfType;
    uint32_t bfSize;
    uint16_t bfReserved1;
    uint16_t bfReserved2;
    uint32_t bfOffBits;
};

struct BMPInfoHeader {
    uint32_t biSize;
    int32_t  biWidth;
    int32_t  biHeight;
    uint16_t biPlanes;
    uint16_t biBitCount;
    uint32_t biCompression;
    uint32_t biSizeImage;
    int32_t  biXPelsPerMeter;
    int32_t  biYPelsPerMeter;
    uint32_t biClrUsed;
    uint32_t biClrImportant;
};

typedef struct vector2 {
    int x;
    int y;
};


#pragma pack(pop)

bool loadBMP(const std::string& path, std::vector<float>& img, int& W, int& H) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;

    BMPHeader h;
    BMPInfoHeader ih;
    f.read((char*)&h, sizeof(h));
    f.read((char*)&ih, sizeof(ih));

    if (h.bfType != 0x4D42) return false;

    W = ih.biWidth;
    H = std::abs(ih.biHeight);
    int channels = ih.biBitCount / 8;
    int row_padded = (W * channels + 3) & (~3);

    img.resize(W * H);

    f.seekg(h.bfOffBits);
    std::vector<unsigned char> row(row_padded);

    for (int y = H - 1; y >= 0; --y) {
        f.read((char*)row.data(), row_padded);
        for (int x = 0; x < W; ++x) {
            float v = 0.0f;
            if (channels == 3) {
                int i = x * 3;
                v = (row[i] + row[i+1] + row[i+2]) / 3.0f;
            } else {
                v = row[x];
            }
            img[y*W + x] = v;
        }
    }
    return true;
}

__device__ inline int idx3(int w, int k, int t, int W, int K) { return (t*W + w)*K + k; }
__device__ inline int idx2(int w, int k, int K) { return w*K + k; }

extern "C" __global__ void compute_best_local(const float* PrevLayerCost, const float* CurrentLayer, float* BestLocalCost, int* BestIndexSlice, int W, int K) {
    int w = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (w >= W || k >= K) return;
    float best = -1e30f;
    int besti = 0;
    const int off[9][2] = {{1,1},{1,0},{1,-1},{0,1},{0,0},{0,-1},{-1,1},{-1,0},{-1,-1}};
    for (int n=0;n<9;++n){
        int ww = w + off[n][0];
        int kk = k + off[n][1];
        float val = 0.0f;
        if (ww>=0 && ww<W && kk>=0 && kk<K) val = PrevLayerCost[idx2(ww,kk,K)];
        val += CurrentLayer[idx2(w,k,K)];
        if (val > best) { best = val; besti = n; }
    }
    BestLocalCost[idx2(w,k,K)] = best;
    BestIndexSlice[idx2(w,k,K)] = besti + 1;
}



int main(int argc, char** argv)
{
    if (argc < 2) {
        std::cerr << "Użycie: ./main <folder_z_bitmapami> <glebokosc warsty (domyslnie = 5)> <plik wyjsciowy (domyslnie wykryte_pozycje.csv)>\n";
        return 1;
    }
    int L = 5; //glebokosc
    if (argc >= 3){
        L = std::stoi(argv[2]);
    }

    std::string output_file = "wykryte_pozycje.csv";
    if (argc >= 4){
        output_file = argv[3];
    }

    std::string image_folder = argv[1];

    namespace fs = std::filesystem;
    std::vector<std::string> images;
    std::vector<vector2> detected_positions;
    for (auto& p : fs::directory_iterator(image_folder)) {
        if (p.path().extension() == ".bmp")
        {
            images.push_back(p.path().string());
        }
    }
    

    if (images.empty()) {
        std::cerr << "Brak bitmap w folderze\n";
        return 1;
    }
    std::sort(images.begin(), images.end());

    int rys9 = images.size();
    // wczytanie pierwszego obrazu -> rozmiar macierzy
    int W, K;
    std::vector<float> tmp;
    if (!loadBMP(images[0], tmp, W, K)) {
        std::cerr << "Nie można wczytać pierwszej bitmapy\n";
        return 1;
    }

    int N = (int)images.size();
    
    

    size_t size3 = (size_t)W * K * N;
    size_t bytes3 = size3 * sizeof(float);

    // wczytanie wszystkich bitmap do X (CPU)
    std::vector<float> h_X(size3);
    for (int t = 0; t < N; ++t) {
        std::vector<float> img;
        int w2, h2;
        loadBMP(images[t], img, w2, h2);
        for (int i = 0; i < W * K; ++i)
            h_X[t * W * K + i] = img[i];
    }

    std::cout << "Wykryty rozmiar: " << W << "x" << K << "\n";

    // CUDA alloc
    float *d_X, *d_Y;
    CUDA_CHECK(cudaMalloc(&d_X, bytes3));
    CUDA_CHECK(cudaMalloc(&d_Y, bytes3));
    CUDA_CHECK(cudaMemcpy(d_X, h_X.data(), bytes3, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_Y, 0, bytes3));

    auto t0 = std::chrono::high_resolution_clock::now();

    for (int okno_idx = 0; okno_idx < rys9; ++okno_idx) {

        size_t sliceBytes = (size_t)W * K * sizeof(float);
        float* d_Prev;
        CUDA_CHECK(cudaMalloc(&d_Prev, sliceBytes));
        CUDA_CHECK(cudaMemcpy(
            d_Prev,
            d_X + (size_t)okno_idx * W * K,
            sliceBytes,
            cudaMemcpyDeviceToDevice));

        int depthSlices = L - 1;
        int* d_bestIndexAll;
        CUDA_CHECK(cudaMalloc(&d_bestIndexAll, (size_t)W * K * depthSlices * sizeof(int)));

        dim3 block(16, 16);
        dim3 grid((K + 15) / 16, (W + 15) / 16);

        for (int depth = 2; depth <= L; ++depth) {
            int t_index = depth + okno_idx - 1;
            if (t_index >= N) break;

            float* d_Current = d_X + (size_t)t_index * W * K;
            float* d_BestLocal;
            CUDA_CHECK(cudaMalloc(&d_BestLocal, sliceBytes));

            compute_best_local<<<grid, block>>>(
                d_Prev,
                d_Current,
                d_BestLocal,
                d_bestIndexAll + (size_t)(depth - 2) * W * K,
                W, K);
            CUDA_CHECK(cudaGetLastError());

            CUDA_CHECK(cudaFree(d_Prev));
            d_Prev = d_BestLocal;
        }

        thrust::device_ptr<float> ptr(d_Prev);
        int flat_idx = thrust::max_element(ptr, ptr + (size_t)W * K) - ptr;
        int w_best = flat_idx / K;
        int k_best = flat_idx % K;

        size_t idY = ((size_t)okno_idx * W + w_best) * K + k_best;
        float one = 1.0f;
        CUDA_CHECK(cudaMemcpy(d_Y + idY, &one, sizeof(float), cudaMemcpyHostToDevice));

        detected_positions.push_back({k_best,W-w_best});

        CUDA_CHECK(cudaFree(d_Prev));
        CUDA_CHECK(cudaFree(d_bestIndexAll));
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(t1 - t0).count();

    std::cout << "Czas calkowity: " << elapsed << " s\n";

    std::ofstream fout(output_file);
    for (size_t i=0;i<detected_positions.size();++i){
        vector2 pos = detected_positions.at(i);
        fout << pos.x << "," << pos.y << "\n";
    }
    
    fout.close();
    std::cout << "Wykryte punkty przelotu zostały zapisane do pliku: " << output_file << "\n";


    CUDA_CHECK(cudaFree(d_X));
    CUDA_CHECK(cudaFree(d_Y));
    return 0;
};
