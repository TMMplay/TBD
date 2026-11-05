# Kompilacja: make            (domyslnie RTX 5090 / Blackwell, sm_120, CUDA >= 12.8)
#             make ARCH=sm_89 (np. RTX 40xx)   lub   make ARCH=native
NVCC  ?= nvcc
ARCH  ?= sm_120
FLAGS ?= -O3 -std=c++17 -lineinfo

main: src/main.cu
	$(NVCC) $(FLAGS) -arch=$(ARCH) -Xcompiler -pthread $< -o $@

bench: main
	./main --bench 2560x1440 300 7

clean:
	rm -f main

.PHONY: bench clean
