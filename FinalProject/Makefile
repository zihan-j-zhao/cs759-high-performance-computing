INC_DIR := include
OUT_DIR := bin
SRC_DIR := src
STD := c++17

CFLAGS := -I$(INC_DIR)

SRCS := $(SRC_DIR)/main.cpp $(SRC_DIR)/mean_reversion/stat.cpp 
CU_SRCS := $(SRC_DIR)/mean_reversion/stat_cuda.cu $(SRC_DIR)/mean_reversion/kernel.cu
OUT := $(OUT_DIR)/trader

all: seq openmp cuda

seq:
	g++ -std=$(STD) $(SRCS) $(CFLAGS) -o $(OUT)

openmp:
	g++ -std=$(STD) -DX_OPENMP -fopenmp $(SRCS) $(CFLAGS) -o $(OUT)_openmp

cuda:
	nvcc -Xptxas -Xcompiler -DX_CUDA -std $(STD) $(SRCS) $(CU_SRCS) $(CFLAGS) -o $(OUT)_cuda

clean:
	rm -rf ./bin/*