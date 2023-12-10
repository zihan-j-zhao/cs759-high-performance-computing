INC_DIR := include
OUT_DIR := bin
SRC_DIR := src
STD := c++17

CFLAGS := -I$(INC_DIR)

SRCS := $(SRC_DIR)/main.cpp $(SRC_DIR)/mean_reversion/stat.cpp 
OUT := $(OUT_DIR)/trader

all:
	g++ -std=$(STD) $(SRCS) $(CFLAGS) -o $(OUT)

openmp:
	g++ -std=$(STD) -DX_OPENMP -fopenmp $(SRCS) $(CFLAGS) -o $(OUT)_openmp