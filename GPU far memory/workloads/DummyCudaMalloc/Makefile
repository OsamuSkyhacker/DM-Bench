# Simple Makefile for gpu_mem_hog

NVCC        ?= nvcc
NVCC_FLAGS  ?= -O2 -std=c++11

TARGET      := gpu_mem_hog
SRC         := gpu_mem_hog.cu

all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(NVCC_FLAGS) $(SRC) -o $(TARGET)

run: $(TARGET)
	@echo "Usage: ./$(TARGET) <GB_to_allocate> [device_id]"
	./$(TARGET)

clean:
	rm -f $(TARGET)

