# 使用的编译器
NVCC        = nvcc

# 编译选项
NVCC_FLAGS  = -arch=sm_80 -O3 -std=c++14

# 目标可执行文件名称
TARGET      = vector_add

# 源文件
SRCS        = vector_add.cu

# 伪目标：编译目标文件
all: $(TARGET)

# 编译映射内存测试可执行文件
$(TARGET): $(SRCS)
	$(NVCC) $(NVCC_FLAGS) -o $@ $^

# 清理目标：删除编译生成的可执行文件
clean:
	rm -f $(TARGET)