# 使用的编译器
NVCC        = nvcc

# 编译选项
# -arch=sm_80 针对 A100，-O3 代表优化级别 3
# 加入 -std=c++11 来启用对 C++11 特性的支持
NVCC_FLAGS  = -arch=sm_80 -O3 -std=c++11

# 目标可执行文件名称
LATENCY_TARGET      = copy_latency_test
BANDWIDTH_TARGET    = copy_bandwidth_test

# 源文件
LATENCY_SRCS        = copy_latency_test.cu
BANDWIDTH_SRCS      = copy_bandwidth_test.cu

# 伪目标：编译所有内容
all: $(LATENCY_TARGET) $(BANDWIDTH_TARGET)

# 编译延迟测试可执行文件
$(LATENCY_TARGET): $(LATENCY_SRCS)
	$(NVCC) $(NVCC_FLAGS) -o $@ $^

# 编译带宽测试可执行文件
$(BANDWIDTH_TARGET): $(BANDWIDTH_SRCS)
	$(NVCC) $(NVCC_FLAGS) -o $@ $^


# 清理目标：删除编译生成的可执行文件
clean:
	rm -f $(LATENCY_TARGET) $(BANDWIDTH_TARGET) $(BANDWIDTH_TEST_TARGET)
