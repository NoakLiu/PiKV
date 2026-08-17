# PiKV CUDA Kernels Makefile (2026)
# Builds routing + full compression + scheduling into libpikv_kernels.so

NVCC ?= nvcc
CC ?= gcc
CXX ?= g++

CUDA_VERSION := $(shell nvcc --version 2>/dev/null | grep "release" | sed 's/.*release \([0-9]\+\.[0-9]\+\).*/\1/')
CUDA_MAJOR := $(shell echo $(CUDA_VERSION) | cut -d. -f1)

# Multi-arch for modern GPUs (V100→H100/Ada). Override: make ARCH_FLAGS="-arch=native"
ifndef ARCH_FLAGS
  ARCH_FLAGS = \
    -gencode arch=compute_70,code=sm_70 \
    -gencode arch=compute_75,code=sm_75 \
    -gencode arch=compute_80,code=sm_80 \
    -gencode arch=compute_86,code=sm_86 \
    -gencode arch=compute_89,code=sm_89 \
    -gencode arch=compute_90,code=sm_90
endif

DEBUG_FLAGS = -g -G -O0 -DDEBUG
RELEASE_FLAGS = -O3 -DNDEBUG --use_fast_math
PROFILE_FLAGS = -O3 -g -lineinfo -DNDEBUG

NVCC_FLAGS = $(ARCH_FLAGS) -std=c++17 -Xcompiler -fPIC --expt-relaxed-constexpr
CXX_FLAGS = -std=c++17 -fPIC -Wall -Wextra
CUDA_LIBS = -lcudart -lcublas -lcusparse

# Optional: CUDA include from CUDA_HOME
ifdef CUDA_HOME
  NVCC_FLAGS += -I$(CUDA_HOME)/include
  CUDA_LIBS += -L$(CUDA_HOME)/lib64
endif

# Full kernel set: unified API + specialized modules (full compression)
CUDA_SOURCES = \
  pikv_kernels.cu \
  pikv_routing_kernel.cu \
  pikv_compression_kernel.cu \
  pikv_scheduling_kernel.cu

CXX_SOURCES = test_kernels.cpp

CUDA_OBJECTS = $(CUDA_SOURCES:.cu=.o)
CXX_OBJECTS = $(CXX_SOURCES:.cpp=.o)

TARGET_LIB = libpikv_kernels.so
TARGET_STATIC = libpikv_kernels.a
TEST_EXEC = test_pikv_kernels

PREFIX ?= /usr/local

.PHONY: all debug release profile compression clean distclean test test-cuda test-py \
        install check-cuda help

all: $(TARGET_LIB) $(TEST_EXEC)

debug: NVCC_FLAGS += $(DEBUG_FLAGS)
debug: CXX_FLAGS += -g -O0 -DDEBUG
debug: all

release: NVCC_FLAGS += $(RELEASE_FLAGS)
release: CXX_FLAGS += -O3 -DNDEBUG
release: all

profile: NVCC_FLAGS += $(PROFILE_FLAGS)
profile: CXX_FLAGS += -O3 -g -DNDEBUG
profile: all

# Explicit alias: full compression suite is always linked in release/all
compression: release

$(TARGET_LIB): $(CUDA_OBJECTS)
	@echo "Linking shared library: $@ (routing + full compression + scheduling)"
	$(NVCC) $(NVCC_FLAGS) -shared -o $@ $^ $(CUDA_LIBS)

$(TARGET_STATIC): $(CUDA_OBJECTS)
	ar rcs $@ $^

$(TEST_EXEC): $(CXX_OBJECTS) $(TARGET_LIB)
	$(CXX) $(CXX_FLAGS) -o $@ $(CXX_OBJECTS) -L. -Wl,-rpath,'$$ORIGIN' -lpikv_kernels $(CUDA_LIBS)

%.o: %.cu
	@echo "Compiling CUDA: $<"
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

%.o: %.cpp
	@echo "Compiling C++: $<"
	$(CXX) $(CXX_FLAGS) -c $< -o $@

install: $(TARGET_LIB)
	mkdir -p $(PREFIX)/lib $(PREFIX)/include
	cp $(TARGET_LIB) $(PREFIX)/lib/
	cp *.h $(PREFIX)/include/ 2>/dev/null || true

clean:
	rm -f *.o *.so *.a $(TEST_EXEC) *.d

distclean: clean

test: $(TEST_EXEC)
	./$(TEST_EXEC)

test-cuda: $(TEST_EXEC)
	CUDA_VISIBLE_DEVICES=0 ./$(TEST_EXEC)

test-py:
	@python -c "from core.cuda.test_kernels import *; print('import OK')" 2>/dev/null \
	  || PYTHONPATH=../.. python test_kernels.py || true

check-cuda:
	@echo "CUDA Version: $(CUDA_VERSION)"
	@nvcc --version | head -2
	@nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader 2>/dev/null || true

help:
	@echo "PiKV CUDA Makefile (full compression)"
	@echo "  make release | debug | profile | compression"
	@echo "  make test | test-cuda | test-py | clean | install PREFIX=..."
	@echo "  ARCH_FLAGS='-arch=native' make release   # single-GPU fast build"
	@echo "Sources: $(CUDA_SOURCES)"
