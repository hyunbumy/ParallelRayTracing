CUDA_INSTALL_PATH := /opt/cuda
ifeq ($(HOSTNAME), hpc-login3)
	CUDA_INSTALL_PATH = /usr/usc/cuda/9.2
endif

CXX := g++
CC := gcc
LINK := nvcc
NVCC  := nvcc

# Includes
INCLUDES = -I. -I$(CUDA_INSTALL_PATH)/include

# Common flags
COMMONFLAGS += $(INCLUDES)
LINKFLAGS += -arch=sm_30 -std=c++11
ifeq ($(HOSTNAME), hpc-login3)
	LINKFLAGS = -arch=sm_60 -std=c++11
endif
NVCCFLAGS += -dc $(COMMONFLAGS) $(LINKFLAGS)
CXXFLAGS += -std=c++11 $(COMMONFLAGS)
CFLAGS += $(COMMONFLAGS)

LIB_CUDA := -L$(CUDA_INSTALL_PATH)/lib64 -lcudart
BUILDDIR = build
OBJS = $(BUILDDIR)/CudaRT.cu.o $(BUILDDIR)/CudaSphere.cu.o $(BUILDDIR)/CudaTriangle.cu.o \
	   $(BUILDDIR)/Main.cpp.o $(BUILDDIR)/CudaRTIter.cu.o
TARGET = main
LINKLINE = $(LINK) $(LINKFLAGS) -o $(BUILDDIR)/$(TARGET) $(OBJS) $(LIB_CUDA)

.SUFFIXES: .c .cpp .cu .o

all: dir $(BUILDDIR)/$(TARGET)

dir:
	mkdir -p $(BUILDDIR)

$(BUILDDIR)/%.c.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

$(BUILDDIR)/%.cu.o: %.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

$(BUILDDIR)/%.cpp.o: %.cpp
	$(NVCC) $(CXXFLAGS) -c $< -o $@

$(BUILDDIR)/$(TARGET): $(OBJS) Makefile
	$(LINKLINE)

clean:
	rm -rf build
