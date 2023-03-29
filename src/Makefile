# Location of the CUDA and CWP

CUDA_PATH ?= /disk1/cuda
# Location of the CUDA Toolkit
CWPROOT ?= /disk1/cwp
# Location of the CWP  http://www.cwp.mines.edu/cwpcodes
CUDASDK_PATH ?= /disk1/cudasdk/NVIDIA_CUDA-10.1_Samples
# Location of the CUDA SDK

SMS ?= 75
#SMS 70 75
TARGET_SIZE ?= 64
HOST_COMPILER ?= g++

# Common includes
INCLUDES  := -I ${CUDASDK_PATH}/common/inc -I ${CWPROOT}/include
LIBRARIES := -L ${CWPROOT}/lib
LIBRARIES += -lcufft -lsu -lpar -lcwp 
NVCC := $(CUDA_PATH)/bin/nvcc -ccbin $(HOST_COMPILER)
ALL_FLAGS := -m${TARGET_SIZE}

# Gencode arguments
GENCODE_FLAGS := -gencode arch=compute_$(SMS),code=sm_$(SMS)

# Target rules
all: build

build: si
	mkdir -p ../bin
	cp si ../bin

si.o:si.cu
	$(NVCC) $(INCLUDES) $(ALL_FLAGS) $(GENCODE_FLAGS) -o $@ -c $<

siCUFFT.o:siCUFFT.cu
	$(NVCC) $(INCLUDES) $(ALL_FLAGS) $(GENCODE_FLAGS) -o $@ -c $<

si: si.o siCUFFT.o
	$(NVCC) $(ALL_FLAGS) $(GENCODE_FLAGS) -o $@ $+ $(LIBRARIES)

run: build
	./si

clean:
	rm -f si ../bin/si si.o siCUFFT.o
	rm -r ../bin 
clobber: clean
