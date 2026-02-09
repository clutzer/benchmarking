
MPI_INCLUDE := $(shell dirname `dpkg -L libopenmpi-dev | grep include/mpi.h`)
MPI_HOME    ?= $(shell dirname $(MPI_INCLUDE))

ifeq ($(MPI_HOME),)
$(error MPI_HOME is not set — cannot build with MPI support)
endif

CUDA_NVCC := $(firstword \
    $(wildcard /usr/local/cuda/bin/nvcc) \
    $(shell which nvcc 2>/dev/null) \
    $(shell dpkg -L nvidia-cuda-toolkit 2>/dev/null | grep -m 1 '/bin/nvcc$$'))

ifeq (,$(CUDA_NVCC))
$(error Cannot find nvcc)
endif

CUDA_BIN  := $(dir $(CUDA_NVCC))
CUDA_HOME ?= $(patsubst %/bin/,%,$(CUDA_BIN))

ifeq ($(CUDA_HOME),)
$(error CUDA_HOME is not set - cannot build at all)
endif

LIB_PATH=./nccl/build/lib:$(LD_LIBRARY_PATH)

NCCL_HOME ?= $(shell realpath ./nccl/build)

.PHONY: help_all_reduce_perf help_vars

help_all_reduce_perf:
	LD_LIBRARY_PATH=$(LIB_PATH) ./nccl-tests/build/all_reduce_perf --help
help_vars:
	@echo "CUDA_HOME:       $(CUDA_HOME)"
	@echo "CUDA_BIN:        $(CUDA_BIN)"
	@echo "LD_LIBRARY_PATH: $(LIB_PATH)"
	@echo "MPI_HOME:        $(MPI_HOME)"
	@echo "NCCL_HOME:       $(NCCL_HOME)"

.PHONY: build-nccl
build-nccl:
	-@git clone https://github.com/NVIDIA/nccl.git
	(cd nccl; make src.build CUDA_HOME=$(CUDA_HOME))

.PHONY: build-nccl-tests
build-nccl-tests:
	#sudo apt update
	#sudo apt -y install libopenmpi-dev libnccl-dev
	echo "MPI_HOME: $(MPI_HOME)"
	-@git clone https://github.com/NVIDIA/nccl-tests.git
	(cd nccl-tests; make MPI_HOME=$(MPI_HOME) CUDA_HOME=$(CUDA_HOME) NCCL_HOME=$(NCCL_HOME) MPI=1)

.PHONY: run-all_reduce_perf-2x-gpu
run-all_reduce_perf-2x-gpu:
	LD_LIBRARY_PATH=$(LIB_PATH) ./nccl-tests/build/all_reduce_perf -b 8 -e 256M -f 2 -g 2

.PHONY: run-mpi-2x-gpu
run-mpi-2x-gpu:
	mpirun -np 2 ./nccl-tests/build/all_reduce_perf -b 8 -e 128M -f 2 -g 1

