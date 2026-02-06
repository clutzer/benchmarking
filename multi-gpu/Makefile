
MPI_INCLUDE := $(shell dirname `dpkg -L libopenmpi-dev | grep include/mpi.h`)
MPI_HOME    ?= $(shell dirname $(MPI_INCLUDE))

CUDA_NVCC   := $(shell dpkg -L nvidia-cuda-toolkit | grep 'toolkit/bin/nvcc$')
CUDA_BIN    := $(shell dirname $(CUDA_NVCC))
CUDA_HOME   ?= $(shell dirname $(CUDA_BIN))

ifeq ($(MPI_HOME),)
$(error MPI_HOME is not set — cannot build with MPI support)
endif

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
	(cd nccl-tests; make MPI_HOME=$(MPI_HOME) CUDA_HOME=$(CUDA_HOME) MPI=1)


.PHONY: run-mpi-2x-gpu
run-mpi-2x-gpu:
	mpirun -np 2 ./nccl-tests/build/all_reduce_perf -b 8 -e 128M -f 2 -g 1

