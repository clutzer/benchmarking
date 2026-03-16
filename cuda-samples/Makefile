
CUDA_SAMPLES := \
	5_Domain_Specific/p2pBandwidthLatencyTest \
	1_Utilities/deviceQuery

.PHONY: ubuntu_build_cuda_samples
ubuntu_build_cuda_samples:
	#-rm -rf build
	@mkdir -p build bin
	@-(cd build; git clone https://github.com/NVIDIA/cuda-samples.git)
	examples_dir=./build/cuda-samples/Samples; \
	if [ -z "$$examples_dir" ]; then \
		echo "ERROR: Could not find CUDA samples examples directory"; \
		exit 1; \
	fi; \
	for sample in $(CUDA_SAMPLES); do \
		echo "Building CUDA sample: $$sample"; \
		(cd build/cuda-samples/Samples/$$sample && cmake . && make) || exit 1; \
		name=$${sample##*/}; \
		echo "Copying [$$name] ($$sample) to bin directory..."; \
		cp ./build/cuda-samples/Samples/$$sample/$$name ./bin; \
	done
	@echo Done.

.PHONY: ubuntu_cuda_samples_location
ubuntu_cuda_samples_location:
	@dpkg -L nvidia-cuda-samples | grep /examples | head -1
