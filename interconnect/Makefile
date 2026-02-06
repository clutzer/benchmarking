
CUDA_SAMPLES := \
	5_Domain_Specific/p2pBandwidthLatencyTest \
	1_Utilities/deviceQuery \
	1_Utilities/bandwidthTest

.PHONY: ubuntu_build_cuda_samples
ubuntu_build_cuda_samples:
	sudo apt install -y nvidia-cuda-samples
	@mkdir -p build bin
	@examples_dir=$$(dpkg -L nvidia-cuda-samples | grep '/examples$$' | head -1); \
	if [ -z "$$examples_dir" ]; then \
		echo "ERROR: Could not find CUDA samples examples directory"; \
		exit 1; \
	fi; \
	echo "Copying CUDA samples from $$examples_dir"; \
	cp -a "$$examples_dir" build/; \
	for sample in $(CUDA_SAMPLES); do \
		echo "Building CUDA sample: $$sample"; \
		(cd build/examples/Samples/$$sample && $(MAKE)) || exit 1; \
	done; \
	echo "Copying binaries into ./bin..."; \
	cp build/examples/bin/x86_64/linux/release/* ./bin

.PHONY: ubuntu_cuda_samples_location
ubuntu_cuda_samples_location:
	@dpkg -L nvidia-cuda-samples | grep /examples | head -1
