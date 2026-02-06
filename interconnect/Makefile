
.PHONY: ubuntu_build_cuda_samples
ubuntu_build_cuda_samples:
	sudo apt install -y nvidia-cuda-samples
	@mkdir -p build bin
	@examples_dir=$$(dpkg -L nvidia-cuda-samples | grep /examples | head -1); \
	if [ -z "$$examples_dir" ]; then \
		echo "ERROR: Could not find CUDA samples examples directory"; \
		exit 1; \
	fi; \
	echo "Copying CUDA samples from $$examples_dir"; \
	cp -a "$$examples_dir" build/; \
	(cd build/examples/Samples/5_Domain_Specific/p2pBandwidthLatencyTest/; make);
	echo "Copying binaries into ./bin..."
	@cp build/examples/bin/x86_64/linux/release/* ./bin
