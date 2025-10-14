# vim: set ts=4 sw=4:

llm-01:
	python llm_bench.py --device cuda:0

val2017.zip:
	wget http://images.cocodataset.org/zips/val2017.zip

image-01: val2017.zip
	# Note: you need to unzip val2017.zip
	python image_bench.py -h

install-fp-quant:
	pip install fp_quant

install-qutlass:
	@echo "Qutlass requires cuda-12-8, i.e. nvcc --version 12.8"
	-git clone https://github.com/IST-DASLab/qutlass.git
	cd qutlass && pip install --no-build-isolation .

install-cuda-12-8:
	wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
	sudo dpkg -i cuda-keyring_1.1-1_all.deb
	sudo apt update
	sudo apt -y install cuda-toolkit-12-8 cuda-nvcc-12-8

install-cuda-13-0:
	wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-ubuntu2404.pin
	sudo mv cuda-ubuntu2404.pin /etc/apt/preferences.d/cuda-repository-pin-600
	wget https://developer.download.nvidia.com/compute/cuda/13.0.2/local_installers/cuda-repo-ubuntu2404-13-0-local_13.0.2-580.95.05-1_amd64.deb
	sudo dpkg -i cuda-repo-ubuntu2404-13-0-local_13.0.2-580.95.05-1_amd64.deb
	sudo cp /var/cuda-repo-ubuntu2404-13-0-local/cuda-*-keyring.gpg /usr/share/keyrings/
	sudo apt-get update
	sudo apt-get -y install cuda-toolkit-13-0
	# needs:
	# export CUDA_HOME="/usr/local/cuda"
	# export PATH="$PATH:$CUDA_HOME/bin"
	# export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"
