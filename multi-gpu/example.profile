export CUDA_HOME=/usr/local/cuda

# Update Path and Library paths using the variable
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

export GPU_0_NIC=eth1
export GPU_1_NIC=eth2

export GPU_0_IP=$(ip -4 addr show $GPU_0_NIC | grep "inet\b" | awk '{print $2}' | cut -d/ -f1)
export GPU_1_IP=$(ip -4 addr show $GPU_1_NIC | grep "inet\b" | awk '{print $2}' | cut -d/ -f1)

echo
echo "E/W Network Summary:"
echo
echo "GPU 0 NIC($GPU_0_NIC) IP: $GPU_0_IP"
echo "GPU 1 NIC($GPU_1_NIC) IP: $GPU_1_IP"
echo
