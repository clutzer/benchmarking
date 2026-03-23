# Force the NIC to use the Peer-to-Peer DMA path
export MLX5_P2P_GDR_EN=1
export MLX5_SCATTER_TO_GDR=1

# Tell the perftest tool to use DMA-BUF if available
ib_write_bw -d mlx5_0 --use_cuda=0 -x 3 -a -F --use_cuda_dmabuf 

echo "----------------------------------------------------"
echo "BENCHMARK COMPLETE."
echo "To close this pane, press: Ctrl+b then x"
echo "To close the whole session, press: Ctrl+b then d"
echo "----------------------------------------------------"
