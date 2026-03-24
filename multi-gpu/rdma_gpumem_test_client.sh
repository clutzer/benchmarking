export MLX5_P2P_GDR_EN=1
export MLX5_SCATTER_TO_GDR=1

ib_write_bw $GPU_0_IP -d mlx5_1 -x 3 -a --use_cuda=1 -F --use_cuda_dmabuf 

echo "----------------------------------------------------"
echo "BENCHMARK COMPLETE."
echo "To close this pane, press: Ctrl+b then x"
echo "To close the whole session, press: Ctrl+b then d"
echo "----------------------------------------------------"
