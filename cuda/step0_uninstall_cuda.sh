#!/bin/sh

sudo systemctl stop nvidia-persistenced

sudo apt-get --purge remove "*cuda*" "*cublas*" "*cufft*" "*curand*" "*cusolver*" "*cusparse*" "*npp*" "*nvjpeg*" "nsight*" "*nvvm*" "*nvidia*"

sudo apt-get autoremove
sudo apt-get autoclean

sudo rm -rf /usr/local/cuda*
