#!/bin/sh

sudo apt update

NVIDIA_DRIVER=$(ubuntu-drivers devices 2>&1 | grep nvidia-driver | grep recommended | awk '{ print $3; }' | head -1)

echo
echo "Installing NVIDIA driver $NVIDIA_DRIVER"
echo
sudo apt install -y $NVIDIA_DRIVER

echo
echo "Attempting to load NVIDIA driver..."
echo
modprobe nvidia

echo
echo "Using nvidia-smi to verify..."
echo
nvidia-smi topo -m
