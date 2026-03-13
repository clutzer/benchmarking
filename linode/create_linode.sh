#!/bin/sh

REGION="${REGION:-us-lax}"
GPUS="${GPUS:-1}"
DISTRO="${DISTRO:-ubuntu22.04}"

#
# Create a 2x RTX PRO 6000 GPU VM in LAX:
#
linode-cli linodes create \
    --region ${REGION} \
    --image linode/${DISTRO} \
    --type g3-gpu-rtxpro6000-blackwell-${GPUS} \
    --tag gpu-benchmarking \
    --label rtx-pro-${GPUS}x-gpu \
    --root_pass `pwgen 20 1`
