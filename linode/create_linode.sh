#!/bin/sh

#
# Create a 2x RTX PRO 6000 GPU VM in LAX:
#
linode-cli linodes create \
    --region us-lax \
    --image linode/ubuntu22.04 \
    --type g3-gpu-rtxpro6000-blackwell-2 \
    --tag gpu-benchmarking \
    --label rtx-pro-2x-gpu \
    --root_pass `pwgen 20 1`
