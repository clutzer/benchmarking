#!/bin/bash

sudo apt install python3-venv python-is-python3

if [ ! -d benchmarking ]; then
    echo "Creating 'benchmarking' Python virtual environment..."
    python -m venv benchmarking
    source benchmarking/bin/activate
    pip install --upgrade pip
    pip install vllm
    pip install transformers
    pip install accelerate
    pip install torch
fi

echo
echo "Please activate your Python virtual environment by running:"
echo
echo "source benchmarking/bin/activate"
