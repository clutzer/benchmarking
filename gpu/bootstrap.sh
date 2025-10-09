#!/bin/bash

# Install pyenv as Ubuntu 24's version of python is incompatible with llmperf
if [ ! -d $HOME/.pyenv ]; then
    echo "Installing 'pyenv' to create a Python 3.11 environment..."
    curl https://pyenv.run | bash
    # Use a here-document with 'cat' to append the lines to your .profile file
    # The `EOF` marker must be on its own line and not be indented
    cat <<'EOF' >> ~/.profile

# pyenv configuration
export PYENV_ROOT="$HOME/.pyenv"
[[ -d $PYENV_ROOT/bin ]] && export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init --path)"
EOF
    source ~/.profile
    pyenv install 3.10
    pyenv versions
fi

if [ ! -d benchmarking ]; then
    pyenv local 3.10
    echo "Creating 'benchmarking' Python virtual environment..."
    python -m venv benchmarking
    source benchmarking/bin/activate
    pip install --upgrade pip
    pip install vllm
    pip install torch
    pip install -U git+https://github.com/ray-project/llmperf.git
fi

echo
echo "Please activate your Python virtual environment by running:"
echo
echo "source benchmarking/bin/activate"
