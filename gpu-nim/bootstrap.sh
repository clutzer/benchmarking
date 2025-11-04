#!/bin/bash

VENV_DIR=python-venv

sudo apt install python3-venv python-is-python3 make gcc nvtop

if [ ! -d $VENV_DIR ]; then
    echo "Creating '$VEND_DIR' Python virtual environment..."
    python -m venv $VENV_DIR
    source $VENV_DIR/bin/activate
    pip install --upgrade pip
    pip install pandas tabulate
fi

echo
echo "Please activate your Python virtual environment by running:"
echo
echo "source $VENV_DIR/bin/activate"
