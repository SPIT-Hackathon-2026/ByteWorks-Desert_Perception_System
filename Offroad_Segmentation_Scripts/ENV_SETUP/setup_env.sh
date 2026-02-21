#!/usr/bin/env bash

set -e   # Stop if any command fails

ENV_NAME="EDU"
PYTHON_VERSION="3.10"

echo "======================================"
echo "Checking if conda is installed..."
echo "======================================"

if ! command -v conda &> /dev/null; then
    echo "Conda is not installed."
    echo "Install Miniconda first:"
    echo "https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

echo "Conda found!"

echo "======================================"
echo "Initializing conda for bash..."
echo "======================================"

eval "$(conda shell.bash hook)"

echo "======================================"
echo "Creating environment: $ENV_NAME"
echo "======================================"

if conda env list | grep -q "$ENV_NAME"; then
    echo "Environment already exists. Skipping creation."
else
    conda create -n $ENV_NAME python=$PYTHON_VERSION -y
fi

echo "======================================"
echo "Activating environment..."
echo "======================================"

conda activate $ENV_NAME

echo "======================================"
echo "Installing PyTorch + CUDA 11.8"
echo "======================================"

conda install -c pytorch -c nvidia -c conda-forge \
    pytorch torchvision pytorch-cuda=11.8 -y

echo "======================================"
echo "Installing additional packages..."
echo "======================================"

pip install \
    ultralytics \
    opencv-contrib-python \
    tqdm \
    matplotlib \
    pillow \
    numpy

echo "======================================"
echo "Environment setup complete!"
echo "Environment name: $ENV_NAME"
echo "======================================"