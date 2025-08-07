#!/bin/bash

echo "Installing MNIST Document Processing Requirements..."
echo

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "Error: Python3 is not installed"
    exit 1
fi

# Check if pip is installed
if ! command -v pip3 &> /dev/null; then
    echo "Error: pip3 is not installed"
    exit 1
fi

echo "Installing required packages..."
pip3 install -r requirements.txt

if [ $? -ne 0 ]; then
    echo "Error: Failed to install some packages"
    exit 1
fi

echo
echo "All packages installed successfully!"
echo
echo "You can now run:"
echo "python3 main.py --mode train --model cnn --epochs 10"
echo
