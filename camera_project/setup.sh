#!/bin/bash
# ================================================
# PaddleOCR Setup Script for Raspberry Pi 5
# Run with: bash setup.sh
# ================================================

echo "Updating system packages..."
sudo apt update && sudo apt upgrade -y

echo "Installing system dependencies..."
sudo apt install -y \
    python3-pip \
    python3-opencv \
    libopencv-dev \
    libatlas-base-dev \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libgl1 \
    python3-picamera2

echo "Installing Python packages..."
pip3 install --break-system-packages \
    paddlepaddle \
    paddleocr \
    opencv-python \
    numpy \
    Pillow

echo ""
echo "Setup complete! Run the OCR script with:"
echo "  python3 bottle_ocr.py"