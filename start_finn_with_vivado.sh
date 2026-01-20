#!/bin/bash
# Start FINN Docker with Vivado 2023.2 and Jupyter Notebook

echo "Starting FINN Docker with Vivado 2023.2..."

# Use the official FINN run-docker.sh script with Vivado environment
cd /home/hritik/Desktop/Hritik/Project/ellipse-regression-project/finn

# Set Vivado environment variables
export FINN_XILINX_PATH=/tools/Xilinx
export FINN_XILINX_VERSION=2023.2
export XILINX_VIVADO=/tools/Xilinx/Vivado/2023.2
export VIVADO_PATH=/tools/Xilinx/Vivado/2023.2

# Run FINN Docker with notebook mode
./run-docker.sh notebook

echo ""
echo "Jupyter stopped. To restart, run this script again."
