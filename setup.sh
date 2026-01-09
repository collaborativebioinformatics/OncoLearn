#!/bin/bash

# Setup script for OncoLearn project

# Activate the conda environment
# Note: Make sure the DeepOmix environment exists before running this
conda activate DeepOmix

# Clone the DeepOmix repository
# Note: Verify the repository URL exists and is accessible
# If the repository doesn't exist, you may need to:
# 1. Check if it's been renamed or moved
# 2. Verify you have access to the repository
# 3. Use the correct repository URL
git clone https://github.com/CancerProfiling/DeepOmix.git

