#!/bin/bash
# Script to download WildScenes V-01 data from CSIRO DAP

# Install AWS CLI
pip install awscli

# Set your credentials (Get these from https://data.csiro.au/collection/csiro:61541)
export AWS_ACCESS_KEY_ID="YOUR_ACCESS_KEY"
export AWS_SECRET_ACCESS_KEY="YOUR_SECRET_KEY"
export AWS_ENDPOINT_URL="https://s3.data.csiro.au"

# Create directory and download ONLY V-01 files
mkdir -p ./data/V-01
aws s3 cp --endpoint-url $AWS_ENDPOINT_URL --recursive \
  s3://dapprd/000061541v003/ ./data/V-01/ \
  --exclude "*" --include "*V-01*"

echo "Download complete. Data is in ./data/V-01/data/WildScenes/"
