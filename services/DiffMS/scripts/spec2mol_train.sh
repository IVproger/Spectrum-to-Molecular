#!/bin/bash

# Define the path to your .env file
ENV_FILE="/home/i_golov/Spectrum_Structure_prediction/Spectrum-to-Molecular/.env"

# Check if the .env file exists
if [ -f "$ENV_FILE" ]; then
    source "$ENV_FILE"
else
    echo "Error: .env file not found at $ENV_FILE"
    exit 1
fi

python src/spec2mol_main.py