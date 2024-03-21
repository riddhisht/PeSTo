#!/bin/bash

# Define the model name
MODEL_NAME="3_21_2"

# Create the directory structure
mkdir -p "runs/$MODEL_NAME"
mkdir -p "model/save/$MODEL_NAME/test"

# Run the Python script
python -W ignore /blue/yanjun.li/riddhishthakare/PeSTo/model/main2.py --model_name $MODEL_NAME


