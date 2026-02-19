#!/bin/bash
# Wipes all training state and starts fresh

echo "Clearing training state..."
rm -rf ./checkpoints ./tmp ./sync ./results ./logs ./log.log
echo "✓ Cleared: checkpoints, model weights, sync weights, results, logs"

echo ""
echo "Starting fresh training..."
python3 main.py
