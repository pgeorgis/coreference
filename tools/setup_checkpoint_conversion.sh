#!/bin/bash

# Create Python virtual environment
python -m venv .venv
source .venv/bin/activate

# Update pip
pip install -U pip setuptools wheel
pip install --upgrade pip

# Install Python dependencies
pip install -r requirements.txt
