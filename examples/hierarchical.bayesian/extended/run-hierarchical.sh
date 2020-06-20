#!/bin/bash

# this will exit the script if a command fails
set -e

python3 phase0.py
python3 phase1.py
python3 phase2.py
python3 phase3a.py
python3 phase3b.py
