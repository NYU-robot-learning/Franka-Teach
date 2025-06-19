#!/bin/bash

# Ensure the log directory exists
mkdir -p server_logs

# Start the left robot server
python -u franka_server.py robot=deoxys_left > server_logs/deoxys_left.log 2>&1 &

# Start the right robot server
python -u franka_server.py robot=deoxys_right > server_logs/deoxys_right.log 2>&1 &

# Optionally wait for both to finish (comment out if not desired)
# wait
