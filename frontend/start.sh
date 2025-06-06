#!/bin/bash

# Start backend in the background
python3 dummy_backend.py &

# Start nginx in the foreground
nginx -g "daemon off;"
