#!/bin/bash

sleep 5 # wait for gcsfuse

mkdir -p logs
cp -R -n ./base_data/logs/* ./logs/

python app.py
