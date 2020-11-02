#!/bin/bash

app_path="$(pwd)/app.py"

mkdir -p logs
cp -R -n ./base_data/logs ./

python $app_path
