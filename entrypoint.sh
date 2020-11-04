#!/bin/bash

sleep 5 # wait for gcsfuse

./scripts/copy_base_model.sh
python app.py
