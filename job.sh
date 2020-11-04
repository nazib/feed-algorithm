#!/bin/bash

sleep 5 # wait for gcsfuse

./extract_feed_data.sh Data/feed_data.tsv
python process_data_train_model.py
