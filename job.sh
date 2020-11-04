#!/bin/bash

sleep 5 # wait for gcsfuse

./copy_base_data.sh # copy base data
./extract_feed_data.sh Data/feed_data.tsv # extract feed_data from db
python process_data_train_model.py
