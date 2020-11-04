#!/bin/bash

sleep 5 # wait for gcsfuse

./scripts/copy_base_data.sh # copy base data
./scripts/extract_feed_data.sh Data/feed_data.tsv
./scripts/extract_groups_data.sh Data/groups_data.tsv
./scripts/extract_interests_data.sh Data/groups_data.tsv
python process_data_train_model.py
