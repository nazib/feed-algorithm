#!/bin/bash

./extract_feed_data.sh Data/feed_data.tsv
python process_data_train_model.py
