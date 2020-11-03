import numpy as np
import pandas as pd
import glob
import pygeohash as pg
import os

def preprocess_data(Data_dir, processed_file):
    if not os.path.exists(Data_dir):
        print("Data folder not exists")
    else:
        files = glob.glob(Data_dir+"*.tsv")
        all_data = pd.DataFrame()

        for x in files:
            data = pd.read_csv(x, sep='\t', lineterminator='\n')
            all_data = pd.concat([all_data, data])

        m, _ = all_data.shape
        text_data = all_data[all_data.columns.to_list()[-1]].fillna(0)
        location = all_data["posted_location"].fillna(0)
        text_len = np.zeros(shape=(m, 1), dtype=float)
        hash_tags = np.zeros(shape=(m, 1), dtype=float)
        lat = np.zeros(shape=(m, 1), dtype=float)
        lng = np.zeros(shape=(m, 1), dtype=float)

        i = 0
        for x in text_data:
            text_len[i] = len(str(x))
            hash_tags[i] = str(x).count('#')
            y = pg.decode(str(location.values[i]))
            lat[i] = y[0]
            lng[i] = y[1]
            i += 1

        cols = ['feed_id', 'uid', 'ptid', 'likes', 'comments', 'post_age',
                'textlength', 'hashtags', 'latitude', 'longitude', 'urls']
        pro_data = pd.DataFrame(columns=cols)
        pro_data["uid"] = all_data['postUserId']
        pro_data["ptid"] = all_data['feedObjectId']
        pro_data["feed_id"] = all_data['feed_id']
        pro_data["likes"] = all_data['likes']
        pro_data["comments"] = all_data['comments']
        pro_data["post_age"] = all_data[all_data.columns[7]]
        pro_data["textlength"] = text_len
        pro_data["hashtags"] = hash_tags
        pro_data["latitude"] = lat
        pro_data["longitude"] = lng
        pro_data["urls"] = all_data['numberOfMediaUrls']
        pro_data.to_csv(Data_dir+"{0}".format(processed_file), index=False)
        print("Data Processed !!! \n File:{0} saved in {1}".format(
            processed_file, Data_dir))


def create_training_data(Data_dir, processed_file):
    data = pd.read_csv(Data_dir + processed_file)
    data.drop(["uid", "ptid", "feed_id"], axis=1, inplace=True)
    return data.values
