import numpy as np
import requests
import json
import pandas as pd

# initialize the Keras REST API endpoint URL along with the input
# image path
payload = {
    "like":10,
    "cmt":5,
    "age":4,
    "textlength":50,
    "hashtags":3,
    "latitude":141.30,
    "longitude":30.3,
    "urls" :3
}
# submit the request
#r = requests.post("http://34.72.186.134:80/simple_rank", json =payload)
r = requests.post("http://0.0.0.0:5000/linear/rank", json =payload)
print(r.json())

'''
Data_dir = "/media/nazib/E20A2DB70A2D899D/Ubuntu_desktop/Travello/RawData/new_feed_data/"
data_file = "AllFeedData.csv"

bulk_data = pd.read_csv(Data_dir + data_file)
N,M = bulk_data.shape
ratio = N*90//100
bulk_data = bulk_data.loc[ratio+1:N,:]
payload = bulk_data.to_json(orient='split')

r = requests.post("http://127.0.0.1:8000/feedRank/bulk", json =payload)

ranked_data = pd.DataFrame(r.json())
ranked_data.to_csv("BulkRank_gcp.csv")
print("Data Saved")
'''

