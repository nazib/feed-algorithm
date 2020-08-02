from PIL import Image
import numpy as np
import flask
from flask import jsonify, make_response
import io
from model import*
from preprocess_data import*
from rank_logics import*
import pandas as pd
from pandas import json_normalize
from rank_logics import*
from LinearModel import*
from NonLinearModel import*

app = flask.Flask(__name__)
model = None
theta =None


@app.route('/linear/rank',methods=['POST'])
def rank():
    #data = np.array([12,10,8,3,16,18,20,8],dtype=float)
    data = flask.request.get_json()
    data = np.array(list(data.values()),dtype=float)
    model = LinearModel("RandomForrest", 0.04)
    rank = model.Rank(data)
    '''
    TH= 60*36
    values = data[1]
    decay = theta[1]*np.exp(1-(values/TH))
    rank = np.sum(data*theta)
    '''
    return jsonify(rank)

@app.route('/linear/bulkrank',methods=['POST'])
def bulk_rank():
    feed_data = flask.request.get_json()
    feed_frame = pd.read_json(feed_data,typ='frame', orient='split')
    feed_frame_num = feed_frame.drop(["uid","ptid","feed_id"],axis=1)
    model = LinearModel("RandomForrest", 0.04)
    feed_frame['RankScore'] = model.BulkRank(feed_frame_num)
    return feed_frame.to_json()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

