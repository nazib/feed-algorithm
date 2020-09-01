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
from NonLinearModel import vae_model, NonLinearModel
from gevent.pywsgi import WSGIServer

app = flask.Flask(__name__)
nonlin_model = None
lin_model = None
theta =None


@app.route('/linear/train_model',methods=['GET'])
def LinTraining():
    FileName =os.getcwd() + "/Data/AllFeedData.csv"
    Training_ratio = 90
    Label_name = "likes"
    lin_model.prepare_feed_data(FileName,Training_ratio,Label_name)
    success = lin_model.fit()
    success = {"Training Status" : success}
    return jsonify(success)

@app.route('/nonlinear/train_model',methods=['GET'])
def NonTraining():
    Data_dir =os.getcwd() + "/Data/"
    FileName = 'AllFeedData.csv'
    success = nonlin_model.fit(Data_dir,FileName)
    success = {"Training Status" : success}
    return jsonify(success)

@app.route('/linear/global_rank',methods=['POST'])
def LinGRank():
    #data = np.array([12,10,8,3,16,18,20,8],dtype=float)
    data = flask.request.get_json(force=True)
    rank = lin_model.GlobalRank(data['FeedData'])
    rank = {"GlobalRank": rank}
    return jsonify(rank)

@app.route('/linear/personal_rank',methods=['POST'])
def LinPRank():
    data = flask.request.get_json(force=True)
    personal_rank, global_rank = lin_model.PersonalRank(data)
    json_obj = {"PersonalRank": personal_rank, "GlobalRank":global_rank}
    return jsonify(json_obj)


@app.route('/nonlinear/global_rank',methods=['POST'])
def NonGRank():
    #data = np.array([12,10,8,3,16,18,20,8],dtype=float)
    data = flask.request.get_json(force=True)
    _,ranks = nonlin_model.GlobalRank(data["feedItems"])
    json_obj = {}
    json_obj["feedItemsRank"] = [x for x in ranks.values()]
    return jsonify(json_obj)

@app.route('/nonlinear/personal_rank',methods=['POST'])
def NonPRank():
    data = flask.request.get_json(force=True)
    ranks = nonlin_model.PersonalRank(data)
    json_obj = {}
    json_obj["feedItemsRank"] = [x for x in ranks.values()]
    return jsonify(json_obj)

if __name__ == "__main__":    
    #app.run(host="0.0.0.0", port=5000,  threaded=False)
    nonlin_model = NonLinearModel()
    lin_model = LinearModel("RandomForrest", 0.04)
    http_server = WSGIServer(('0.0.0.0', 5000), app)
    http_server.serve_forever()

