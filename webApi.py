import numpy as np
import flask
from flask import jsonify, make_response, abort
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
import logging
import datetime

app = flask.Flask(__name__)
nonlin_model = None
lin_model = None
theta =None

@app.errorhandler(400)
def value_error(e):
    return jsonify(error=str(e)), 400

def check_attributes(userdata):
    for x in userdata.keys():
        if x == 'gender' or x == 'statusLevel': #x == 'city' or x == 'country' or 
            if isinstance(userdata[x],str):
                if len(userdata[x]) == 0:
                    app.logger.error('{} should not be an empty string User/Poster Attribute'.format(x))
                    abort(400, description='{} should not be an empty string in User/Poster Attribute'.format(x))
            else:
                app.logger.error('{} should be string'.format(x))
                #raise ValueError('invalid entry in {}'.format(x))
                abort(400,description= '{} should be string in User/ Poster Attribute'.format(x))

        if x == "totalReceivedPostComments" or x == "totalReceivedPostLikes" or x=="numberOfFollowers":
            if not isinstance(userdata[x],(float,int)):
                if userdata[x] == None:
                    userdata[x] =0.0
                else:
                    app.logger.error("{} should be float value".format(x))
                    #raise ValueError('invalid entry in {}'.format(x))
                    abort(400,description= '{} should be float value in User/Poster Attribute'.format(x))

def check_feeditems(feeditems):
    for feed in feeditems:
        for key in feed.keys():
            if key == 'feedItemId':
                if isinstance(feed[key],str):
                    if len(feed[key]) == 0:
                        app.logger.error('Feed id should not be empty')
                        abort(400,description= 'Feed ID should not be empty')
                else:
                    app.logger.error("Feed id must be string")
                    abort(400,description= 'Feed ID must be string')
            elif key== 'numberOfLikes' or \
                 key == 'numberOfComments' or \
                 key== 'postTextLength' or \
                 key == 'numberOfHashTags' or key =='latitude' or key=='longitude' or key== 'numberOfMediaUrls':
                 if not isinstance(feed[key],(float,int)):
                     if feed[key] == None:
                         feed[key] = 0.0
                     else:
                         app.logger.error("Value of {} is not appropriate in Feed ID {}".format(key, feed['feedItemId']))
                        #raise ValueError("Value of {} is not appropriate".format(key))
                         abort(400,description= "Value of {} is not appropriate in Feed ID {}".format(key,feed['feedItemId']))
            elif key == "postedDate":
                if not isinstance(feed[key],str):
                    abort(400,description= "Value of {} must be string in Feed ID {}".format(key, feed['feedItemId']))
                else:
                    try:
                        datetime.datetime.strptime(feed[key],'%Y-%m-%d %H:%M:%S')
                    except ValueError:
                        abort(400, description = "Date time format is not correct in Feed ID {}".format( feed['feedItemId']))
            elif key == 'posterAttributes':
                check_attributes(feed[key])
            else:
                app.logger.error("Please Check payload ")
                abort(400, description = 'Please Check payload')

def check_json(data):
    for x in data.keys():
        if x == 'userAttributes':
            check_attributes(data['userAttributes'])
        elif x == 'feedItems':
            check_feeditems(data['feedItems'])
        else:
            abort(400, description = 'Data attributes are not correct in the payload')

@app.route('/health',methods=['GET'])
def health_check():
    status = {200:"Container running successfully"}
    return jsonify(status)

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
    _,ranks = lin_model.GlobalRank(data['feedItems'])
    json_obj = {}
    json_obj["feedItemsRank"] = [x for x in ranks.values()]
    return jsonify(json_obj)

@app.route('/linear/personal_rank',methods=['POST'])
def LinPRank():
    data = flask.request.get_json(force=True)
    ranks = lin_model.PersonalRank(data)
    json_obj = {}
    json_obj["feedItemsRank"] = [x for x in ranks.values()]
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
    check_json(data)
    ranks = nonlin_model.PersonalRank(data)
    json_obj = {}
    json_obj["feedItemsRank"] = [x for x in ranks.values()]
    return jsonify(json_obj)

if __name__ == "__main__":
    nonlin_model = NonLinearModel()
    lin_model = LinearModel("RandomForrest", 0.04)
    #app.run(host="0.0.0.0", port=5000,  threaded=False)
    http_server = WSGIServer(('0.0.0.0', 8080), app)
    http_server.serve_forever()

