from keras.applications import ResNet50
from keras.preprocessing.image import img_to_array
from keras.applications import imagenet_utils
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

app = flask.Flask(__name__)
model = None
theta =None

def load_model():
    # load the pre-trained Keras model (here we are using a model
    # pre-trained on ImageNet and provided by Keras, but you can
    # substitute in your own networks just as easily)
    global model
    model = vae_model([512,256,128,64,32,16,8])
    model,_,_ = model.create_model()
    model.load_weights("Flask-App/VAE_noisy.h5")
    mu = model.get_layer('mu')
    mu_wgt = mu.get_weights()[1]
    var = model.get_layer('var')
    var_wgt = var.get_weights()[1]
    ### Generating Weights parameters from Distribution ####
    global theta
    theta = mu_wgt + var_wgt * np.random.randn(8)

@app.route('/simple_rank',methods=['POST'])
def rank():
    #data = np.array([12,10,8,3,16,18,20,8],dtype=float)
    data = flask.request.get_json()
    data = np.array(list(data.values()),dtype=float)
    TH= 60*36
    values = data[2]
    decay = theta[2]*np.exp(1-(values/TH))
    rank = np.sum(data*theta)
    return jsonify(rank)

@app.route('/bulkrank',methods=['POST'])
def bulk_rank():
    feed_data = flask.request.get_json()
    feed_frame = pd.read_json(feed_data,typ='frame', orient='split')
    feed_frame_num = feed_frame.drop(["uid","ptid","feed_id"],axis=1)
    rank_scores = GlobalRank(feed_frame_num.values,theta)
    feed_frame['RankScore'] = rank_scores
    return feed_frame.to_json()

if __name__ == "__main__":
    print(("* Loading Keras model and Flask starting server..."
        "please wait until server has fully started"))
    load_model()
    app.run(host="0.0.0.0", port=5000)
