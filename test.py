import numpy as np
'''
import tensorflow as tf
from model import*
import pandas as pd
from preprocess_data import*
import os
from datetime import datetime
import keras
from scipy.spatial import distance
from scipy.signal import correlate2d
from keras.models import Model
from rank_logics import*

model_dir = os.path.abspath(os.getcwd()+"/logs/20200630-115457_noisy/")

model = vae_model([512,256,128,64,32,16,8])
model,_,_ = model.create_model()
model.load_weights(model_dir+"/VAE_noisy.h5")
mu = model.get_layer('mu')
mu_wgt = mu.get_weights()[1]
var = model.get_layer('var')
var_wgt = var.get_weights()[1]

### Generating Weights parameters from Distribution ####
theta = mu_wgt + var_wgt* np.random.rand(8)


Data_dir = "/media/nazib/E20A2DB70A2D899D/Ubuntu_desktop/Travello/RawData/new_feed_data/"
data_file = "AllFeedData.csv"
data = create_training_data(Data_dir,data_file)
N,M = data.shape
train_ratio = N*90//100
test_data = data[train_ratio+1:N,]

#output = model.predict(test_data)[0]
#pfm = np.corrcoef(output.ravel(),test_data.ravel()) 
#print("Performance ",np.mean(pfm))
#vae_mu = Model(inputs= model.input, outputs= [model.get_layer('mu').output,model.get_layer('var').output])
#mu, var = vae_mu.predict(test_data)
global_rank = GlobalRank(test_data, theta,Data_dir+data_file, train_ratio)
global_rank.to_csv("Global_Rank_by NN_noisy500.csv",index=False)
print("Processed")
'''
import math
time = np.linspace(0,1000,1000)
TH= 500
sigma = 200
decay = np.zeros(100)
decay = np.exp(-np.power(time - TH, 2.) / (2 * np.power(sigma, 2.)))

from matplotlib import pyplot as pl
pl.plot(time,decay)
pl.show()














