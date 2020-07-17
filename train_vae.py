import numpy as np
import tensorflow as tf
from model import*
from keras.optimizers import Adam
from preprocess_data import*
from losses import KL_loss
import os
from datetime import datetime
import keras 

Data_dir = "/media/nazib/E20A2DB70A2D899D/Ubuntu_desktop/Travello/RawData/new_feed_data/"

#user_data = prep_user_interaction()

if not os.path.exists("logs"):
    os.mkdir("logs")
    
logdir = "logs/" + datetime.now().strftime("%Y%m%d-%H%M%S_noisy")
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)

data_file = "AllFeedData.csv"
data = create_training_data(Data_dir,data_file)
N,M = data.shape
train_ratio = N*80//100
valid_ratio = N*10//100
test_ratio =  N*10//100


label_data = data[0:train_ratio,:]
x, y = label_data.shape
###### Adding Gaussian Noise to the the data ###
train_data = label_data + np.random.normal(0,1,(x,y))

valid_data = data[train_ratio+1:train_ratio+valid_ratio,]

VEA = vae_model([512,256,128,64,32,16,8])
VEA_model, z_mean,z_log_var = VEA.create_model()
VEA_model.summary()

VEA_model.compile(optimizer='SGD', loss=[ 'mean_squared_logarithmic_error',KL_loss(z_mean,z_log_var)],
loss_weights=[1,0.5],metrics=['accuracy'])

np.random.shuffle(train_data)
np.random.shuffle(valid_data)

VEA_model.fit(train_data,[label_data,label_data],
                validation_data=(valid_data, [valid_data,valid_data]),
                epochs=100,
                batch_size=500, callbacks=[tensorboard_callback])
VEA_model.save(logdir+"/VAE_noisy.h5")

#loss,acc = VEA_model.evaluate(test_data, [test_data,test_data], verbose=2)
#print("Restored model, accuracy: {:5.2f}%".format(100*acc))








