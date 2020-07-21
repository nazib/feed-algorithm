import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.models import Model
from keras.layers import Conv1D, Activation, Input, UpSampling1D, Softmax
from keras.layers import LeakyReLU, Reshape, Lambda, BatchNormalization, Dense, Flatten, ReLU,Dropout 
from preprocess_data import*
from losses import KL_loss
import os
from datetime import datetime
import keras
from rank_logics import*
import pandas as pd

class vae_model:
    def __init__(self, latent_layers):
        self.latenet_layers = latent_layers
        print("Model initialized")

    def sampling(self,args):
        z_mean, z_log_var = args
        batch = K.shape(z_mean)[0]
        dim = K.int_shape(z_mean)[1]
        # by default, random_normal has mean = 0 and std = 1.0
        epsilon = K.random_normal(shape=(batch, dim))
        return z_mean + K.exp(0.5 * z_log_var) * epsilon
    
    def myConv(self,x_in, nf, strides=1,rate=1):
        x_out = Conv1D(nf, kernel_size=3, padding='same',kernel_initializer='he_normal', strides=strides)(x_in)
        x_out = LeakyReLU(0.2)(x_out)
        return x_out
    
    def encoder(self):
        inp = Input(shape=(8,),name='input')
        x = Dense(self.latenet_layers.pop(0),activation='relu')(inp)
        for filters in self.latenet_layers:
            x = Dense(filters,activation='relu')(x)
            x = BatchNormalization()(x)
            #x = Dropout(0.2)(x)
        
        z_mu = Dense(8,activation='relu', name='mu')(x)
        z_var = Dense(8,activation='relu', name= "var")(x)

        z = Lambda(self.sampling, output_shape=(8,), name='z')([z_mu, z_var])
        self.latent_rep = z
        self.mean = z_mu
        self.logvar = z_var
        model = Model(inputs=inp, outputs=z, name='Encoder')
        #model.summary()
        return model
    
    def decoder(self):
        inp = Input(shape=(8,),name='input')
        x = Dense(self.latenet_layers[-1],activation='relu')(inp)

        for filters in reversed(self.latenet_layers):
            x = Dense(filters,activation='relu')(x)
            x = BatchNormalization()(x)
            #x = Dropout(0.2)(x)

        output = Dense(8,activation='relu')(x)
        #output = Softmax()(output)
        model = Model(inputs=inp, outputs=output,name='Decoder')
        #model.summary()
        return model
    
    def create_model(self):
        encode = self.encoder()
        decode = self.decoder()
        model = Model(inputs=encode.input, outputs=[decode(encode.output), self.latent_rep])
        return model, self.mean, self.logvar

class NonLinearModel(vae_model):
    def __init__(self):
        self.Model = vae_model([512,256,128,64,32,16,8])
        self.Model, self.z_mean,self.z_log_var = self.Model.create_model()
        self.Model.compile(optimizer='SGD', loss=[ 'mean_squared_logarithmic_error',KL_loss(self.z_mean,self.z_log_var)],
        loss_weights=[1,0.5],metrics=['accuracy'])
    
    def fit(self, Data_dir, data_file):
        #Data_dir = "/media/nazib/E20A2DB70A2D899D/Ubuntu_desktop/Travello/RawData/new_feed_data/"
        #user_data = prep_user_interaction()
        if not os.path.exists("logs"):
            os.mkdir("logs")
            
        logdir = "logs/" + datetime.now().strftime("%Y%m%d-%H%M%S_noisy")
        tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)
        data = create_training_data(Data_dir,data_file)
        N,M = data.shape
        train_ratio = N*80//100
        valid_ratio = N*10//100

        label_data = data[0:train_ratio,:]
        x, y = label_data.shape
        ###### Adding Gaussian Noise to the the data ###
        train_data = label_data + np.random.normal(0,1,(x,y))

        valid_data = data[train_ratio+1:train_ratio+valid_ratio,]
        np.random.shuffle(train_data)
        np.random.shuffle(valid_data)

        self.Model.fit(train_data,[label_data,label_data],
                        validation_data=(valid_data, [valid_data,valid_data]),
                        epochs=100,
                        batch_size=500, callbacks=[tensorboard_callback])
        self.Model.save(logdir+"/VAE_noisy.h5")
        return 0

    def Rank(self,data):
        model_dir = os.path.abspath(os.getcwd()+"/logs/20200629-155817_noisy/")
        #model = vae_model([512,256,128,64,32,16,8])
        #model,_,_ = model.create_model()
        self.Model.load_weights(model_dir+"/VAE_noisy.h5")
        mu = self.Model.get_layer('mu')
        mu_wgt = mu.get_weights()[1]
        var = self.Model.get_layer('var')
        var_wgt = var.get_weights()[1]

        ### Generating Weights parameters from Distribution ####
        theta = mu_wgt + var_wgt
        global_rank = GlobalRank(data, theta)
        #global_rank.to_csv("Global_Rank_by NN_noisy500.csv",index=False)
        #print("Processed")
        return global_rank


if __name__ == "__main__":
    obj = NonLinearModel()
    
    #### Train ####
    obj.fit("/media/nazib/E20A2DB70A2D899D/Ubuntu_desktop/Travello/RawData/new_feed_data/","AllFeedData.csv")
    #### Test #####
    data = pd.read_csv("/media/nazib/E20A2DB70A2D899D/Ubuntu_desktop/Travello/RawData/new_feed_data/AllFeedData.csv")
    cols = data.columns 
    values = data[cols[3:]].values
    glb_ranks = obj.predict(values)
    data["GlobalRanks"] = glb_ranks
    data.sort_values(by=data.columns.values[-1], ascending=False, inplace=True)
    data.to_csv("NonLinearRank.csv")





                
