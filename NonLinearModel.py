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
import time
from sklearn import preprocessing
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
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

        path = os.path.abspath(os.getcwd()+"/logs/")
        dirs = os.listdir(path)
        dir_dict ={}

        ### Defini Lebel Encoders for Country, City and StatusLevel. Only Statuslevel is implemented
        # Other two will implemented in future ###
        self.Enc_level = preprocessing.LabelEncoder()
        labels = ['Rookie','Nomad','Explorer','Expert','Guru']
        self.Enc_level.fit(labels)
        
        self.Enc_gender = preprocessing.LabelEncoder()
        labels = ['Male','Female']
        self.Enc_gender.fit(labels)
        
        if len(dirs) == 0:
            self.istrained = False
        else:
            #### Getting Saved model directories and selecting most rect one ####
            dirs = [path +"/"+ k +"/" for k in dirs]
            #times = [os.path.getmtime(k) for k in dirs]
            '''
            for i in range(len(dirs)):
                dir_dict[times[i]] = dirs[i]
            '''
            self.model_dir = dirs[len(dirs)-1]
            self.Model.load_weights(self.model_dir+"VAE_noisy.h5")
            mu = self.Model.get_layer('mu')
            mu_wgt = mu.get_weights()[1]
            var = self.Model.get_layer('var')
            var_wgt = var.get_weights()[1]
            ### Generating Weights parameters from Distribution ####
            self.coefficients = mu_wgt + var_wgt
            self.istrained = True
    
    def fit(self, Data_dir, data_file):
        #Data_dir = "/media/nazib/E20A2DB70A2D899D/Ubuntu_desktop/Travello/RawData/new_feed_data/"
        #user_data = prep_user_interaction()
        if not os.path.exists("logs"):
            os.mkdir("logs")
            
        logdir = "logs/" + datetime.now().strftime("%Y%m%d-%H%M%S_noisy")
        tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)
        data = create_training_data(Data_dir,data_file)
        N,M = data.shape
        train_ratio = N*90//100
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
        self.istrained = True
        return "Non Linear Model Trained Successfully"
    
    def calculate_weight(self, sid,pid):
        if sid==0.0 and pid ==0.0:
            return 0
        else:
            return np.exp((sid-pid)/(sid+pid)) 

    def GlobalRank(self,feed_data):
        #TH= 60*36
        output = {}

        for i in range(len(feed_data)):
            TH = 3.0
            post_date =datetime.strptime(feed_data[i]["postedDate"], "%Y-%m-%d %H:%M:%S")
            curr_date = datetime.now()
            post_age = float((curr_date - post_date).days)
            decay = self.coefficients[2]*np.exp(1-(post_age/TH))

            weights = np.array([self.coefficients[0],
            self.coefficients[1],
            self.coefficients[3],
            self.coefficients[4],
            self.coefficients[5],
            self.coefficients[6],
            self.coefficients[7]
            ])            
            data = np.array([feed_data[i]["numberOfLikes"],
            feed_data[i]["numberOfComments"],
            feed_data[i]["postTextLength"],
            feed_data[i]["numberOfHashTags"],
            feed_data[i]["latitude"],
            feed_data[i]["longitude"],
            feed_data[i]["numberOfMediaUrls"],]
            )
            output[i] = {}
            feed_data[i]["globalRank"] = np.sum(data*weights*decay)
            output[i]['feedItemId'] = feed_data[i]['feedItemId']
            output[i]['global'] = feed_data[i]['globalRank']
        return feed_data, output

    def PersonalRank(self,data):
        feed_data = data["feedItems"]
        feed_data, _ = self.GlobalRank(feed_data)

        user_data = data["userAttributes"]
        user_weights = {}
        output = {}

        for i in range(len(feed_data)):
            user_weights[i] = {}
            output[i] = {}

            poster_data = feed_data[i]['posterAttributes']
            '''
            if user_data['UserCity'] == poster_data['PosterCity']:
                user_weights['city'] = 1.0
            else:
                user_weights['city'] = 0.0
            ### Country weight ###
            if user_data['UserCountry'] == poster_data['PosterCountry']:
                user_weights['country'] = 1.0
            else:
                user_weights['country'] = 0.0
            '''
            ### Gender weight ###
            user_gender = self.Enc_gender.transform(np.array([user_data['gender']]))
            poster_gender =self.Enc_gender.transform(np.array([poster_data['gender']]))
            
            if user_gender == poster_gender:
                user_weights[i]['gender'] = 1.0
            else:
                user_weights[i]['gender'] = 0.0
            
            user_weights[i]['totalReceivedPostComments'] = self.calculate_weight(user_data["totalReceivedPostComments"],poster_data["totalReceivedPostComments"])
            
            ### Like weight ###
            user_weights[i]['totalReceivedPostLikes'] = self.calculate_weight(user_data["totalReceivedPostLikes"],poster_data["totalReceivedPostLikes"])

            ### Follower weight ###
            user_weights[i]['numberOfFollowers'] = self.calculate_weight(user_data["numberOfFollowers"],poster_data["numberOfFollowers"])
            
            ### Status Level weight ###
            user_level = self.Enc_level.transform(np.array([user_data['statusLevel']]))
            poster_level =self.Enc_level.transform(np.array([poster_data['statusLevel']]))
            user_weights[i]['statusLevel'] = self.calculate_weight(user_level,poster_level)

            #### creating Feature Array ###
            #user_feature = np.array(list(user_data.values()), dtype=float)
            user_feature = np.array([user_gender,
            user_data['totalReceivedPostComments'],
            user_data['totalReceivedPostLikes'],
            user_data["numberOfFollowers"],
            user_level], dtype=float
            )

            ### Creating Weights Array ###
            #weights = np.array(list(user_weights.values()),dtype=float)
            weights = np.array([user_weights[i]["gender"],
            user_weights[i]['totalReceivedPostComments'],
            user_weights[i]['totalReceivedPostLikes'],
            user_weights[i]["numberOfFollowers"],
            user_weights[i]['statusLevel']], dtype=float
            )
            personal_rank = np.sum(user_feature * weights * feed_data[i]['globalRank'])

            output[i]['feedItemId'] = feed_data[i]['feedItemId']
            output[i]['personalised'] = personal_rank
            output[i]['global'] = feed_data[i]['globalRank']
        
        return output


if __name__ == "__main__":
    obj = NonLinearModel()
    
    #### Train ####
    #obj.fit("/media/nazib/E20A2DB70A2D899D/Ubuntu_desktop/Travello/RawData/new_feed_data/","AllFeedData.csv")
    #### Test #####
    data = pd.read_csv("/media/nazib/E20A2DB70A2D899D/Ubuntu_desktop/Travello/RawData/new_feed_data/AllFeedData.csv")
    cols = data.columns 
    values = data[cols[3:]].values
    glb_ranks = obj.Rank(values)
    data["GlobalRanks"] = glb_ranks
    data.sort_values(by=data.columns.values[-1], ascending=False, inplace=True)
    data.to_csv("NonLinearRank.csv")





                
