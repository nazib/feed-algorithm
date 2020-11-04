import numpy as np
import tensorflow as tf
from datetime import datetime
import keras
import pandas as pd
import os
from sklearn import preprocessing
from model.vae_model import vae_model
from model.preprocess_data import create_training_data
from model.losses import KL_loss
from model.utils import calculate_weight, get_global_model_load_path, get_global_model_save_path, similarity
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


class NonLinearModel(vae_model):
    def __init__(self):
        self.Model = vae_model([512, 256, 128, 64, 32, 16, 8])
        self.Model, self.z_mean, self.z_log_var = self.Model.create_model()
        self.Model.compile(
            optimizer='SGD', loss=['mean_squared_logarithmic_error', KL_loss(self.z_mean, self.z_log_var)],
            loss_weights=[1, 0.5], metrics=['accuracy'])

        model_path = get_global_model_load_path()
        self.model_path = model_path

        # Defini Lebel Encoders for Country, City and StatusLevel. Only Statuslevel is implemented
        # Other two will implemented in future ###
        self.Enc_level = preprocessing.LabelEncoder()
        labels = ['Rookie', 'Nomad', 'Explorer', 'Expert', 'Guru']
        self.Enc_level.fit(labels)

        self.Enc_gender = preprocessing.LabelEncoder()
        labels = ['Male', 'Female', 'male', 'female',
                  'other', 'Non-binary', 'WITHHELD']
        self.Enc_gender.fit(labels)
        
        ### Label Encoders for Interets ####   
        self.Enc_interests = preprocessing.LabelEncoder()
        labels = pd.read_csv(os.getcwd()+'/base_data/Data/interests.tsv',sep='\t')
        labels = labels['object_id']
        self.Enc_interests.fit(labels)

        ### Label Encoders for Groups #### 
        self.Enc_groups = preprocessing.LabelEncoder()
        glabels = pd.read_csv(os.getcwd()+'/base_data/Data/public_active_groups.tsv',sep='\t')
        glabels = glabels['name']
        self.Enc_groups.fit(glabels)

        if not bool(model_path):
            self.istrained = False
        else:
            #### Getting Saved model directories and selecting most rect one ####
            # dirs = [path + "/" + k + "/" for k in dirs]
            #times = [os.path.getmtime(k) for k in dirs]
            '''
            for i in range(len(dirs)):
                dir_dict[times[i]] = dirs[i]
            '''
            self.Model.load_weights(model_path)
            mu = self.Model.get_layer('mu')
            mu_wgt = mu.get_weights()[1]
            var = self.Model.get_layer('var')
            var_wgt = var.get_weights()[1]
            ### Generating Weights parameters from Distribution ####
            self.coefficients = mu_wgt + var_wgt
            self.istrained = True

    def fit(self, Data_dir, data_file):
        (logdir, model_path) = get_global_model_save_path()
        tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)
        data = create_training_data(Data_dir, data_file)
        N, M = data.shape
        train_ratio = N*90//100
        valid_ratio = N*10//100

        label_data = data[0:train_ratio, :]
        x, y = label_data.shape
        ###### Adding Gaussian Noise to the the data ###
        train_data = label_data + np.random.normal(0, 1, (x, y))

        valid_data = data[train_ratio+1:train_ratio+valid_ratio, ]
        np.random.shuffle(train_data)
        np.random.shuffle(valid_data)

        self.Model.fit(
            train_data, [label_data, label_data],
            validation_data=(valid_data, [valid_data, valid_data]),
            epochs=100,
            batch_size=500, callbacks=[tensorboard_callback])
        self.Model.save(model_path)
        self.istrained = True
        return "Non Linear Model Trained Successfully"

    def GlobalRank(self, feed_data):
        #TH= 60*36
        output = {}

        for i in range(len(feed_data)):
            TH = 3.0 * 24*3600  # 3 days in seconds
            post_date = datetime.strptime(
                feed_data[i]["postedDate"], "%Y-%m-%d %H:%M:%S")
            curr_date = datetime.now()
            #post_age = float((curr_date - post_date).days)
            post_age = float((curr_date - post_date).total_seconds())
            decay = self.coefficients[2]*np.exp(1-(post_age/TH))

            ########## Imposing Gaussian decay on text length for proper balence ##########
            text_TH = 200
            text_sigma = 300
            text_len = feed_data[i]["postTextLength"]
            text_decay = np.exp(-np.power(text_len - text_TH,
                                          2.) / (2 * np.power(text_sigma, 2.)))
            ########## Imposing Gaussian decay on hashtags for proper balence ##########
            h_TH = 5
            h_sigma = 5
            hash = feed_data[i]["numberOfHashTags"]
            h_decay = np.exp(-np.power(hash - h_TH, 2.) /
                             (2 * np.power(h_sigma, 2.)))

            weights = np.array(
                [
                    self.coefficients[0],
                    self.coefficients[1],
                    self.coefficients[3],
                    self.coefficients[4],
                    self.coefficients[5],
                    self.coefficients[6],
                    self.coefficients[7]
                ])
            data = np.array(
                [
                    feed_data[i]["numberOfLikes"],
                    feed_data[i]["numberOfComments"],
                    feed_data[i]["postTextLength"],
                    feed_data[i]["numberOfHashTags"],
                    feed_data[i]["latitude"],
                    feed_data[i]["longitude"],
                    feed_data[i]["numberOfMediaUrls"],
                ]
            )
            output[i] = {}
            feed_data[i]["globalRank"] = np.sum(
                data*weights*decay*text_decay*h_decay)
            output[i]['feedItemId'] = feed_data[i]['feedItemId']
            output[i]['global'] = feed_data[i]['globalRank']
        return feed_data, output

    def PersonalRank(self, data):
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
            user_gender = self.Enc_gender.transform(
                np.array([user_data['gender']]))
            poster_gender = self.Enc_gender.transform(
                np.array([poster_data['gender']]))

            if user_gender == poster_gender:
                user_weights[i]['gender'] = 1.0
            else:
                user_weights[i]['gender'] = 0.0

            user_weights[i]['totalReceivedPostComments'] = calculate_weight(
                user_data["totalReceivedPostComments"], poster_data["totalReceivedPostComments"])

            ### Like weight ###
            user_weights[i]['totalReceivedPostLikes'] = calculate_weight(
                user_data["totalReceivedPostLikes"], poster_data["totalReceivedPostLikes"])

            ### Follower weight ###
            user_weights[i]['numberOfFollowers'] = calculate_weight(
                user_data["numberOfFollowers"], poster_data["numberOfFollowers"])

            ### Status Level weight ###
            user_level = self.Enc_level.transform(
                np.array([user_data['statusLevel']]))
            poster_level = self.Enc_level.transform(
                np.array([poster_data['statusLevel']]))
            user_weights[i]['statusLevel'] = calculate_weight(
                user_level, poster_level)
            
            ### Interests weight ####
            user_interests = self.Enc_interests.transform(np.array(user_data["interests"]))
            poster_interests = self.Enc_interests.transform(np.array(poster_data["interests"]))
            interest_similarity = similarity(user_interests,poster_interests)
            
            ### Group Weights ####
            user_groups = self.Enc_groups.transform(np.array(user_data["Groups"]))
            poster_groups = self.Enc_groups.transform(np.array(poster_data["Groups"]))
            group_similarity = similarity(user_groups,poster_groups)
            
            #### creating Feature Array ###
            #user_feature = np.array(list(user_data.values()), dtype=float)
            user_feature = np.array(
                [
                    user_gender,
                    user_data['totalReceivedPostComments'],
                    user_data['totalReceivedPostLikes'],
                    user_data["numberOfFollowers"],
                    user_level
                ], dtype=float
            )

            ### Creating Weights Array ###
            #weights = np.array(list(user_weights.values()),dtype=float)
            weights = np.array(
                [
                    user_weights[i]["gender"],
                    user_weights[i]['totalReceivedPostComments'],
                    user_weights[i]['totalReceivedPostLikes'],
                    user_weights[i]["numberOfFollowers"],
                    user_weights[i]['statusLevel']], dtype=float
            )
            personal_rank = np.sum(
                user_feature * weights * feed_data[i]['globalRank'])+ interest_similarity + group_similarity

            output[i]['feedItemId'] = feed_data[i]['feedItemId']
            output[i]['personalised'] = personal_rank
            output[i]['global'] = feed_data[i]['globalRank']

        return output
