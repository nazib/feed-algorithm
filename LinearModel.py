#!/usr/bin/env python3
import os 
import numpy as np
import pandas as pd
import pygeohash as pg
import math
from scipy.spatial import distance
from numpy import matlib as mt
from matplotlib import pyplot as plt
from sklearn.feature_selection import RFE, f_regression
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import MinMaxScaler
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor
from sklearn.mixture import GaussianMixture as GMM
from sklearn.mixture import BayesianGaussianMixture as BGMM
from preprocess_data import*
from rank_logics import*
import argparse
import sys
from datetime import datetime

class LinearModel:
    def __init__(self, model_type, lr):
        self.model_type = model_type

        if self.model_type == "Lesso":
            self.Model = Lasso(alpha=lr)
        elif self.model_type == "LinearRegression":
            self.Model = LinearRegression(normalize=True)
        elif self.model_type == "Ridge":
            self.Model = Ridge(alpha=lr)
        elif self.model_type == "RandomForrest":
            self.Model = RandomForestRegressor(n_jobs=-1,n_estimators=50,verbose=3)
        else:
            print("Model type not mached")
        self.isdata = False

        if os.path.isfile("LinearModel.dat"):
            self.istrained = True
            self.coefficients = np.fromfile("LinearModel.dat")
            self.Enc_level = preprocessing.LabelEncoder()
            labels = ['Rookie','Nomad','Explorer','Expert','Guru']
            self.Enc_level.fit(labels)
            self.Enc_gender = preprocessing.LabelEncoder()
            labels = ['Male','Female']
            self.Enc_gender.fit(labels)
        else:
            self.istrained = False
            self.coefficients = None

        
    def prepare_feed_data(self,file_name, training_ratio, data_label):
        data = pd.read_csv(file_name)
        cols = data.columns
        data_array = data[cols[3:]]
        N,M = data_array.shape
        train_index = N*training_ratio//100

        train_data = data_array.loc[0:train_index,data_array.columns]
        train_label = train_data[data_label].values
        train_data.drop(data_label,axis=1,inplace=True)
        train_data = train_data.values
        self.train_data = train_data
        self.train_label = train_label
        self.isdata = True
    
    def fit(self):
        if self.isdata == False:
            #print("Please Prepare data for training by calling 'prepare_feed_data' function")
            return "Please Prepare data for training by calling 'prepare_feed_data' function"
        else:
            self.Model.fit(self.train_data,self.train_label)
            self.istrained = True

            if self.model_type == "Lesso":
                self.coefficients = self.Model.coef_
            elif self.model_type == "LinearRegression":
                self.coefficients = self.Model.coef_
            elif self.model_type == "Ridge":
                self.coefficients = self.Model.coef_
            elif self.model_type == "RandomForrest":
                self.coefficients = self.Model.feature_importances_
            else:
                print("Model is not trained properly")
            
            self.coefficients.tofile("LinearModel.dat")
            return "Model Trained Successfully"
    
    def calculate_weight(self, sid,pid):
        if sid==0.0 and pid ==0.0:
            return 0
        else:
            return np.exp((sid-pid)/(sid+pid))
    
    def BulkRank(self, data):
        if self.istrained == False:
            print("Train the model first")
        else:
            rank_score = GlobalRank(data, self.coefficients)
            return rank_score
    
    def GlobalRank(self, feed_data):
        if self.istrained == False:
            return "Model Not Trained"
        else:
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
                ])            
                data = np.array([
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
        '''
        user_data = data["UserData"]
        poster_data = data["PosterData"]
        user_weights = {}

        if user_data['UserCity'] == poster_data['PosterCity']:
            user_weights['city'] = 1.0
        else:
            user_weights['city'] = 0.0
        ### Country weight ###
        if user_data['UserCountry'] == poster_data['PosterCountry']:
            user_weights['country'] = 1.0
        else:
            user_weights['country'] = 0.0
        ### Gender weight ###
        if user_data["UserGender"] == poster_data['PosterGender']:
            user_weights['gender'] = 1.0
        else:
            user_weights['gender'] = 0.0
        
        user_weights['total_comments'] = self.calculate_weight(user_data["UserTotalComments"],poster_data["PosterTotalComments"])
        
        ### Like weight ###
        user_weights['total_likes'] = self.calculate_weight(user_data["UserTotalLikes"],poster_data["PosterTotalLikes"])

        ### Follower weight ###
        user_weights['total_followers'] = self.calculate_weight(user_data["UserNumberOfFollowers"],poster_data["PosterNumberOfFollowers"])
        ### Status Level weight ###
        user_weights['level'] = self.calculate_weight(user_data["UserStatusLevel"],poster_data["PosterStatusLevel"])

        user_feature = np.array(list(user_data.values()), dtype=float)
        weights = np.array(list(user_weights.values()),dtype=float)
        personal_rank = np.sum(user_feature * weights * global_rank) 
        return personal_rank, global_rank
        '''
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
'''
if __name__== "__main__":
    
    main_parser = argparse.ArgumentParser(prog='LinearModel')
    #main_parser.add_argument("-opt","--Option",help="Training the model")
    #main_parser.add_argument("-ts","--Test",help="Testing the model")

    parser = main_parser.add_subparsers(dest="command")
    train_parser = parser.add_parser("train")
    train_parser.add_argument("-F","--FileName",type=str, help="Training data file in csv format")
    train_parser.add_argument("-R","--Training-ratio",type=int,help="Parcentages of Rows in csv file to be used for training")
    train_parser.add_argument("-Lab","--Label-name",type=str,help="Give the column name that is to be used as label data")
    test_parser = parser.add_parser("test") 
    test_parser.add_argument('-l','--Likes',type=float,help="Number of Likes")
    
    test_parser.add_argument('-c','--Comments',type=float,help="Number of Comments")
    test_parser.add_argument('-a','--Age',type=float,help="Number of Minutes passed after posting")
    test_parser.add_argument('-pl','--PostLength',type=float,help="Number of charecters in the post")
    test_parser.add_argument('-ha','--Hash',type=float,help="Number of Hashtags in the post")
    test_parser.add_argument('-la','--Latitude',type=float,help="Latitude of the location of the post")
    test_parser.add_argument('-lo','--Longitude',type=float,help="Longitude of the location of the post")
    test_parser.add_argument('-u','--Url',type=float,help="Number of web links or urls in the post")
    
    #command = "train -F /media/nazib/E20A2DB70A2D899D/Ubuntu_desktop/Travello/RawData/new_feed_data/AllFeedData.csv -R 90 -Lab likes"
    command = "test -l 100 -c 10 -a 1200 -pl 200 -ha 2 -la 43.9 -lo 23.33 -u 3"
    args = main_parser.parse_args(command.split())
    obj = LinearModel("RandomForrest", 0.04)

    if args.command == "train":    
        obj.prepare_feed_data(args.FileName,args.Training_ratio,args.Label_name)
        obj.fit()
        print("Model Trained")
    elif args.command == "test":
        data = np.zeros((7),dtype=float)        
        data[0] = args.Comments
        data[1] = args.Age
        data[2] = args.PostLength
        data[3] = args.Hash
        data[4] = args.Latitude
        data[5] = args.Longitude
        data[6] = args.Url
        obj.Rank(data)

'''    

        

        




        


