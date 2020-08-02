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
            print("Please Prepare data for training by calling 'prepare_feed_data' function")
            return 1
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
    
    def BulkRank(self, data):
        if self.istrained == False:
            print("Train the model first")
        else:
            rank_score = GlobalRank(data, self.coefficients)
            return rank_score
    
    def Rank(self, data):
        if self.istrained == False:
            print("Train the model first")
        else:
            TH= 60*36
            values = data[1]
            decay = self.coefficients[2]*np.exp(1-(values/TH))
            weights = np.array([self.coefficients[0],self.coefficients[2],self.coefficients[3],self.coefficients[4],self.coefficients[5]])            
            data = np.array([data[0],data[2],data[3],data[4],data[5]])
            feed_rank = np.sum(data*weights*decay)
            return feed_rank


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

    

        

        




        


