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
            return 0
    
    def Rank(self, data):
        if self.istrained == False:
            print("Train the model first")
        else:
            rank_score = GlobalRank(data, self.coefficients)
            return rank_score


if __name__== "__main__":
    obj = LinearModel("RandomForrest", 0.04)
    data_file = "/media/nazib/E20A2DB70A2D899D/Ubuntu_desktop/Travello/RawData/new_feed_data/AllFeedData.csv"
    obj.prepare_feed_data(data_file,90,'likes')
    obj.fit()
    rank_score = obj.Rank(obj.train_data)

        

        




        


