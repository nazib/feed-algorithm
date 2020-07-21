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

def ranking(ranks, names, order=1):
    minmax = MinMaxScaler()
    ranks = minmax.fit_transform(order*np.array([ranks]).T).T[0]
    ranks = map(lambda x: round(x,2), ranks)
    return dict(zip(names, ranks))

#pro_data = pd.read_csv("/media/nazib/E20A2DB70A2D899D/Ubuntu_desktop/Travello/RawData/processed_feed_data10K.csv")
pro_data =  pd.read_csv("/media/nazib/E20A2DB70A2D899D/Ubuntu_desktop/Travello/RawData/new_feed_data/AllFeedData.csv")
pro_data.drop('uid',axis=1,inplace=True)
pro_data.drop('ptid',axis=1,inplace=True)
pro_data.drop('feed_id',axis=1,inplace=True)

###########################################################################
#              Clustering Data To appropriate Lebel generation            #
###########################################################################
N,M = pro_data.shape
n_components = N//2
random_state = M//2
labels = np.random.randint(1,10,size=(N,1))

data = pro_data.values
data = np.concatenate((data,labels),axis=1)
print(data[0:5,-1])

####### Adding clustered labels to fit the regression model #########
#gmm = GMM(10)
#new_labels = gmm.fit_predict(data)
#pro_data["Labels"] = new_labels

train_ratio = N*90//100
test_ratio = N*10//100

label = 'likes'
train_data = pro_data.loc[0:train_ratio,pro_data.columns]
train_label = train_data[label].values
train_data.drop(label,axis=1,inplace=True)
train_data = train_data.values

test_data = pro_data.loc[train_ratio+1:N,pro_data.columns]
test_label = test_data[label].values
test_data.drop([label],axis=1,inplace=True)
test_data = test_data.values
#columns = train_data.columns

###########################################################################
#                   Applying Linear Models                                #
###########################################################################
rlasso = Lasso(alpha=0.4)
rlasso.fit(train_data,train_label)
rlasso_pred = rlasso.predict(test_data)
rlasso_pfm = distance.euclidean(test_label,rlasso_pred)

### Applying Linear Regression ######
Lr = LinearRegression(normalize=True)
Lr.fit(train_data,train_label)
Lr_pred  = Lr.predict(test_data)
Lr_pfm = distance.euclidean(Lr_pred,test_label)

rfe = RFE(Lr,n_features_to_select=1,verbose=3)
rfe.fit(train_data,train_label)
rfe_pred = rfe.predict(test_data)
rfe_pfm = distance.euclidean(rfe_pred,test_label)

rdg = Ridge(alpha=0.04)
rdg.fit(train_data,train_label)
rdg_pred = rdg.predict(test_data)
rdg_pfm = distance.euclidean(rdg_pred,test_label)

RF = RandomForestRegressor(n_jobs=-1,n_estimators=50,verbose=3)
RF.fit(train_data,train_label)
RF_pred = RF.predict(test_data)
RF_pfm = distance.euclidean(RF_pred,test_label)
print(RF.feature_importances_)
rank = {'RLesso':rlasso_pfm,'Linear Regression':Lr_pfm,'RFE':rfe_pfm,'RDG':rdg_pfm,'RandomForrest':RF_pfm}
rank= sorted(rank.items(), key=lambda x: x[1])
print(rank)

'''
sns.factorplot(x="Mean Ranking", y="Feature", data = rank, kind="bar", 
               size=14, aspect=1.9, palette='coolwarm')
plt.savefig('AlgRank.png')
'''
###########################################################################
#                        Global Feed Ranking Logic                        #
###########################################################################
TH= 60*36
values = test_data[:,2]
decay = RF.feature_importances_[2]*np.exp(1-(values/TH))

rank_feat = np.zeros(shape=(test_data.shape[0],5),dtype=float)
#rank_feat[:,0] = test_data[:,0]
rank_feat[:,1] = test_data[:,1]
rank_feat[:,2] = test_data[:,3]
rank_feat[:,3] = test_data[:,4]
rank_feat[:,4] = test_data[:,5]

theta = RF.feature_importances_
theta = [theta[0],theta[1],theta[3],theta[4],theta[5]]
theta = mt.repmat(theta,test_data.shape[0],1)
feed_rank = np.zeros(shape=(theta.shape[0],1),dtype=float)

for i in range(1,theta.shape[0]):
    feed_rank[i] = np.sum(rank_feat[i,:] * theta[i,:] *decay[i])

#feed_rank = np.sort(feed_rank,axis=0)

#ranked_feed = pd.read_csv("/media/nazib/E20A2DB70A2D899D/Ubuntu_desktop/Travello/RawData/processed_feed_data10K.csv")
ranked_feed =  pd.read_csv("/media/nazib/E20A2DB70A2D899D/Ubuntu_desktop/Travello/RawData/new_feed_data/AllFeedData.csv")
ranked_feed = ranked_feed.loc[train_ratio+1:N,ranked_feed.columns]
ranked_feed['RankScore'] = feed_rank
ranked_feed.sort_values(by=ranked_feed.columns.values[-1], ascending=False, inplace=True)
#ranked_feed.to_csv("/media/nazib/E20A2DB70A2D899D/Ubuntu_desktop/Travello/RawData/new_feed_data/Global_Rank_all_feed.csv",index=False)

###########################################################################
#                       Personalised Feed Ranking Logic                  #
###########################################################################
#user_data = pd.read_csv("/media/nazib/E20A2DB70A2D899D/Ubuntu_desktop/Travello/RawData/Processed_User_data.csv")
#user_data = pd.read_csv("/media/nazib/E20A2DB70A2D899D/Ubuntu_desktop/Travello/RawData/new_feed_interaction/All_User_data.csv")

### Processing User interaction data
user_data = prep_user_interaction()
### Extracting user info of Ranked Feed ###
ranked_user_info = extract(user_data,ranked_feed)

### Selecting a Random User for Ranking ###
select = np.random.randint(user_data.shape[0])
values = user_data.iloc[select].values
values = np.expand_dims(values,axis=0)
selected_uid = pd.DataFrame(data=values, columns=user_data.columns)

### One-to-many Adjecency Matrix  from the Global Ranked data###
user_weights = pd.DataFrame(data=np.zeros(shape=(ranked_feed.shape[0],selected_uid.shape[1])),columns=user_data.columns)
personal_rank = Personalized_ranks(ranked_feed, ranked_user_info,user_data, selected_uid)

ranked_feed["Personalized Rank"] = personal_rank
ranked_feed.sort_values(by=ranked_feed.columns.values[-1], ascending=False, inplace=True)
ranked_feed.to_csv("/home/nazib/Desktop/Travello/RawData/Personal_Ranked_feed_10k.csv",index=False)







































