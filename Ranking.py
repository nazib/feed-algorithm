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

def ranking(ranks, names, order=1):
    minmax = MinMaxScaler()
    ranks = minmax.fit_transform(order*np.array([ranks]).T).T[0]
    ranks = map(lambda x: round(x,2), ranks)
    return dict(zip(names, ranks))

def remove_tab(data):
    for x in data.columns:
        cols = []
        for i in range(data[x].values.shape[0]):
            if data[x][i] == '\t':
                cols.insert(i,"0")
            else:
                cols.insert(i,str(data[x][i]).replace("\t",""))
        data[x]=cols
    return data   

def extract(user, ranked):

    ext_data = pd.DataFrame(columns=user_data.columns)

    for i in range(ranked.shape[0]):
        person = user[user['user_id']==ranked['uid'][i]].values
        ext_data.loc[i,:] = person
    
    return ext_data

#pro_data = pd.read_csv("/home/nazib/Desktop/Travello/RawData/processed_feed_data10K.csv")
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

ranked_feed = pd.read_csv("/media/nazib/E20A2DB70A2D899D/Ubuntu_desktop/Travello/RawData/new_feed_data/AllFeedData.csv")
ranked_feed = ranked_feed.loc[train_ratio+1:N,ranked_feed.columns]
ranked_feed['RankScore'] = feed_rank
ranked_feed.sort_values(by=ranked_feed.columns.values[-1], ascending=False, inplace=True)
ranked_feed.to_csv("/media/nazib/E20A2DB70A2D899D/Ubuntu_desktop/Travello/RawData/new_feed_data/Global_Rank_all_feed.csv",index=False)

###########################################################################
#                       Personalised Feed Ranking Logic                  #
###########################################################################
user_data = pd.read_csv("/home/nazib/Desktop/Travello/RawData/Cohort_Users_For_Feed.csv")
x = []
for i in range(len(user_data.columns)):
    x.insert(i,str(user_data.columns[i]).replace("\t",""))

user_data.columns = x
user_data = user_data[['user_id','city','country','gp:num_followers',
'gp:total_comments','gp:total_likes','gp:tv_gender','gp:status_level']]

user_data = remove_tab(user_data)

text_data = pd.DataFrame()
text_data["city"] = user_data["city"]
text_data["country"] = user_data["country"]
text_data["gp:tv_gender"] = user_data["gp:tv_gender"]
text_data["gp:status_level"] = user_data["gp:status_level"]
#text_data["gp:tv_groups"] = user_data["gp:tv_groups"]

Enc = preprocessing.LabelEncoder()
Enc_text = text_data.apply(Enc.fit_transform)
#print(Enc_text.head())
user_data["city"] = Enc_text["city"]
user_data["country"] = Enc_text["country"]
user_data["gp:tv_gender"] = Enc_text["gp:tv_gender"]
user_data["gp:status_level"] = Enc_text["gp:status_level"]

user_data.fillna(value=0,inplace=True)
#user_data.to_csv("/home/nazib/Desktop/Travello/RawData/Processed_User_data.csv", index=False)

### Extracting user info of Ranked Feed ###
ranked_user_info = extract(user_data,ranked_feed)

### Selecting a Random User for Ranking ###
select = np.random.randint(user_data.shape[0])
values = user_data.iloc[select].values
values = np.expand_dims(values,axis=0)
selected_uid = pd.DataFrame(data=values, columns=user_data.columns)

### One-to-many Adjecency Matrix  from the Global Ranked data###
user_weights =pd.DataFrame(data=np.zeros(shape=(ranked_feed.shape[0],selected_uid.shape[1])),columns=user_data.columns)

for i in range(ranked_user_info.shape[0]):
    
    values = ranked_user_info.iloc[i].values
    values = np.expand_dims(values, axis=0)

    poster_id = pd.DataFrame(data=values,columns=user_data.columns)
    '''
    ### Same id weight ###
    if poster_id['user_id'][0] == selected_uid['user_id'][0]:
        user_weights['user_id'][i] =1.0
    else:
        user_weights['user_id'][i] = 0.0
    '''
    ### City Weight ###
    if selected_uid['city'][0] == poster_id['city'][0]:
        user_weights['city'][i] = 1.0
    else:
        user_weights['city'][i]= 0.0
    ### Country weight ###
    if selected_uid['country'][0] == poster_id['country'][0]:
        user_weights['country'][i] = 1.0
    else:
        user_weights['country'][i] = 0.0
    ### Gender weight ###
    if selected_uid['gp:tv_gender'][0] == poster_id['gp:tv_gender'][0]:
        user_weights['gp:tv_gender'][i] == 1.0
    else:
        user_weights['gp:tv_gender'][i] == 0.0
    ### Level weight ###
    if selected_uid['gp:status_level'][0] == poster_id['gp:status_level'][0]:
        user_weights['gp:status_level'][i] == 1.0
    else:
        user_weights['gp:status_level'][i] == 0.0

    ### Comments weight ###
    sid_data  = float(selected_uid['gp:total_comments'][0])
    pid_data =  float(poster_id['gp:total_comments'][0])
    user_weights['gp:total_comments'][i] = calculate_weight(sid_data,pid_data)
    
    ### Like weight ###
    sid_data  = float(selected_uid['gp:total_likes'][0])
    pid_data =  float(poster_id['gp:total_likes'][0])
    user_weights['gp:total_likes'][i] = calculate_weight(sid_data,pid_data)

    ### Follower weight ###
    sid_data  = float(selected_uid['gp:num_followers'][0])
    pid_data =  float(poster_id['gp:num_followers'][0])
    user_weights['gp:num_followers'][i] = calculate_weight(sid_data,pid_data)
    
    ### Status Level weight ###
    sid_data  = float(selected_uid['gp:status_level'][0])
    pid_data =  float(poster_id['gp:status_level'][0])
    user_weights['gp:status_level'][i] = calculate_weight(sid_data,pid_data)

user_weights = user_weights.values
ranked_user_info.drop("user_id",axis=1,inplace=True)
user_feature = np.zeros(shape=user_weights.shape, dtype=float)
user_feature[:,1:] = ranked_user_info.values
user_feature[:,0] = user_weights[:,0]
global_ranks = pd.to_numeric(ranked_feed["RankScore"])

personal_rank = np.zeros(shape=(user_weights.shape[0],1),dtype=float)

for i in range(user_weights.shape[0]):
    personal_rank[i] = np.sum(user_feature[i,:]*user_weights[i,:]*global_ranks[i])

ranked_feed["Personalized Rank"] = personal_rank
ranked_feed.sort_values(by=ranked_feed.columns.values[-1], ascending=False, inplace=True)
ranked_feed.to_csv("/home/nazib/Desktop/Travello/RawData/Personal_Ranked_feed_10k.csv",index=False)







































