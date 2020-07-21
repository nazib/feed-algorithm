import numpy as np
import pandas as pd
from numpy import matlib as mt
from preprocess_data import*

###########################################################################
#                        Global Feed Ranking Logic                        #
###########################################################################
def GlobalRank(data, weights):
    TH= 60*36
    values = data[:,2]
    decay = weights[2]*np.exp(1-(values/TH))

    rank_feat = np.zeros(shape=(data.shape[0],5),dtype=float)
    #rank_feat[:,0] = test_data[:,0]
    rank_feat[:,1] = data[:,1]
    rank_feat[:,2] = data[:,3]
    rank_feat[:,3] = data[:,4]
    rank_feat[:,4] = data[:,5]

    #theta = RF.feature_importances_
    weights = [weights[0],weights[1],weights[3],weights[4],weights[5]]
    weights = mt.repmat(weights,data.shape[0],1)
    feed_rank = np.zeros(shape=(weights.shape[0],1),dtype=float)

    for i in range(1,weights.shape[0]):
        feed_rank[i] = np.sum(rank_feat[i,:] * weights[i,:] *decay[i])
    '''
    #feed_rank = np.sort(feed_rank,axis=0)
    ranked_feed = pd.read_csv(file_path)
    N,M = ranked_feed.shape
    ranked_feed = ranked_feed.loc[ratio+1:N,ranked_feed.columns]
    ranked_feed['RankScore'] = feed_rank
    ranked_feed.sort_values(by=ranked_feed.columns.values[-1], ascending=False, inplace=True)
    '''
    return feed_rank

def Personalized_ranks(ranked_feed,ranked_user_info,user_data,selected_uid):
    ### One-to-many Adjecency Matrix  from the Global Ranked data###
    user_weights =pd.DataFrame(data=np.zeros(shape=(ranked_user_info.shape[0],selected_uid.shape[1])),columns=user_data.columns)

    for i in range(ranked_user_info.shape[0]):
        
        values = ranked_user_info.iloc[i].values
        values = np.expand_dims(values, axis=0)

        poster_id = pd.DataFrame(data=values,columns=user_data.columns)
        
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
    global_ranks = ranked_feed["RankScore"].values#pd.to_numeric(ranked_feed["RankScore"])

    personal_rank = np.zeros(shape=(user_weights.shape[0],1),dtype=float)

    for i in range(user_weights.shape[0]):
        personal_rank[i] = np.sum(user_feature[i,:]*user_weights[i,:])*global_ranks[i]
    
    return personal_rank
    