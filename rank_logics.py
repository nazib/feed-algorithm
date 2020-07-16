import numpy as np
import pandas as pd
from numpy import matlib as mt

###########################################################################
#                        Global Feed Ranking Logic                        #
###########################################################################
def GlobalRank(data, weights, file_path, ratio):
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

    #feed_rank = np.sort(feed_rank,axis=0)
    ranked_feed = pd.read_csv(file_path)
    N,M = ranked_feed.shape
    ranked_feed = ranked_feed.loc[ratio+1:N,ranked_feed.columns]
    ranked_feed['RankScore'] = feed_rank
    ranked_feed.sort_values(by=ranked_feed.columns.values[-1], ascending=False, inplace=True)
    return ranked_feed
    