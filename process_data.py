import os 
import numpy as np
import pandas as pd
import pygeohash as pg
import math
from scipy.spatial import distance
from numpy import matlib as mt
import glob.glob

#raw_data = pd.read_csv("/home/nazib/Desktop/Travello/RawData/feed_data10K.tsv")
#columns = ['uid','ptid','likes','comments','ptime','ctime','seconds','group','text','location','mylocation','urls']
#new_data = pd.DataFrame(raw_data.values,columns=columns)
#new_data.to_csv("/home/nazib/Desktop/Travello/RawData/feed_data.csv")

###########################################################################
#                   Pre-processing Begins                                 #
###########################################################################

data = pd.read_csv("/home/nazib/Desktop/Travello/RawData/feed_data10K.tsv", sep='\t')
m,n = data.shape
text_data =data[data.columns.to_list()[-1]]
location = data["posted_location"].fillna(0)
mylocation =data['location'].fillna(0)

text_len = np.zeros(len(text_data),dtype=float)
hash_tags = np.zeros(len(text_data),dtype=float)
dist = np.zeros(len(text_data),dtype=float)

i=0
for x in text_data:
    text_len[i] = len(str(text_data[i]))
    hash_tags[i] = str(text_data[i]).count('#')
    
    if (location.values[i] == 0) or (mylocation.values[i] ==0):
        dist[i]=0.0
    else:
        loc = pg.decode(str(location.values[i]))
        mloc = pg.decode(str(mylocation.values[i]))
        dist[i] = distance.euclidean(loc,mloc)
    print(i,dist[i])
    i+=1

cols =['feed_id','uid','ptid','likes','comments','post_age','textlength','hashtags','distance','urls']
pro_data = pd.DataFrame(columns=cols)
pro_data["uid"] = data['uid']
pro_data["ptid"] = data['feedObjectId']
pro_data["feed_id"] = data['feed_id']
pro_data["likes"] = data['likes']
pro_data["comments"] = data['comments']
pro_data["post_age"] = data['Minutes']
pro_data["textlength"] = text_len
pro_data["hashtags"] = hash_tags
pro_data["distance"] = dist
pro_data["urls"] = data['numberOfMediaUrls']

pro_data.to_csv("/home/nazib/Desktop/Travello/RawData/processed_feed_data10K.csv",index=False)
'''
pro_data.drop(['uid'],axis=1)
pro_data.drop(['feed_id'],axis=1)
with sns.plotting_context("notebook",font_scale=1.5):
    g = sns.pairplot(pro_data[['likes','comments','post_age','textlength','hashtags','distance','urls']], 
                 hue='distance', palette='tab20',height=5)
g.set(xticklabels=[])
plt.show()

############# Correlation HeatMap to See feature correlations ##########
str_list = [] # empty list to contain columns with strings (words)
for colname, colvalue in pro_data.iteritems():
    if type(colvalue[1]) == str:
         str_list.append(colname)
# Get to the numeric columns by inversion            
num_list = pro_data.columns.difference(str_list) 
# Create Dataframe containing only numerical features
house_num = pro_data[num_list]
f, ax = plt.subplots(figsize=(16, 12))
plt.title('Pearson Correlation of features')
# Draw the heatmap using seaborn
#sns.heatmap(house_num.astype(float).corr(),linewidths=0.25,vmax=1.0, square=True, cmap="PuBuGn", linecolor='k', annot=True)
sns.heatmap(house_num.astype(float).corr(),linewidths=0.25,vmax=1.0, square=True, cmap="cubehelix", linecolor='k', annot=True)
plt.show()
'''