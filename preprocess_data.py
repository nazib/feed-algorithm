import numpy as np 
import pandas as pd
import glob
import pygeohash as pg
import os
from sklearn import preprocessing

def calculate_weight(sid,pid):
    if sid==0.0 and pid ==0.0:
        return 0
    else:
        return np.exp((sid-pid)/(sid+pid)) 

def extract(user, ranked):
    ext_data = pd.DataFrame(columns=user.columns)
    for i in range(ranked.shape[0]):
        person = user[user['user_id']==ranked['uid'].iloc[i]].values
        ext_data.loc[i,:] = person
    return ext_data

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

def pre_process_data(Data_dir, processed_file):

    if not os.path.exists(Data_dir):
        print("Data folder not exists")
    else:    
        files = glob.glob(Data_dir+"*.tsv")
        all_data = pd.DataFrame()

        for x in files:
            data = pd.read_csv(x, sep='\t')
            all_data = pd.concat([all_data,data])

        m,n = all_data.shape
        text_data =all_data[all_data.columns.to_list()[-1]].fillna(0)
        location = all_data["posted_location"].fillna(0)
        mylocation = all_data['location'].fillna(0)
        text_len = np.zeros(shape=(m,1),dtype=float)
        hash_tags = np.zeros(shape=(m,1),dtype=float)
        lat = np.zeros(shape=(m,1),dtype=float)
        lng = np.zeros(shape=(m,1),dtype=float)

        i=0
        for x in text_data:
            text_len[i] = len(str(x))
            hash_tags[i] = str(x).count('#')
            y = pg.decode(str(location.values[i]))
            lat[i] = y[0]
            lng[i] = y[1]
            print(i)
            i+=1

        cols =['feed_id','uid','ptid','likes','comments','post_age','textlength','hashtags','latitude','longitude','urls']
        pro_data = pd.DataFrame(columns=cols)
        pro_data["uid"] = all_data['postUserId']
        pro_data["ptid"] = all_data['feedObjectId']
        pro_data["feed_id"] = all_data['feed_id']
        pro_data["likes"] = all_data['likes']
        pro_data["comments"] = all_data['comments']
        pro_data["post_age"] = all_data[all_data.columns[7]]
        pro_data["textlength"] = text_len
        pro_data["hashtags"] = hash_tags
        pro_data["latitude"] = lat
        pro_data["longitude"] = lng
        pro_data["urls"] = data['numberOfMediaUrls']
        pro_data.to_csv(Data_dir+"{0}".format(processed_file),index=False)
        print("Data Processed !!! \n File:{0} saved in {1}".format(processed_file,Data_dir))

def create_training_data(Data_dir,processed_file):
    data = pd.read_csv(Data_dir+ processed_file)
    data.drop(["uid","ptid","feed_id"],axis=1,inplace=True)
    return data.values

def prep_user_interaction(Data_dir = "/media/nazib/E20A2DB70A2D899D/Ubuntu_desktop/Travello/RawData/new_feed_interaction/"):

    if not os.path.exists(Data_dir):
        print("Data folder not exists")
    else:    
        '''
        files = glob.glob(Data_dir+"*.tsv")
        user_data = pd.DataFrame()

        for x in files:
            data = pd.read_csv(x, sep='\t')
            user_data = pd.concat([user_data,data])
        
        user_data.to_csv(Data_dir+"All_User_data.csv", index=False)
        x = []
        for i in range(len(user_data.columns)):
            x.insert(i,str(user_data.columns[i]).replace("\t",""))
        user_data.columns = x
        '''
        user_data = pd.read_csv(Data_dir+"All_Cohort_Feed_Data.csv")
        
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
        #user_data.to_csv(Data_dir+"All_User_data.csv", index=False)
        return user_data






    