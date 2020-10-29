import numpy as np
import pandas as pd


def calculate_weight(sid, pid):
    if sid == 0.0 and pid == 0.0:
        return 0
    else:
        return np.exp((sid-pid)/(sid+pid))


def extract(user, ranked):
    ext_data = pd.DataFrame(columns=user.columns)
    for i in range(ranked.shape[0]):
        person = user[user['user_id'] == ranked['uid'].iloc[i]].values
        ext_data.loc[i, :] = person
    return ext_data


def remove_tab(data):
    for x in data.columns:
        cols = []
        for i in range(data[x].values.shape[0]):
            if data[x][i] == '\t':
                cols.insert(i, "0")
            else:
                cols.insert(i, str(data[x][i]).replace("\t", ""))
        data[x] = cols
    return data
