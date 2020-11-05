
import os
import glob
import numpy as np
from datetime import datetime


def calculate_weight(sid, pid):
    if (sid == 0.0 and pid == 0.0) or pid == 0.0:
        return 0
    else:
        # np.exp((sid-pid)/(sid+pid))
        return np.exp(-np.power(sid - pid, 2.) / (2 * np.power(pid, 2.)))


def get_global_model_load_path():
    logs_dir = os.path.abspath(os.getcwd()+"/logs")
    files = sorted(
        glob.glob(logs_dir + "/**/VAE_noisy.h5"),
        key=os.path.getmtime,
        reverse=True)

    if len(files) <= 0:
        return ''
    return files[0]


def get_global_model_save_path():
    if not os.path.exists("logs"):
        os.mkdir("logs")
    logdir = "logs/" + datetime.now().strftime("%Y%m%d-%H%M%S_noisy")

    return (
        logdir,
        logdir+"/VAE_noisy.h5"
    )

def similarity(user_interests,poster_interests):
    ulen = len(user_interests)
    plen = len(poster_interests)

    if ulen > plen or ulen == plen:
        udata = np.zeros(shape=(ulen))
        pdata = np.zeros(shape=(ulen))
        udata = user_interests
        pdata[:plen] = poster_interests
    elif plen > ulen:
        udata = np.zeros(shape=(plen))
        pdata = np.zeros(shape=(plen))
        udata[:ulen] = user_interests
        pdata = poster_interests


    product = np.mean((udata - udata.mean()) * (pdata - pdata.mean()))
    return product
    ''''
    stds = udata.std() * pdata.std()
    if stds == 0:
        return 0
    else:
        product /= stds
        return product
    '''
