
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
        key=os.path.getmtime)

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
