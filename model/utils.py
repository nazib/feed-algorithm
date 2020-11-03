import numpy as np


def calculate_weight(sid, pid):
    if sid == 0.0 and pid == 0.0:
        return 0
    else:
        # np.exp((sid-pid)/(sid+pid))
        return np.exp(-np.power(sid - pid, 2.) / (2 * np.power(pid, 2.)))
