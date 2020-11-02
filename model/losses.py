import numpy as np
import tensorflow as tf
from keras import backend as K


def KL_loss(z_mean, z_log_var, raxis=1):
    def loss(y_true, y_pred):
        kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
        kl_loss = K.sum(kl_loss, axis=-1)
        kl_loss *= -0.5
        return kl_loss
    return loss
