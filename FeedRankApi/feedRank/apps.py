from django.apps import AppConfig
from FeedRankApi import settings
import numpy as np
import tensorflow as tf
import keras
from . model import*
import pdb
import os

class FeedrankConfig(AppConfig):
    path = os.path.join(settings.MODELS_DIR, "VAE_noisy.h5")
    name = 'feedRank'
    model = vae_model([512,256,128,64,32,16,8])
    model,_,_ = model.create_model()
    model.load_weights(path)
    mu = model.get_layer('mu')
    mu_wgt = mu.get_weights()[1]
    var = model.get_layer('var')
    var_wgt = var.get_weights()[1]
    ### Generating Weights parameters from Distribution ####
    theta = mu_wgt + var_wgt * np.random.randn(8)
