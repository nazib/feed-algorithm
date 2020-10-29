import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.models import Model
from keras.layers import Conv1D, Input
from keras.layers import LeakyReLU, Lambda, BatchNormalization, Dense


class vae_model:
    def __init__(self, latent_layers):
        self.latenet_layers = latent_layers
        print("Model initialized")

    def sampling(self, args):
        z_mean, z_log_var = args
        batch = K.shape(z_mean)[0]
        dim = K.int_shape(z_mean)[1]
        # by default, random_normal has mean = 0 and std = 1.0
        epsilon = K.random_normal(shape=(batch, dim))
        return z_mean + K.exp(0.5 * z_log_var) * epsilon

    def myConv(self, x_in, nf, strides=1, rate=1):
        x_out = Conv1D(nf, kernel_size=3, padding='same',
                       kernel_initializer='he_normal', strides=strides)(x_in)
        x_out = LeakyReLU(0.2)(x_out)
        return x_out

    def encoder(self):
        inp = Input(shape=(8,), name='input')
        x = Dense(self.latenet_layers.pop(0), activation='relu')(inp)
        for filters in self.latenet_layers:
            x = Dense(filters, activation='relu')(x)
            x = BatchNormalization()(x)

        z_mu = Dense(8, activation='relu', name='mu')(x)
        z_var = Dense(8, activation='relu', name="var")(x)

        z = Lambda(self.sampling, output_shape=(8,), name='z')([z_mu, z_var])
        self.latent_rep = z
        self.mean = z_mu
        self.logvar = z_var
        model = Model(inputs=inp, outputs=z, name='Encoder')
        # model.summary()
        return model

    def decoder(self):
        inp = Input(shape=(8,), name='input')
        x = Dense(self.latenet_layers[-1], activation='relu')(inp)

        for filters in reversed(self.latenet_layers):
            x = Dense(filters, activation='relu')(x)
            x = BatchNormalization()(x)

        output = Dense(8, activation='relu')(x)
        model = Model(inputs=inp, outputs=output, name='Decoder')
        return model

    def create_model(self):
        encode = self.encoder()
        decode = self.decoder()
        model = Model(inputs=encode.input, outputs=[
                      decode(encode.output), self.latent_rep])
        return model, self.mean, self.logvar
