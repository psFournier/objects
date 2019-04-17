from keras.models import Model
from keras.initializers import RandomUniform, lecun_uniform
from keras.regularizers import l2
from keras.layers import Dense, Input, Lambda, Reshape, Dropout
from keras.optimizers import Adam
import keras.backend as K
from keras.layers.merge import concatenate, multiply, add, subtract, maximum, Dot
import numpy as np
from keras.losses import MSE

class Predictor(object):
    def __init__(self, wrapper, layers, dropout, l2reg):
        S = Input(shape=(wrapper.state_dim,))
        A = Input(shape=(1,))
        y_true = Input(shape=(wrapper.state_dim,))
        y_pred = self.create_network(S, A, wrapper.state_dim,  layers, dropout, l2reg)
        model = Model([S,A], y_pred)
        loss = K.mean(MSE(y_true, y_pred))
        updates = Adam(lr=0.001).get_updates(loss, model.trainable_weights)
        self._pred = K.function([S, A], [y_pred], updates=None)
        self._train = K.function([S, A, y_true], [loss], updates)

    def create_network(self, S, A, S_dim, layers, dropout, l2reg):
        h = concatenate([S, A])
        for l in layers:
            h = Dense(l,
                      activation="relu",
                      kernel_initializer=lecun_uniform(),
                      bias_initializer=lecun_uniform(),
                      kernel_regularizer=l2(l2reg),
                      )(h)
            h = Dropout(rate=dropout)(h)
        y_pred = Dense(S_dim,
                       activation='tanh',
                       kernel_initializer=lecun_uniform(),
                       bias_initializer=lecun_uniform(),
                       kernel_regularizer=l2(l2reg),
                       )(h)
        return y_pred

    def train(self, exps):
        states = np.vstack([exp['s0'] for exp in exps])
        actions = np.vstack([exp['a0'] for exp in exps])
        y_true = np.vstack([exp['s1'] for exp in exps])
        loss, = self._train([states, actions, y_true - states])
        return loss, np.mean(np.var(states, axis=0))

