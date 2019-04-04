from keras.models import Model
from keras.initializers import RandomUniform, lecun_uniform
from keras.regularizers import l2
from keras.layers import Dense, Input, Lambda, Reshape, Dropout
from keras.optimizers import Adam
import keras.backend as K
from keras.layers.merge import concatenate, multiply, add, subtract, maximum, Dot
import numpy as np

from keras.losses import mse
import tensorflow as tf
import time

class Dqn(object):
    def __init__(self, S_dim, num_A):
        self.S_dim, self.num_A = S_dim, num_A
        self.tau = 0.01
        self.initModels()
        self.initTargetModels()

    def initModels(self):

        ### Inputs
        S = Input(shape=self.S_dim)
        A = Input(shape=(1,), dtype='uint8')
        G = Input(shape=self.S_dim)
        W = Input(shape=self.S_dim)
        TARGETS = Input(shape=(1,))

        ### Q values and action models
        qvals = self.create_critic_network(S, G, W)
        self.model = Model([S, G, W], qvals)
        self.qvals = K.function(inputs=[S, G, W], outputs=[qvals], updates=None)
        actionProbs = K.softmax(qvals)
        self.actionProbs = K.function(inputs=[S, G, W], outputs=[actionProbs], updates=None)
        actionFilter = K.squeeze(K.one_hot(A, self.num_A), axis=1)
        qval = K.sum(actionFilter * qvals, axis=1, keepdims=True)
        self.qval = K.function(inputs=[S, G, W, A], outputs=[qval], updates=None)

        ### DQN loss
        td_errors = qval - TARGETS
        l2_loss = K.square(td_errors)
        loss = K.mean(l2_loss)

        inputs = [S, A, G, W, TARGETS]
        updates = Adam(lr=0.001).get_updates(loss, self.model.trainable_weights)
        metrics = [loss, qval, td_errors]
        self.train = K.function(inputs, metrics, updates)

    def initTargetModels(self):
        S = Input(shape=self.S_dim)
        A = Input(shape=(1,), dtype='uint8')
        G = Input(shape=self.S_dim)
        W = Input(shape=self.S_dim)
        targetQvals = self.create_critic_network(S, G, W)
        self.targetmodel = Model([S, G, W], targetQvals)
        self.targetqvals = K.function(inputs=[S, G, W], outputs=[targetQvals], updates=None)
        actionFilter = K.squeeze(K.one_hot(A, self.num_A), axis=1)
        targetQval = K.sum(actionFilter * targetQvals, axis=1, keepdims=True)
        self.targetqval = K.function(inputs=[S, G, W, A], outputs=[targetQval], updates=None)

    def target_train(self):
        weights = self.model.get_weights()
        target_weights = self.targetmodel.get_weights()
        for i in range(len(weights)):
            target_weights[i] = self.tau * weights[i] + (1 - self.tau) * target_weights[i]
        self.targetmodel.set_weights(target_weights)

    def create_critic_network(self, S, G, W):
        # TODO split S, apply hidden to w separately
        h = concatenate([S, G, W])
        for l in [128, 128]:
            h = Dense(l, activation="relu",
                      kernel_initializer=lecun_uniform()
                      )(h)
        Q_values = Dense(self.num_A,
                         activation='linear',
                         kernel_initializer=RandomUniform(minval=-3e-4, maxval=3e-4),
                         bias_initializer=RandomUniform(minval=-3e-4, maxval=3e-4),
                         kernel_regularizer=l2(0.01),
                         )(h)
        # ValAndAdv = Dense(self.num_A + 1,
        #                   activation='linear',
        #                   kernel_initializer=RandomUniform(minval=-3e-4, maxval=3e-4),
        #                   bias_initializer=RandomUniform(minval=-3e-4, maxval=3e-4)
        #                   )(h)
        # Q_values = Lambda(
        #     lambda a: K.expand_dims(a[:, 0], axis=-1) + a[:, 1:] - K.mean(a[:, 1:], keepdims=True, axis=1),
        #     output_shape=(self.num_A,))(ValAndAdv)
        return Q_values
