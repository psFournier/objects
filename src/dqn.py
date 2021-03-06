from keras.models import Model
from keras.initializers import RandomUniform, lecun_uniform, he_normal
from keras.regularizers import l2
from keras.layers import Dense, Input, Lambda, Reshape, Dropout
from keras.optimizers import Adam
import keras.backend as K
from keras.layers.merge import concatenate, multiply, add, subtract, maximum, Dot
import numpy as np
from utils import softmax, merge_two_dicts
import tensorflow as tf

class Controller(object):
    def __init__(self, agent, nstep, _gamma, _lambda, IS, layers, dropout, l2reg):
        self.nstep = nstep
        self._gamma = _gamma
        self._lambda = _lambda
        self.IS = IS
        self.layers = layers
        self.agent = agent
        self.name = 'dqn_model'

        S, G = Input(shape=(agent.wrapper.state_dim,)), Input(shape=(agent.wrapper.goal_dim,))
        A = Input(shape=(1,), dtype='uint8')
        targets = Input(shape=(1,))
        qvals = self.create_network(S, G, agent.wrapper.action_dim, dropout, l2reg)
        actionFilter = K.squeeze(K.one_hot(A, agent.wrapper.action_dim), axis=1)
        qval = K.sum(actionFilter * qvals, axis=1, keepdims=True)
        # loss = tf.losses.huber_loss(labels=targets, predictions=qval)
        td_errors = qval - targets
        l2_loss = K.square(td_errors)
        loss = K.mean(l2_loss, axis=0)

        self.model = Model([S, G], qvals)
        updates = Adam(lr=0.001).get_updates(loss, self.model.trainable_weights)
        self._qvals = K.function(inputs=[S, G], outputs=[qvals], updates=None)
        self._qval = K.function(inputs=[S, G, A], outputs=[qval], updates=None)
        self._train = K.function([S, G, A, targets], [loss, qval], updates=updates)
        self._test = K.function([S, G, A, targets], [loss, qval], updates=None)

        S_target, G_target = Input(shape=(agent.wrapper.state_dim,)), Input(shape=(agent.wrapper.goal_dim,))
        targetQvals = self.create_network(S_target, G_target, agent.wrapper.action_dim, dropout, l2reg)

        self.targetmodel = Model([S_target, G_target], targetQvals)
        self._targetqvals = K.function(inputs=[S_target, G_target], outputs=[targetQvals], updates=None)

        # self.rho = 0
        self.qvals = np.zeros(agent.wrapper.env.nbObjects)
        self.stat_steps = np.zeros(agent.wrapper.env.nbObjects)

    def target_train(self):
        weights = self.model.get_weights()
        target_weights = self.targetmodel.get_weights()
        for i in range(len(weights)):
            target_weights[i] = 0.001 * weights[i] + 0.999 * target_weights[i]
        self.targetmodel.set_weights(target_weights)

    def create_network(self, S, G, num_A, dropout, l2reg):
        h = concatenate([S, G])
        for l in self.layers:
            h = Dense(l,
                      activation="relu",
                      kernel_initializer=he_normal()
                      )(h)
        Q_values = Dense(num_A,
                         activation='linear',
                         kernel_initializer=RandomUniform(minval=-3e-4, maxval=3e-4)
                         )(h)
        return Q_values

    def train(self, exps, obj_nb):
        nStepExpes = self.getQvaluesAndBootstraps(exps)
        states, actions, goals, targets, rhos = self.getTargetsSumTD(nStepExpes)
        inputs = [states, actions, goals, targets]
        # Beware: loss is scalar, qval and tderrors are vectors
        loss_train_before, qval_train_before = self._train(inputs)
        self.target_train()
        # self.rho += np.mean(rhos)
        # TODO
        if self.stat_steps[obj_nb] == 0:
            self.qvals[obj_nb] = np.mean(qval_train_before)
        else:
            self.qvals[obj_nb] += np.mean(qval_train_before)
        self.stat_steps[obj_nb] += 1
        return loss_train_before, qval_train_before

    def getQvaluesAndBootstraps(self, exps):

        nStepExpes = []
        for exp in exps:
            nStepSeq = [exp]
            i = 1
            while i <= self.nstep - 1 and exp['next'] != None:
                exp = exp['next']
                nStepSeq.append(exp)
                i += 1
            nStepExpes.append(nStepSeq)

        states, goals = [], []
        for nStepSeq in nStepExpes:
            first = nStepSeq[0]
            l = len(nStepSeq)

            # potential_goals = []
            # if first['g'].shape == first['s1'].shape:
            #     potential_goals.append(first['g'])
            # potential_goals += [first[key] for key in first.keys() if key.startswith('her_g')]
            # g = potential_goals[np.random.randint(len(potential_goals))]
            # goals += [g] * (l+1)

            goals += [first['g']] * (l+1)
            for exp in nStepSeq:
                states.append(exp['s0'])
            states.append(nStepSeq[-1]['s1'])

        states = np.vstack(states)
        goals = np.vstack(goals)
        ## Beware: states and goals here are normalized, so the reward cannot be computed directly
        rewards, terminals = self.agent.wrapper.get_r(states, goals)
        # mean_reward = np.mean(rewards)

        qvals = self._qvals([states, goals])[0]
        target_qvals = self._targetqvals([states, goals])[0]
        actionProbs = softmax(qvals, axis=1, theta=1000)

        i = 0
        for nStepExpe in nStepExpes:
            # nStepExpe[0]['g'] = goals[i]
            for exp in nStepExpe:
                exp['q'] = qvals[i]
                exp['tq'] = target_qvals[i]
                exp['pi'] = actionProbs[i]
                i += 1
                exp['reward'] = rewards[i]
                exp['terminal'] = terminals[i]
            end = {'q': qvals[i], 'tq': target_qvals[i], 'pi': actionProbs[i]}
            nStepExpe.append(end)
            i += 1

        return nStepExpes

    def getTargetsSumTD(self, nStepExpes):
        targets = []
        states = []
        actions = []
        goals = []
        ros = []
        for nStepExpe in nStepExpes:
            tdErrors = []
            cs = []
            for exp0, exp1 in zip(nStepExpe[:-1], nStepExpe[1:]):

                b = np.sum(np.multiply(exp1['pi'], exp1['tq']), keepdims=True)
                b = exp0['reward'] + (1 - exp0['terminal']) * self._gamma * b
                #TODO influence target clipping
                b = np.clip(b, self.agent.wrapper.rNotTerm / (1 - self._gamma), self.agent.wrapper.rTerm)
                tdErrors.append((b - exp0['q'][exp0['a0']]).squeeze())

                ### Calcul des ratios variable selon la méthode
                if self.IS == 'no':
                    cs.append(self._gamma * self._lambda)
                elif self.IS == 'standard':
                    ro = exp0['pi'][exp0['a0']] / exp0['mu0']
                    cs.append(ro * self._gamma * self._lambda)
                elif self.IS == 'retrace':
                    ro = exp0['pi'][exp0['a0']] / exp0['mu0']
                    cs.append(min(1, ro) * self._gamma * self._lambda)
                elif self.IS == 'tb':
                    cs.append(exp0['pi'][exp0['a0']] * self._gamma * self._lambda)
                else:
                    raise RuntimeError

            cs[0] = 1
            exp = nStepExpe[0]
            states.append(exp['s0'])
            actions.append(exp['a0'])
            goals.append(exp['g'])
            ros.append(np.mean(cs))
            delta = np.sum(np.multiply(tdErrors, np.cumprod(cs)))
            targets.append(exp['q'][exp['a0']] + delta)

        res = [np.vstack(x) for x in [states, goals, actions, targets, ros]]
        return res

    def stats(self):
        d = {}
        steps = sum(self.stat_steps)
        if sum(self.stat_steps) != 0:
            d['qval'] = sum(self.qvals) / steps
            # d['rho'] = self.rho / steps
        for i, s in enumerate(self.stat_steps):
            if s != 0:
                self.qvals[i] /= s
            self.stat_steps[i] = 0
        d['qvals'] = self.qvals
        return d
