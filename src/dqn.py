from keras.models import Model
from keras.initializers import RandomUniform, TruncatedNormal
from keras.regularizers import l2
from keras.layers import Dense, Input, Lambda, Reshape, Dropout
from keras.optimizers import Adam
import keras.backend as K
from keras.layers.merge import concatenate, multiply, add, subtract, maximum, Dot
import numpy as np
from utils import softmax, merge_two_dicts

class Controller(object):
    def __init__(self, wrapper, nstep, _gamma, _lambda, IS, layers, dropout, l2reg):
        self.nstep = nstep
        self._gamma = _gamma
        self._lambda = _lambda
        self.IS = IS
        self.layers = layers
        self.wrapper = wrapper
        self.name = 'dqn_model'

        S, G = Input(shape=(wrapper.state_dim,)), Input(shape=(wrapper.goal_dim,))
        A = Input(shape=(1,), dtype='uint8')
        targets = Input(shape=(1,))
        qvals = self.create_network(S, G, wrapper.action_dim, dropout, l2reg)
        actionFilter = K.squeeze(K.one_hot(A, wrapper.action_dim), axis=1)
        qval = K.sum(actionFilter * qvals, axis=1, keepdims=True)
        td_errors = qval - targets
        l2_loss = K.square(td_errors)
        loss = K.mean(l2_loss)

        self.model = Model([S, G], qvals)
        updates = Adam(lr=0.001).get_updates(loss, self.model.trainable_weights)
        self._qvals = K.function(inputs=[S, G], outputs=[qvals], updates=None)
        self._qval = K.function(inputs=[S, G, A], outputs=[qval], updates=None)
        self._train = K.function([S, G, A, targets], [loss, qval, td_errors], updates)

        S_target, G_target = Input(shape=(wrapper.state_dim,)), Input(shape=(wrapper.goal_dim,))
        targetQvals = self.create_network(S_target, G_target, wrapper.action_dim, dropout, l2reg)

        self.targetmodel = Model([S_target, G_target], targetQvals)
        self._targetqvals = K.function(inputs=[S_target, G_target], outputs=[targetQvals], updates=None)

        self.rho = 0
        self.target = 0
        self.reward = 0
        self.qval = 0
        self.tderror = 0
        self.stat_steps = 0

    def target_train(self):
        weights = self.model.get_weights()
        target_weights = self.targetmodel.get_weights()
        for i in range(len(weights)):
            target_weights[i] = 0.01 * weights[i] + 0.99 * target_weights[i]
        self.targetmodel.set_weights(target_weights)

    def create_network(self, S, G, num_A, dropout, l2reg):
        h = concatenate([S, G])
        for l in self.layers:
            Dense(l,
                  activation="relu",
                  kernel_initializer=TruncatedNormal(),
                  bias_initializer=TruncatedNormal(),
                  kernel_regularizer=l2(l2reg),
                  )(h)
            h = Dropout(rate=dropout)(h)
        Q_values = Dense(num_A,
                         activation='linear',
                         kernel_initializer=TruncatedNormal(),
                         bias_initializer=TruncatedNormal(),
                         kernel_regularizer=l2(l2reg),
                         )(h)
        return Q_values

    def train(self, exps):
        inputs = self.get_inputs(exps)
        loss, qval, td_errors = self._train(inputs)
        self.target_train()
        self.tderror += np.mean(td_errors)
        self.qval += np.mean(qval)
        self.stat_steps += 1

    def get_inputs(self, exps):
        nStepExpes = self.getNStepSequences(exps)
        nStepExpes, mean_reward = self.getQvaluesAndBootstraps(nStepExpes)
        states, actions, goals, targets, rhos = self.getTargetsSumTD(nStepExpes)
        inputs = [states, actions, goals, targets]
        self.rho += np.mean(rhos)
        self.target += np.mean(targets)
        self.reward += mean_reward
        return inputs

    def getNStepSequences(self, exps):
        nStepSeqs = []
        for exp in exps:
            nStepSeq = [exp]
            i = 1
            while i <= self.nstep - 1 and exp['next'] != None:
                exp = exp['next']
                nStepSeq.append(exp)
                i += 1
            nStepSeqs.append(nStepSeq)
        return nStepSeqs

    def getQvaluesAndBootstraps(self, nStepExpes):

        states, goals = [], []
        for nStepExpe in nStepExpes:
            first = nStepExpe[0]
            l = len(nStepExpe)

            # potential_goals = []
            # if first['g'].shape == first['s1'].shape:
            #     potential_goals.append(first['g'])
            # potential_goals += [first[key] for key in first.keys() if key.startswith('her_g')]
            # g = potential_goals[np.random.randint(len(potential_goals))]
            # goals += [g] * (l+1)

            goals += [first['g']] * (l+1)
            for exp in nStepExpe:
                states.append(exp['s0'])
            states.append(nStepExpe[-1]['s1'])

        states = np.vstack(states)
        goals = np.vstack(goals)

        rewards, terminals = self.wrapper.get_r(states, goals)
        mean_reward = np.mean(rewards)
        qvals = self._qvals([states, goals])[0]
        target_qvals = self._targetqvals([states, goals])[0]
        actionProbs = softmax(qvals, axis=1, theta=1000)

        i = 0
        for nStepExpe in nStepExpes:
            nStepExpe[0]['g'] = goals[i]
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

        return nStepExpes, mean_reward

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
                b = np.clip(b, self.wrapper.rNotTerm / (1 - self._gamma), self.wrapper.rTerm)
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

    @property
    def stats(self):
        d = {'qval':self.qval / self.stat_steps,
                'tderror': self.tderror / self.stat_steps,
                'target': self.target / self.stat_steps,
                'reward': self.reward / self.stat_steps,
                'rho': self.rho / self.stat_steps}
        self.stat_steps = 0
        self.qval = 0
        self.tderror = 0
        self.reward = 0
        self.rho = 0
        self.target = 0
        return d