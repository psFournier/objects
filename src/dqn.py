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
        self._train = K.function([S, G, A, targets], [loss, qval, td_errors], updates)

        S_target, G_target = Input(shape=(agent.wrapper.state_dim,)), Input(shape=(agent.wrapper.goal_dim,))
        targetQvals = self.create_network(S_target, G_target, agent.wrapper.action_dim, dropout, l2reg)

        self.targetmodel = Model([S_target, G_target], targetQvals)
        self._targetqvals = K.function(inputs=[S_target, G_target], outputs=[targetQvals], updates=None)

        self.rho = 0
        self.targets = np.zeros(agent.wrapper.env.nbObjects)
        self.rewards = np.zeros(agent.wrapper.env.nbObjects)
        self.qvals = np.zeros(agent.wrapper.env.nbObjects)
        self.tderrors = np.zeros(agent.wrapper.env.nbObjects)
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

    def train(self, object):
        if self.agent.buffer._buffers[object]._numsamples > self.agent.batch_size:
            for _ in range(self.agent.train_steps):
                exps = self.agent.buffer.sample(self.agent.batch_size, object)
                nStepExpes = self.getNStepSequences(exps)
                nStepExpes, mean_reward = self.getQvaluesAndBootstraps(nStepExpes)
                states, actions, goals, targets, rhos = self.getTargetsSumTD(nStepExpes)
                inputs = [states, actions, goals, targets]
                loss, qval, td_errors = self._train(inputs)
                self.target_train()
                self.rho += np.mean(rhos)
                if self.stat_steps[object] == 0:
                    self.targets[object] = np.mean(targets)
                    self.rewards[object] = mean_reward
                    self.tderrors[object] = np.mean(td_errors)
                    self.qvals[object] = np.mean(qval)
                else:
                    self.targets[object] += np.mean(targets)
                    self.rewards[object] += mean_reward
                    self.tderrors[object] += np.mean(td_errors)
                    self.qvals[object] += np.mean(qval)
                self.stat_steps[object] += 1
                self.agent.train_step += 1
        else:
            print('not enough samples for batchsize')

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

        rewards, terminals = self.agent.wrapper.get_r(states, goals)
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
            d['reward'] = sum(self.rewards)/ steps
            d['tderror'] = sum(self.tderrors) / steps
            d['qval'] = sum(self.qvals) / steps
            d['target'] = sum(self.targets) / steps
            d['rho'] = self.rho / steps
        for i, s in enumerate(self.stat_steps):
            if s != 0:
                self.rewards[i] /= s
                self.tderrors[i] /= s
                self.targets[i] /= s
                self.qvals[i] /= s
            self.stat_steps[i] = 0
        d['rewards'] = self.rewards
        d['tderrors'] = self.tderrors
        d['qvals'] = self.qvals
        d['targets'] = self.targets
        return d
