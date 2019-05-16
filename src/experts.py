import numpy as np
from collections import deque
from utils import softmax

class LP_expert(object):
    def __init__(self, agent, eta, beta, maxlen):
        self.agent = agent
        self.name = '_'.join(['lp', str(eta), str(beta), str(maxlen)]) + '_expert'
        self.competence_queues = [deque([agent.env_steps * agent.wrapper.rNotTerm], maxlen=maxlen)
                                  for _ in range(agent.env.nbObjects)]
        self.lps = np.zeros(agent.env.nbObjects)
        self.eta = eta
        self.beta = beta

    def update_probs(self, object, reward):
        lp = reward - np.mean(self.competence_queues[object])
        self.competence_queues[object].append(reward)
        self.lps[object] += self.eta * (lp - self.lps[object])

    @property
    def probs(self):
        return softmax(self.lps - min(self.lps), theta=self.beta)

    def stats(self):
        d = {'probs': self.probs}
        return d

class Reached_states_variance_maximizer_expert(object):
    def __init__(self, agent):
        self.K = agent.env.nbObjects
        self.buffer = agent.buffer
        self.probs = None
        self.name = 'rsv'

    def update_probs(self, object, reward):
        reached_states_all = []
        for buffer in self.buffer._buffers:
            if buffer._numsamples != 0:
                reached_states_all.append(
                    np.vstack([self.buffer._storage[idx]['s1'] for idx in buffer._storage])
                )
            else:
                reached_states_all.append(0)
        vars = [np.var(reached_states) for reached_states in reached_states_all]
        if np.sum(vars) != 0:
            vars /= np.sum(vars)
        else:
            vars = [1 / self.K] * self.K
        self.probs = np.array(vars)

class Uniform_expert(object):
    def __init__(self, agent):
        self.K = agent.env.nbObjects
        self.name = 'uni'

    def update_probs(self, object, reward):
        pass

    @property
    def probs(self):
        return np.array(1 / self.K * np.ones(self.K))

    def stats(self):
        return {}