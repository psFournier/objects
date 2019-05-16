import numpy as np
from utils import softmax
from collections import deque

class Uniform_object_selector(object):
    def __init__(self, K):
        self.K = K
        self.name = 'uni_obj'

    def select(self):
        return np.random.randint(self.K)

    def update_weights(self, object, reward):
        pass

    def stats(self):
        return {}

class EXP4(object):
    def __init__(self, agent, gamma):
        self.agent = agent
        self.gamma = gamma
        self.K = agent.env.nbObjects
        self.attempts = np.zeros(self.K, dtype=int)
        self.experts_weights = np.ones(len(agent.experts), dtype='float32')
        self.exp4_probs = None
        self.name = 'exp4_obj'

    def update_probs(self):
        experts_weights_sum = np.sum(self.experts_weights)
        probs = (1 - self.gamma) * np.dot(self.experts_weights, self.experts_probs) / experts_weights_sum
        probs += self.gamma / self.K
        probs /= probs.sum()  # In case it is not already exactly one
        self.exp4_probs = probs

    def get_probs(self):
        return self.exp4_probs

    def update_weights(self, object, reward):
        for i, expert in enumerate(self.agent.experts.values()):
            a = self.gamma * reward * expert.probs[object]
            b = self.K * self.exp4_probs[object]
            self.experts_weights[i] *= np.exp(a/b)

    def select(self):
        self.update_probs()
        p = self.get_probs()
        object = np.random.choice(self.K, p=p)
        self.attempts[object] += 1
        return object

    @property
    def experts_probs(self):
        return np.vstack([expert.probs for expert in self.agent.experts.values()])

    def stats(self):
        d = {}
        d['probs'] = self.exp4_probs
        d['weights'] = self.experts_weights
        return d