import numpy as np
from utils import softmax
from collections import deque

class Uniform_object_selector(object):
    def __init__(self, agent):
        self.K = agent.env.nbObjects
        self.name = 'uni_obj'

    def select(self):
        return np.random.randint(self.K)

    def update_weights(self, object, reward):
        pass

    def stats(self):
        return {}

class EXP4(object):
    def __init__(self, agent):
        self.agent = agent
        self.gamma = float(agent.args['--exp4gamma'])
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
        # print(reward)
        for i, expert in enumerate(self.agent.experts.values()):
            a = self.gamma * reward * expert.probs[object]
            b = self.K * self.exp4_probs[object]
            # if i==3: print(expert.name, expert.probs[object], self.exp4_probs[object], np.exp(a/b))
            self.experts_weights[i] *= np.exp(a/b)
            # if i==3: print(self.experts_weights[i])
            # if a/b > 10: print(object, reward, expert.name,expert.probs[object], /)

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
        for i,e in enumerate(self.agent.experts.values()):
            d['weight_'+e.name] = self.experts_weights[i]
        return d