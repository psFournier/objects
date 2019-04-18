import numpy as np
from utils import softmax

class Random_action_selector(object):
    def __init__(self, agent):
        self.nbActions = agent.env.nbActions

    def select(self, state, goal):
        return np.random.choice(self.nbActions), np.ones(self.nbActions)/self.nbActions

class NN_action_selector(object):
    def __init__(self, agent):
        self.model = agent.model

    def select(self, state, goal):
        input = [np.expand_dims(state, axis=0), np.expand_dims(goal, axis=0)]
        qvals = self.model._qvals(input)[0].squeeze()
        probs = softmax(qvals, theta=1)
        action = np.random.choice(len(probs), p=probs)
        return action, probs