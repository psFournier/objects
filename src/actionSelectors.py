import numpy as np
from utils import softmax

class Random_action_selector(object):
    def __init__(self, nbActions):
        self.nbActions = nbActions

    def select(self, state, goal):
        return np.random.choice(self.nbActions), np.ones(self.nbActions)/self.nbActions

class NN_action_selector(object):
    def __init__(self, nn):
        self.nn = nn
        self.exploration = 1

    def select(self, state, goal):
        qvals = self.nn.compute_qvals(state, goal)
        probs = softmax(qvals, theta=self.exploration)
        action = np.random.choice(len(probs), p=probs)
        return probs, action