import numpy as np
from utils import FeedForwardNetwork
from gym import Env
from sklearn.neighbors import LocalOutlierFactor
import time
from sklearn import svm
from scipy.stats import norm

class Obj():
    def __init__(self, env, init_state):
        self.env = env
        self.init_state = init_state
        self.state = np.zeros_like(init_state)

class ObjectsEasy(Env):
    metadata = {'render.modes': ['human', 'ansi']}

    def __init__(self, seed=None, nbFeatures=2, nbObjects=10, density=0.1, nbActions=5, amplitude=1):
        self.nbFeatures = nbFeatures
        self.nbObjects = nbObjects
        self.nbActions = nbActions
        self.lastaction = None
        self.objects = []
        init_states = [np.hstack([np.zeros(nbFeatures)]) for _ in range(self.nbObjects)]
        for init_state in init_states:
            self.objects.append(Obj(self, init_state=init_state))


    def step(self, a):

        env_a = a
        if self.lastaction is not None and np.random.rand() < 0:
            env_a = self.lastaction

        self.lastaction = a

        for object in self.objects:
            next_state = self.next_state(object.state, env_a)
            object.state = np.clip(next_state, -1, 1)

        return self.state, 0, 0, {}

    def next_state(self, state, a):
        if a == 0:
            state[0] += 0.01
        elif a == 1:
            state[0] -= 0.01
        elif a == 2:
            state[1] += 0.01
        elif a == 3:
            state[1] -= 0.01
        return state

    def reset(self):
        for i, object in enumerate(self.objects):
            object.state = object.init_state.copy()
        self.lastaction = None
        return self.state

    @property
    def state(self):
        res = np.hstack([obj.state for obj in self.objects])
        return res
