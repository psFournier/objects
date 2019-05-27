import numpy as np
from utils import FeedForwardNetwork
from gym import Env
from sklearn.neighbors import LocalOutlierFactor
import time
from sklearn import svm
from scipy.stats import norm

class Obj():
    def __init__(self, env, init_val):
        self.env = env
        self.init_val = init_val
        self.reset()

    def reset(self):
        self.state = np.array([0.,
                               0.,
                               0.,
                               0.,
                               self.init_val[0],
                               self.init_val[1]
                               ])

class Objects4(Env):
    metadata = {'render.modes': ['human', 'ansi']}

    def __init__(self, seed=None):
        self.nbFeatures = 6
        self.nbActions = 3
        self.lastaction = None
        # rng = np.random.RandomState(seed)
        # self.FFN = FeedForwardNetwork(2, [4], 1, rng, 1, 0.05)
        self.ranges = np.array([
            [-0.1, 0.1],
            [-0.019, 0.019],
            [-0.07, 0.07],
            [-0.019, 0.019],
            [0, 1],
            [0, 1]
        ])
        self.avgs = np.mean(self.ranges, axis=1)
        self.spans = self.ranges[:, 1] - self.ranges[:, 0]

        self.set_objects()

    def set_objects(self, n=None):
        # initvals = [[0.1, 0.1], [0.5, 0.1], [0.9, 0.1], [0.5, 0.1], [0.5, 0.5], [0.5, 0.9], [0.9, 0.1], [0.9, 0.5], [0.9, 0.9]]
        initvals = [[1,0]]
        self.nbObjects = len(initvals)
        self.objects = [Obj(self, np.array(initval)) for initval in initvals]

    def step(self, a):

        env_a = a
        if self.lastaction is not None and np.random.rand() < 0:
            env_a = self.lastaction

        self.lastaction = a

        for object in self.objects:
            object.state = self.next_state(object.state, env_a)

        return self.state, 0, 0, {}

    def next_state(self, state, a):
        if a == 1:
            state[1] = np.clip(state[1] + 0.001, -0.019, 0.019)
        elif a == 2:
            state[1] = np.clip(state[1] - 0.001, -0.019, 0.019)
        newstate0 = np.clip(state[0] + state[1], -0.1, 0.1)
        if (newstate0 > state[0]  and state[0] <= state[2] and state[2] <= newstate0) \
                or (newstate0 < state[0] and state[2] >= newstate0 and state[0] >= state[2]):
            state[3] = state[1] * state[4]
        else:
            state[3] = state[5] * state[3]
        state[2] = np.clip(state[2] + state[3], -0.07, 0.07)
        state[0] = newstate0
        return state

    def reset(self):
        for object in self.objects:
            object.reset()
        self.lastaction = None
        return self.state

    @property
    def state(self):
        res = np.hstack([obj.state for obj in self.objects])
        return res
