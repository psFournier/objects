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
                               0.,
                               0.,
                               0.,
                               0.,
                               self.init_val[0],
                               self.init_val[1]
                               ])

class Objects3(Env):
    metadata = {'render.modes': ['human', 'ansi']}

    def __init__(self, seed=None):
        self.nbFeatures = 10
        self.nbActions = 5
        self.lastaction = None
        self.ranges = np.array([
            [-0.1, 0.1],
            [-0.1, 0.1],
            [-0.02, 0.02],
            [-0.02, 0.02],
            [-0.07, 0.07],
            [-0.07, 0.07],
            [-0.02, 0.02],
            [-0.02, 0.02],
            [0, 1],
            [0, 1]
        ])
        self.avgs = np.mean(self.ranges, axis=1)
        self.spans = self.ranges[:, 1] - self.ranges[:, 0]
        self.set_objects()

    def set_objects(self, n=None):
        initvals = [[0.1, 0.1], [0.9, 0.9], [0.1, 0.9], [0.9, 0.1]]
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
            state[2] = np.clip(state[2] + 0.001, -0.02, 0.02)
        elif a == 2:
            state[2] = np.clip(state[2] - 0.001, -0.02, 0.02)
        elif a == 3:
            state[3] = np.clip(state[3] + 0.001, -0.02, 0.02)
        elif a == 4:
            state[3] = np.clip(state[3] - 0.001, -0.02, 0.02)

        newstate0 = np.clip(state[0] + state[2], -0.1, 0.1)
        if (newstate0 > state[0] and state[0] <= state[4] and state[4] <= newstate0) \
                or (newstate0 < state[0] and state[4] >= newstate0 and state[0] >= state[4]):
            state[6] = state[2]
        else:
            state[6] = state[6] * state[8]

        newstate1 = np.clip(state[1] + state[3], -0.1, 0.1)
        if (newstate1 > state[1] and state[1] <= state[5] and state[5] <= newstate1) \
                or (newstate1 < state[1] and state[5] >= newstate1 and state[1] >= state[5]):
            state[7] = state[3]
        else:
            state[7] = state[7] * state[9]

        state[4] = np.clip(state[4] + state[6], -0.07, 0.07)
        state[5] = np.clip(state[5] + state[7], -0.07, 0.07)
        state[0] = newstate0
        state[1] = newstate1

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
