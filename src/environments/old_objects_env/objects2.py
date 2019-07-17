import numpy as np
from utils import FeedForwardNetwork
from gym import Env
from sklearn.neighbors import LocalOutlierFactor
import time
from sklearn import svm
from scipy.stats import norm

class Obj0():
    def __init__(self, env):
        self.env = env
        self.reset()

    def reset(self):
        self.state = np.array([0.,
                               0.,
                               # 0.,
                               np.random.uniform(-.05, .05),
                               0.,
                               .01
                               ])

class Obj1():
    def __init__(self, env):
        self.env = env
        self.reset()

    def reset(self):
        self.state = np.array([0.,
                               0.,
                               np.random.uniform(-.05, .05),
                               # 0.,
                               0.,
                               .02
                               ])

class Objects2(Env):
    metadata = {'render.modes': ['human', 'ansi']}

    def __init__(self, seed=None):
        self.nbFeatures = 5
        self.nbObjects = 20
        self.nbActions = 5
        self.lastaction = None
        rng = np.random.RandomState(seed)
        self.FFN = FeedForwardNetwork(2, [4], 1, rng, 1, 0.05)
        self.objects = [Obj0(self) for _ in range(19)] + [Obj1(self)]

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
            state[1] = np.clip(state[1] + 0.001, -0.02, 0.02)
        elif a == 2:
            state[1] = np.clip(state[1] + 0.005, -0.02, 0.02)
        elif a == 3:
            state[1] = np.clip(state[1] - 0.001, -0.02, 0.02)
        elif a == 4:
            state[1] = np.clip(state[1] - 0.005, -0.02, 0.02)
        newstate0 = np.clip(state[0] + state[1], -0.05, 0.05)
        if (newstate0 > state[0]  and state[0] <= state[2] and state[2] <= newstate0) \
                or (newstate0 < state[0] and state[2] >= newstate0 and state[0] >= state[2]):

            # if np.random.rand() > 10*state[4]:
            #     state[3] = 0.01 * np.sign(state[1])
            # else:
            #     state[3] = 0

            state[3] = 0.01 * np.sign(state[1]) * 100 *state[4]
        else:
            state[3] = 0
        state[2] = np.clip(state[2] + state[3], -0.05, 0.05)
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
