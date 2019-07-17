import numpy as np
from utils import FeedForwardNetwork
from gym import Env
from sklearn.neighbors import LocalOutlierFactor
import time
from sklearn import svm
from scipy.stats import norm


class Obj():
    def __init__(self, env, varying_feature_ranges, fixed_feature_values, fixed_feature_ranges):
        self.env = env
        self.varying_feature_ranges = varying_feature_ranges
        self.fixed_feature_ranges = fixed_feature_ranges
        self.fixed_feature_values = fixed_feature_values
        ranges = np.vstack([varying_feature_ranges, fixed_feature_ranges])
        self.avgs = np.mean(ranges, axis=1)
        self.spans = ranges[:, 1] - ranges[:, 0]
        self.reset()

    def reset(self):
        self.state = np.hstack([np.random.uniform(self.varying_feature_ranges[:2, 0],
                                                  self.varying_feature_ranges[:2, 1]),
                                np.zeros(2),
                                self.fixed_feature_values])


class Objects1(Env):
    metadata = {'render.modes': ['human', 'ansi']}

    def __init__(self, seed=None):
        self.nbFeatures = 9
        self.nbActions = 5
        self.lastaction = None

        self.set_objects()

    def set_objects(self, n=0):
        self.objects = []
        fixed_feature_values = np.array([1, 1, 0, 0])
        for i in range(10):
            vals = np.append(fixed_feature_values, i/20)
            self.objects.append(
                Obj(self,
                    varying_feature_ranges=np.array([
                        [-0.05, 0.05],
                        [-0.05, 0.05],
                        [-0.02, 0.02],
                        [-0.02, 0.02],
                    ]),
                    fixed_feature_ranges=np.array([
                        [0, 1],
                        [0, 1],
                        [0, 1],
                        [0, 1],
                        [0, 1]
                    ]),
                    fixed_feature_values=vals
                    )
            )
        self.nbObjects = len(self.objects)

    def step(self, a):

        env_a = a
        if self.lastaction is not None and np.random.rand() < 0:
            env_a = self.lastaction

        self.lastaction = a

        for object in self.objects:
            object.state = self.next_state(object.state, env_a)

        return self.state, 0, 0, {}

    def next_state(self, state, a):

        if self.lastaction is not None and np.random.rand() < state[8]:
            a = self.lastaction

        if a == 1:
            state[2] = np.clip(state[2] + 0.001 * state[4], -0.02, 0.02)
            state[3] = state[3] * state[7]
        elif a == 2:
            state[2] = np.clip(state[2] - 0.001 * state[4], -0.02, 0.02)
            state[3] = state[3] * state[7]
        elif a == 3:
            state[3] = np.clip(state[3] + 0.001 * state[5], -0.02, 0.02)
            state[2] = state[2] * state[6]
        elif a == 4:
            state[3] = np.clip(state[3] - 0.001 * state[5], -0.02, 0.02)
            state[2] = state[2] * state[6]
        state[0] = np.clip(state[0] + state[2], -0.1, 0.1)
        state[1] = np.clip(state[1] + state[3], -0.1, 0.1)

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
