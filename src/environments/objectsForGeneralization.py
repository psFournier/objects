import numpy as np
from utils import FeedForwardNetwork
from gym import Env
from sklearn.neighbors import LocalOutlierFactor
import time
from sklearn import svm
from scipy.stats import norm


class Obj():
    def __init__(self, env, nb, fixed_feature_values):
        self.env = env
        self.nb = nb
        self.fixed_feature_values = fixed_feature_values
        self.reset()
        self.lastaction = None

    def reset(self):
        self.state = np.hstack([np.random.uniform(self.env.varying_feature_ranges[:2, 0],
                                                  self.env.varying_feature_ranges[:2, 1]),
                                np.zeros(2),
                                self.fixed_feature_values])
        self.lastaction = None

    def step(self, a):

        env_a = a
        if self.lastaction is not None and np.random.rand() < 0.05:
            env_a = self.lastaction
        self.lastaction = a

        self.state = self.env.next_state(self.state, env_a)

class ObjectsForGeneralization(Env):

    def __init__(self, seed=None):
        self.nbFeatures = 5
        self.nbActions = 5
        self.fixed_feature_ranges = np.array([
            [1, 2]
        ])
        self.varying_feature_ranges = np.array([
            [-0.05, 0.05],
            [-0.05, 0.05],
            [-0.02, 0.02],
            [-0.02, 0.02],
        ])
        ranges = np.vstack([self.varying_feature_ranges, self.fixed_feature_ranges])
        self.avgs = np.mean(ranges, axis=1)
        self.spans = ranges[:, 1] - ranges[:, 0]
        self.set_objects()

    def set_objects(self, n=0):
        self.objects = []

        for i in range(n):
            self.objects.append(
                Obj(self,
                    i,
                    fixed_feature_values=1+i/n
                    )
            )
        self.nbObjects = len(self.objects)

    def next_state(self, state, a):

        if a == 1:
            state[2] = np.clip(state[2] + 0.001 * state[4], -0.02, 0.02)
            state[3] = state[3] * 0
        elif a == 2:
            state[2] = np.clip(state[2] - 0.001 * state[4], -0.02, 0.02)
            state[3] = state[3] * 0
        elif a == 3:
            state[3] = np.clip(state[3] + 0.001 * state[4], -0.02, 0.02)
            state[2] = state[2] * 0
        elif a == 4:
            state[3] = np.clip(state[3] - 0.001 * state[4], -0.02, 0.02)
            state[2] = state[2] * 0
        state[0] = np.clip(state[0] + state[2], -0.1, 0.1)
        state[1] = np.clip(state[1] + state[3], -0.1, 0.1)

        return state

    def reset(self):
        for object in self.objects:
            object.reset()
        return self.state

    @property
    def state(self):
        res = np.hstack([obj.state for obj in self.objects])
        return res
