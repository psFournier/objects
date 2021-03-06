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

class ObjectsForExp4(Env):
    metadata = {'render.modes': ['human', 'ansi']}

    def __init__(self, seed=None, nbFeatures=3, density=0.5, nbActions=5, amplitude=0.5):
        self.nbFeatures = nbFeatures
        self.nbObjects = 21
        self.nbActions = nbActions
        self.lastaction = None
        rng = np.random.RandomState(seed)
        self.FFNs = [FeedForwardNetwork(self.nbFeatures, [8], self.nbFeatures - 1, rng, density, amplitude)
                     for _ in range(self.nbActions - 1)]
        self.centers = np.vstack([rng.uniform(-1, 1, size=self.nbFeatures)
                                  for _ in range(self.nbActions - 1)])

        init_states = [np.hstack([rng.uniform(-1, 1, size=self.nbFeatures - 1), 0.5]) for _ in range(16)] + \
                      [np.hstack([rng.uniform(-1, 1, size=self.nbFeatures - 1), -0.5]) for _ in range(4)] + \
                      [np.hstack([rng.uniform(-1, 1, size=self.nbFeatures - 1), 0]) for _ in range(1)]

        self.objects = []
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
        if a < self.nbActions - 1 and np.linalg.norm(state - self.centers[a]) < 10:
            output = self.FFNs[a].forward(state)
            state[:-1] += output.squeeze()
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
