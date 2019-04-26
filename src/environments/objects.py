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

class Objects(Env):
    metadata = {'render.modes': ['human', 'ansi']}

    def __init__(self, seed=None, nbFeatures=3, nbObjects=5, density=0.5, nbActions=10):
        self.nbFeatures = nbFeatures
        self.nbObjects = nbObjects
        self.nbActions = nbActions
        self.lastaction = None
        #
        # self.As = [np.array([[-0.1, 0, 0.05],
        #                      [0, 0.1, 0],
        #                      [0, 0.05, -0.1]]),
        #            np.array([[0.1, 0.05, 0],
        #                      [0, -0.1, 0],
        #                      [0, -0.05, -0.1]]),
        #            np.array([[-0.1, 0, 0.05],
        #                      [-0.05, 0.1, 0],
        #                      [0, 0.05, 0.1]]),
        #            np.array([[-0.1, 0, 0.05],
        #                      [0, -0.1, 0.05],
        #                      [0, 0, 0.1]])]
        # self.As.append(np.zeros((self.nbFeatures,self.nbFeatures)))
        # self.As = np.array(self.As)
        #
        rng = np.random.RandomState(seed)
        self.FFNs = [FeedForwardNetwork(self.nbFeatures, [32], self.nbFeatures, rng, density)
                     for _ in range(self.nbActions - 1)]
        self.centers = np.vstack([rng.uniform(-1, 1, size=self.nbFeatures)
                                  for _ in range(self.nbActions - 1)])
        self.objects = []
        init_states = [rng.uniform(-1, 1, size=self.nbFeatures) for _ in range(self.nbObjects)]
        # init_states = [
        #     # np.array([0, 0, 0]),
        #     np.array([0.5, 0.5, 0.5]),
        #     np.array([0.4, 0.5, 0.5]),
        #     # np.array([0.5, 0.5, 0.4]),
        #     # np.array([0.5, 0.4, 0.5]),
        #     # np.array([0.5, 0.5, -0.5]),
        #     # np.array([0.5, -0.5, 0.5]),
        #     # np.array([0.5, -0.5, -0.5]),
        #     # np.array([-0.5, 0.5, 0.5]),
        #     # np.array([-0.5, 0.5, -0.5]),
        #     # np.array([-0.5, -0.5, 0.5]),
        #     np.array([-0.5, -0.5, -0.5]),
        #     np.array([-0.5, -0.4, -0.5]),
        #     np.array([-0.5, -0.5, -0.4]),
        #     np.array([-0.4, -0.5, -0.5]),
        #     np.array([-0.6, -0.5, -0.5]),
        #     np.array([-0.5, -0.6, -0.5]),
        #     np.array([-0.5, -0.5, -0.6]),
        #     np.array([-0.9, -0.5, -0.5]),
        #     np.array([-0.5, -0.9, -0.5]),
        #     np.array([-0.5, -0.5, -0.9]),
        #     np.array([-0.7, -0.5, -0.5]),
        #     np.array([-0.5, -0.7, -0.5]),
        #     np.array([-0.5, -0.5, -0.7]),
        #     np.array([-0.8, -0.5, -0.5]),
        #     np.array([-0.5, -0.5, -0.8]),
        #     np.array([-0.5, -0.8, -0.5])
        # ]
        for init_state in init_states:
            self.objects.append(Obj(self, init_state=init_state))


    def step(self, a):

        env_a = a[1]
        if self.lastaction is not None and np.random.rand() < 0:
            env_a = self.lastaction

        self.lastaction = a[1]

        object = self.objects[a[0]]

        next_state = self.next_state(object.state, env_a)
        object.state = np.clip(next_state, -1, 1)

        return self.state, 0, 0, {}

    def next_state(self, state, a):
        # A = self.As[a].copy()
        # if a < self.nbActions - 1 and np.linalg.norm(state - self.Ms[a]) > 0.6:
        #     A = np.zeros(self.nbFeatures)
        # state = np.dot(2*A + np.eye(self.nbFeatures), state)
        if a < self.nbActions - 1 and np.linalg.norm(state - self.centers[a]) < 1:
            output = self.FFNs[a].forward(state)
            state += output.squeeze()

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
