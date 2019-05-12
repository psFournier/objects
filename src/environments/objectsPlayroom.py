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

class ObjectsPlayroom(Env):
    metadata = {'render.modes': ['human', 'ansi']}

    def __init__(self, seed=None, nbObjects=2):
        # Object state = (obj_pos_abs, shape, color)
        self.nbFeatures = 3
        self.nbObjects = nbObjects
        self.nbActions = 2
        self.lastaction = None
        self.objects = []
        rng = np.random.RandomState(seed)
        for i in range(self.nbObjects):
            self.objects.append(Obj(self, init_state=np.array([rng.uniform(-1, 1),
                                                               0,
                                                               rng.uniform(-1, 1)
                                                               ])))


    def step(self, a):

        env_a = a
        if self.lastaction is not None and np.random.rand() < 0:
            env_a = self.lastaction

        self.lastaction = a

        for object in self.objects:
            object.state = self.next_state(object.state, env_a)


        return self.state, 0, 0, {}

    def next_state(self, state, a):
        if a == 0:
            state[0] = np.clip(state[0] + 0.1, -1, 1)
        # if a == 1:
        #     if state[1] == 0:
        #         state[0] = np.clip(state[0] + 0.01, -1, 1)
        #     elif state[1] == 1:
        #         state[0] = np.clip(state[0] + 0.05, -1, 1)
        if a == 1:
            state[0] = np.clip(state[0] - 0.1, -1, 1)
        # if a == 3:
        #     if state[1] == 0:
        #         state[0] = np.clip(state[0] - 0.01, -1, 1)
        #     elif state[1] == 1:
        #         state[0] = np.clip(state[0] - 0.05, -1, 1)
        return state


    # def next_state(self, state, a):
    #     abs_pos = self.agent_pos + state[:2]
    #     if a == 0:
    #         self.agent_pos[0] = np.clip(self.agent_pos[0] + .1, -1, 1)
    #         if state[0] < 0.1 and state[0] > 0 and np.abs(state[1]) < 0.05:
    #             if state[2] == 0:
    #                 abs_pos[0] = np.clip(abs_pos[0] + .4, -1, 1)
    #             elif state[2] == 1:
    #                 abs_pos[0] = np.clip(abs_pos[0] + .1, -1, 1)
    #     if a == 1:
    #         self.agent_pos[0] = np.clip(self.agent_pos[0] - .1, -1, 1)
    #         if state[0] > -0.1 and state[0] < 0 and np.abs(state[1]) < 0.05:
    #             if state[2] == 0:
    #                 abs_pos[0] = np.clip(abs_pos[0] - .4, -1, 1)
    #             elif state[2] == 1:
    #                 abs_pos[0] = np.clip(abs_pos[0] - .1, -1, 1)
    #     if a == 2:
    #         self.agent_pos[1] = np.clip(self.agent_pos[1] + .1, -1, 1)
    #         if state[1] < 0.1 and state[1] > 0 and np.abs(state[0]) < 0.05:
    #             if state[2] == 0:
    #                 abs_pos[1] = np.clip(abs_pos[1] + .4, -1, 1)
    #             elif state[2] == 1:
    #                 abs_pos[1] = np.clip(abs_pos[1] + .1, -1, 1)
    #     if a == 3:
    #         self.agent_pos[1] = np.clip(self.agent_pos[1] - .1, -1, 1)
    #         if state[1] > -0.1 and state[1] < 0 and np.abs(state[0]) < 0.05:
    #             if state[2] == 0:
    #                 abs_pos[1] = np.clip(abs_pos[1] - .4, -1, 1)
    #             elif state[2] == 1:
    #                 abs_pos[1] = np.clip(abs_pos[1] - .1, -1, 1)
    #     state[:2] = abs_pos - self.agent_pos
    #     return state

    def reset(self):
        for i, object in enumerate(self.objects):
            object.state = object.init_state.copy()
        self.lastaction = None
        return self.state

    @property
    def state(self):
        res = np.hstack([obj.state for obj in self.objects])
        return res
