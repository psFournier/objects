import numpy as np
from gym import Env
from sklearn.neighbors import LocalOutlierFactor
import time
from sklearn import svm
from scipy.stats import norm

class Obj():
    def __init__(self, env, nb_init, init_states, min_state, max_state):
        self.env = env
        self.nb_init = nb_init
        self.init_states = init_states
        self.min_state = min_state
        self.max_state = max_state
        self.state = np.zeros_like(min_state)


class Objects(Env):
    metadata = {'render.modes': ['human', 'ansi']}

    def __init__(self, init, nbFeatures=6, nbObjects=1, nbDependences=3, nbSubspaces=0, nbActions=5):
        self.nbFeatures = nbFeatures
        self.nbObjects = nbObjects
        self.nbDependences = nbDependences
        self.nbSubspaces = nbSubspaces
        self.nbActions = nbActions

        self.As = []
        for _ in range(self.nbActions-1):
            A = np.diag(np.random.normal(0, 0.5, self.nbFeatures))
            for i in range(self.nbFeatures):
                js = np.random.choice([k for k in range(self.nbFeatures) if k != i], replace=False,
                                      size=self.nbDependences - 1)
                for j in js:
                    A[i, j] += np.random.normal(0, 0.5)
            self.As.append(A)
        self.As.append(np.zeros((self.nbFeatures,self.nbFeatures)))
        self.As = np.array(self.As)

        self.bounds = []
        for feature in range(self.nbFeatures):
            bounds = np.sort(np.random.uniform(low=-1, high=1, size=2 * self.nbSubspaces))
            list = []
            for i in range(self.nbSubspaces):
                list.append((bounds[2 * i], bounds[2 * i + 1]))
            list.append((-1, 1))
            self.bounds.append(list)

        self.objects = []
        for object in range(self.nbObjects):
            bounds = []
            for feature in range(self.nbFeatures):
                bounds.append(self.bounds[feature][np.random.choice(self.nbSubspaces+1)])

            init_state = np.array([np.random.uniform(b[0], b[1]) for b in bounds])
            init_state = (init[object] * init_state) / (np.linalg.norm(init_state)) + \
                         np.random.normal(0, 0.01)
            init_states = np.reshape(np.clip(init_state, -1, 1), (1, self.nbFeatures))

            # init_states = np.reshape(np.array([]), (0, self.nbFeatures))
            # for _ in range(init[object]):
            #     init_state = np.array([np.random.uniform(b[0], b[1]) for b in bounds])
            #     init_states = np.vstack([init_states, init_state])

            min_state = np.array([b[0] for b in bounds])
            max_state = np.array([b[1] for b in bounds])
            self.objects.append(Obj(self,
                                    nb_init = init[object],
                                    init_states=init_states,
                                    min_state=min_state,
                                    max_state=max_state,))

    def step(self, a):

        env_a = a[1]
        if self.lastaction is not None and np.random.rand() < 0:
            env_a = self.lastaction

        self.lastaction = a[1]

        object = self.objects[a[0]]

        next_state = self.next_state(object.state, env_a)
        object.state = np.clip(next_state,
                               object.min_state,
                               object.max_state)

        return self.state, 0, 0, {}

    def next_state(self, state, a=None):
        if a is None:
            A = self.As
        else:
            A = self.As[a]
        state = np.dot(norm.pdf(np.linalg.norm(state), 0, 0.5) * A + np.eye(self.nbFeatures), state)
        # state = np.dot(A + np.eye(self.nbFeatures), state)
        return state

    def reset(self):
        for i, object in enumerate(self.objects):
            # object.state = object.init_states[np.random.randint(object.nb_init)]
            object.state = object.init_states[0]
        self.lastaction = None
        return self.state

    @property
    def state(self):
        res = np.hstack([obj.state for obj in self.objects])
        return res