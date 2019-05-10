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

class ObjectsId(Env):
    metadata = {'render.modes': ['human', 'ansi']}

    def __init__(self, seed=None, nbFeatures=4, nbObjects=10, density=0.1, nbActions=5, amplitude=1):
        self.nbFeatures = nbFeatures
        self.nbObjects = nbObjects
        self.nbActions = nbActions
        self.lastaction = None
        rng = np.random.RandomState(seed)
        while True:
            self.FFNs = [FeedForwardNetwork(self.nbFeatures - 2, [2*i], self.nbFeatures - 2, rng, density, amplitude)
                         for i in range(self.nbActions - 1)]
            nb = 0
            for i,f in enumerate(self.FFNs):
                if np.max(np.abs(np.dot(np.transpose(f.weights[0]), np.transpose(f.weights[1])[:2*i,:]))) > 0:
                    nb += 1
            print(nb)
            if nb > 0:

                break

        self.centers = np.vstack([rng.uniform(-1, 1, size=self.nbFeatures)
                                  for _ in range(self.nbActions - 1)])
        self.objects = []
        states = np.array([]).reshape(0, 2)
        for _ in range(20):
            states = np.vstack([states, np.expand_dims(np.array([0.4,-0.4]), axis=0)])
            for step in range(200):
                act = np.random.choice(self.nbActions - 1)
                state = states[-1] + self.FFNs[act].forward(states[-1]).squeeze()
                state = np.clip(state, -1, 1).squeeze()
                states = np.vstack([states, np.expand_dims(state, axis=0)])
        # detector = svm.OneClassSVM(nu=0.02, kernel="rbf")
        detector = LocalOutlierFactor(n_neighbors=35, contamination=0.1)
        detector.fit(states)
        y_pred = (1-detector.fit_predict(states))//2
        far = np.ones(20 * 201, dtype=int)
        far[np.where(np.tile(np.arange(201).reshape(201, 1), reps=(20,1)) < 20)[0]] = 0
        self.goal = states[np.random.choice(np.where(y_pred*far == 1)[0])]
        init_states = [np.hstack([np.random.uniform(-1, 1, 2), states[0]]) for _ in range(self.nbObjects)]
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
            output = self.FFNs[a].forward(state[2:])
            state[2:] += output.squeeze()
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
