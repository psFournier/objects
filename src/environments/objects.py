import numpy as np
from gym import Env
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
from sklearn.neighbors import LocalOutlierFactor
import time
from sklearn import svm
from scipy.stats import norm

def get_cmap(n, name='hsv'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n)

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

    def __init__(self, nbFeatures=6, nbObjects=1, nbDependences=3, nbSubspaces=0, nbActions=5):
        self.nbFeatures = nbFeatures
        self.nbObjects = nbObjects
        self.nbDependences = nbDependences
        self.nbSubspaces = nbSubspaces
        self.cmap = get_cmap(self.nbObjects)
        self.nbActions = nbActions

        self.As = []
        for _ in range(self.nbActions-1):
            A = np.diag(np.random.normal(0, 0.01, self.nbFeatures))
            for i in range(self.nbFeatures):
                js = np.random.choice([k for k in range(self.nbFeatures) if k != i], replace=False,
                                      size=self.nbDependences - 1)
                for j in js:
                    A[i, j] += np.random.normal(0, 0.01)
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

            # init_state = np.array([np.random.uniform(b[0], b[1]) for b in bounds])
            # init_state = (object * init_state * 2) / (np.linalg.norm(init_state) * (self.nbObjects - 1)) + \
            #              np.random.normal(0, 0.01)
            # init_state = np.clip(init_state, -1, 1)

            init_states = np.reshape(np.array([]), (0, self.nbFeatures))
            nb_init = (object + 1) ** 2
            for _ in range(nb_init):
                init_states = np.vstack([init_states, np.array([np.random.uniform(b[0], b[1]) for b in bounds])])

            min_state = np.array([b[0] for b in bounds])
            max_state = np.array([b[1] for b in bounds])
            self.objects.append(Obj(self,
                                    nb_init = nb_init,
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
        # object.state = np.clip(np.dot(norm.pdf(object.state, 0, 0.1) * A + np.eye(self.nbFeatures), object.state),
        #                        object.min_state,
        #                        object.max_state)
        state = np.dot(A + np.eye(self.nbFeatures), state)
        return state

    def reset(self):
        for i, object in enumerate(self.objects):
            object.state = object.init_states[np.random.randint(object.nb_init)]
        self.lastaction = None
        return self.state

    @property
    def state(self):
        res = np.hstack([obj.state for obj in self.objects])
        return res

if __name__ == '__main__':


    colors = np.array(['#377eb8', '#ff7f00'])

    env = Objects(nbObjects=4, nbFeatures=3, nbActions=5, nbDependences=2, nbSubspaces=0)
    fig = plt.figure()

    states = np.array([], dtype=np.int64).reshape(0, env.nbFeatures)
    cs = []
    for ep in range(env.nbObjects):
        obj = ep
        for _ in range(20):
            env.reset()
            states = np.vstack([states, np.expand_dims(env.objects[obj].state, axis=0)])
            cs.append(cm.hot(ep/env.nbObjects))
            for step in range(50):
                act = (obj, np.random.randint(5))
                env.step(act)
                states = np.vstack([states, np.expand_dims(env.objects[obj].state, axis=0)])
                cs.append(cm.hot(ep/env.nbObjects))

    ax1 = fig.add_subplot(111, projection='3d')
    ax1.set_xlabel('X Label')
    ax1.set_ylabel('Y Label')
    ax1.set_zlabel('Z Label')
    ax1.set_xlim(-1, 1)
    ax1.set_ylim(-1, 1)
    ax1.set_zlim(-1, 1)
    ax1.scatter(states[::1, 0], states[::1, 1], states[::1, 2], s=5, color=cs[::1])

    # ax2 = fig.add_subplot(122, projection='3d')
    # ax2.set_xlabel('X Label')
    # ax2.set_ylabel('Y Label')
    # ax2.set_zlabel('Z Label')
    # ax2.set_xlim(-1, 1)
    # ax2.set_ylim(-1, 1)
    # ax2.set_zlim(-1, 1)
    # ax2.scatter(states[::1, 3], states[::1, 4], states[::1, 5], s=5, color=cs)

    if False:

        detector = svm.OneClassSVM(nu=0.05, kernel="rbf")
        detector.fit(states)

        for ep in range(10):
            new_states = np.array([], dtype=np.int64).reshape(0, env.nbFeatures)
            env.reset()
            obj = np.random.choice(env.nbObjects)
            new_states = np.vstack([new_states, np.expand_dims(env.objects[obj].state, axis=0)])
            cs.append(cm.hot(ep/10))
            for step in range(100):
                act = (obj, np.random.choice(range(5), p=[0.1,0.1,0.1,0.6,0.1]))
                s = env.step(act)
                new_states = np.vstack([new_states, np.expand_dims(env.objects[obj].state, axis=0)])
                cs.append(cm.hot(ep/10))
            y_pred = detector.predict(new_states)
            ax.scatter(new_states[:, 0], new_states[:, 1], new_states[:, 2], s=10, color=colors[(y_pred + 1) // 2])


    plt.show(block=True)
