import numpy as np
from gym import Env
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm

def get_cmap(n, name='hsv'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n)

class Actions:
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3
    NOOP = 4

class Obj():
    def __init__(self, env, init_state, min_state, max_state):
        self.env = env
        self.init_state = init_state
        self.min_state = min_state
        self.max_state = max_state
        self.state = init_state.copy()

class Objects(Env):
    metadata = {'render.modes': ['human', 'ansi']}

    def __init__(self, nbFeatures=3, nbObjects=2, nbDependences=2, nbSubspaces=2, nbActions=5):
        self.nbFeatures = nbFeatures
        self.nbObjects = nbObjects
        self.nbDependences = nbDependences
        self.nbSubspaces = nbSubspaces
        self.cmap = get_cmap(self.nbObjects)
        self.nbActions = nbActions

        self.As = []
        for _ in range(self.nbActions):
            A = np.eye(self.nbFeatures) + np.diag(np.random.normal(0, 0.05, self.nbFeatures))
            for i in range(self.nbFeatures):
                js = np.random.choice([k for k in range(self.nbFeatures) if k != i],
                                      size=self.nbDependences - 1)
                for j in js:
                    A[i, j] += np.random.normal(0, 0.05)
            self.As.append(A)
        self.As.append(np.eye(self.nbFeatures))

        self.bounds = []
        for feature in range(self.nbFeatures):
            bounds = np.sort(np.random.uniform(low=-1, high=1, size=2 * self.nbSubspaces))
            list = []
            for i in range(self.nbSubspaces):
                list.append((bounds[2 * i], bounds[2 * i + 1]))
            self.bounds.append(list)

        self.objects = []
        for object in range(self.nbObjects):
            bounds = []
            for feature in range(self.nbFeatures):
                bounds.append(self.bounds[feature][np.random.choice(self.nbSubspaces)])
            self.objects.append(Obj(self,
                                    init_state=np.array([np.random.uniform(b[0], b[1]) for b in bounds]),
                                    min_state=np.array([b[0] for b in bounds]),
                                    max_state=np.array([b[1] for b in bounds])))


    def step(self, a):
        object = a[0]
        env_a = a[1]
        if self.lastaction is not None and np.random.rand() < 0:
            env_a = self.lastaction
        self.lastaction = a[1]

        A = self.As[env_a]
        object = self.objects[object]
        object.state = np.clip(np.dot(A, object.state),
                               object.min_state,
                               object.max_state)

        return self.state, 0, 0, {}

    def reset(self):
        for object in self.objects:
            object.state = object.init_state
        self.lastaction = None
        return self.state

    @property
    def state(self):
        res = np.hstack([obj.state for obj in self.objects])
        return res

if __name__ == '__main__':
    env = Objects(nbObjects=100)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    xs, ys, zs, cs = [], [], [], []
    for ep in range(50):
        s = env.reset()
        obj = np.random.choice(env.nbObjects)
        xs.append(env.objects[obj].state[0])
        ys.append(env.objects[obj].state[1])
        zs.append(env.objects[obj].state[2])
        cs.append(cm.hot(obj/env.nbObjects))
        for step in range(100):
            act = (obj, np.random.choice(5))
            s = env.step(act)
            xs.append(env.objects[obj].state[0])
            ys.append(env.objects[obj].state[1])
            zs.append(env.objects[obj].state[2])
            cs.append(cm.hot(obj/env.nbObjects))

    ax.scatter(xs, ys, zs, c=cs, s=2)

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    ax.set_xlim(-1,1)
    ax.set_ylim(-1,1)
    ax.set_zlim(-1,1)

    plt.show()
