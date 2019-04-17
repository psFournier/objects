from src.environments.objects import Objects
from src.environments.objectsEasy import ObjectsEasy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
from sklearn.neighbors import LocalOutlierFactor
import time
from sklearn import svm
from scipy.stats import norm
import numpy as np


def get_cmap(n, name='hsv'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n)

if __name__ == '__main__':


    colors = np.array(['#377eb8', '#ff7f00'])
    # np.random.seed(1000)
    env = ObjectsEasy()
    fig = plt.figure()


    states = np.array([], dtype=np.int64).reshape(0, env.nbFeatures)
    cs = []
    sizes = []
    for ep in range(2):
        obj = ep
        for _ in range(20):
            env.reset()
            states = np.vstack([states, np.expand_dims(env.objects[obj].state, axis=0)])
            cs.append('green')
            sizes.append(5)
            # print('______________')
            for step in range(50):
                act = (obj, np.random.randint(5))
                # print(act[1])
                env.step(act)
                state = env.objects[obj].state
                # if any(state != states[-1]):
                #     c = 'red'
                #     s= 10
                # else:
                #     c= 'yellow'
                #     s=5
                states = np.vstack([states, np.expand_dims(state, axis=0)])
                cs.append('yellow')
                sizes.append(5)

    ax1 = fig.add_subplot(111, projection='3d')
    ax1.set_xlabel('X Label')
    ax1.set_ylabel('Y Label')
    ax1.set_zlabel('Z Label')
    ax1.set_xlim(-1, 1)
    ax1.set_ylim(-1, 1)
    ax1.set_zlim(-1, 1)
    ax1.scatter(states[::1, 0], states[::1, 1], states[::1, 2], s=sizes[::1], color=cs[::1])
    ax1.scatter(env.Ms[:,0], env.Ms[:,1], env.Ms[:,2], s=30, color='blue')

    # ax2 = fig.add_subplot(122, projection='3d')
    # ax2.set_xlabel('X Label')
    # ax2.set_ylabel('Y Label')
    # ax2.set_zlabel('Z Label')
    # ax2.set_xlim(-1, 1)
    # ax2.set_ylim(-1, 1)
    # ax2.set_zlim(-1, 1)
    # ax2.scatter(states[::1, 3], states[::1, 4], states[::1, 5], s=5, color=cs)

    if True:

        detector = svm.OneClassSVM(nu=0.15, kernel="rbf")
        detector.fit(states)

        for ep in range(10):
            new_states = np.array([], dtype=np.int64).reshape(0, env.nbFeatures)
            env.reset()
            obj = np.random.choice(2)
            new_states = np.vstack([new_states, np.expand_dims(env.objects[obj].state, axis=0)])
            # cs.append(cm.hot(ep/10))
            for step in range(100):
                # p = [0.2] * 5
                p = [0.1, 0.6, 0.1, 0.1, 0.1]
                act = (obj, np.random.choice(range(5),p=p))
                env.step(act)
                state = env.objects[obj].state
                new_states = np.vstack([new_states, np.expand_dims(state, axis=0)])
                # cs.append(cm.hot(ep/10))
            y_pred = detector.predict(new_states)
            ax1.scatter(new_states[:, 0], new_states[:, 1], new_states[:, 2], s=10, color=colors[(y_pred + 1) // 2])


    plt.show(block=True)
