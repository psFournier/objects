from src.environments.objects import Objects
from src.environments.objectsId import ObjectsId
from src.environments.objectsForExp4 import ObjectsForExp4

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
    env = ObjectsId(seed=10)
    fig = plt.figure()
    # ax1 = fig.add_subplot(211, projection='3d')
    ax2 = fig.add_subplot(111)

    for ep in range(1):
        obj = ep
        states = np.array([], dtype=np.int64).reshape(0, env.nbFeatures)
        inits = np.array([], dtype=np.int64).reshape(0, env.nbFeatures)
        steps = np.tile(np.arange(201).reshape(201, 1), reps=(20,1))
        for _ in range(20):
            env.reset()
            # states = np.vstack([states, np.expand_dims(env.objects[obj].state, axis=0)])
            state = env.objects[obj].state
            inits = np.vstack([inits, np.expand_dims(state, axis=0)])
            states = np.vstack([states, np.expand_dims(state, axis=0)])
            # print('______________')
            for step in range(200):
                act = np.random.choice(env.nbActions - 1)
                # nb = 0
                # while np.linalg.norm(state - env.centers[act]) > 1 and nb < 10:
                #     act = np.random.choice(env.nbActions - 1)
                #     nb += 1
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

        xx, yy = np.meshgrid(np.linspace(-1, 1, 150),
                             np.linspace(-1, 1, 150))
        # detector = svm.OneClassSVM(nu=0.001, kernel="rbf")
        detector = LocalOutlierFactor(n_neighbors=35, contamination=0.1)
        detector.fit(states[:,2:])
        # Z = detector.predict(np.c_[xx.ravel(), yy.ravel()])
        # Z = Z.reshape(xx.shape)
        # ax2.contour(xx, yy, Z, levels=[0], linewidths=2, colors='black')
        y_pred = (1-detector.fit_predict(states[:,2:])[::1])//2
        far = np.ones(20*201, dtype=int)
        far[np.where(steps < 20)[0]] = 0
        # ax1.scatter(inits[::1, 0], inits[::1, 1], inits[::1, 2], s=100, color='red')
        # ax1.scatter(states[::1, 0], states[::1, 1], states[::1, 2], s=10, color=colors[(y_pred[::1] + 1) // 2])
        ax2.scatter(inits[::1, 2], inits[::1, 3], s=50, color='red')
        ax2.scatter(states[::1, 2], states[::1, 3], s=5, color=colors[far*y_pred])
        ax2.scatter(env.goal[0], env.goal[1], s=50, color='black')


    # ax1.set_xlabel('X Label')
    # ax1.set_ylabel('Y Label')
    # ax1.set_zlabel('Z Label')
    # ax1.set_xlim(-1, 1)
    # ax1.set_ylim(-1, 1)
    # ax1.set_zlim(-1, 1)
    ax2.set_xlabel('X Label')
    ax2.set_ylabel('Y Label')
    # ax2.set_zlabel('Z Label')
    ax2.set_xlim(-1.05, 1.05)
    ax2.set_ylim(-1.05, 1.05)
    # ax2.set_zlim(-1, 1)
    # ax1.scatter(states[::5, 0], states[::5, 1], states[::5, 2], s=sizes[::5], color=cs[::5])
    # ax1.scatter(env.centers[:,0], env.centers[:,1], env.centers[:,2], s=30, color='green')
    # ax2.scatter(env.centers[:, 1], env.centers[:, 2], env.centers[:, 3], s=30, color='green')
    # ax2 = fig.add_subplot(122, projection='3d')
    # ax2.set_xlabel('X Label')
    # ax2.set_ylabel('Y Label')
    # ax2.set_zlabel('Z Label')
    # ax2.set_xlim(-1, 1)
    # ax2.set_ylim(-1, 1)
    # ax2.set_zlim(-1, 1)
    # ax2.scatter(states[::1, 3], states[::1, 4], states[::1, 5], s=5, color=cs)

    if False:

        detector = svm.OneClassSVM(nu=0.15, kernel="rbf")
        detector.fit(states)
        y_pred = detector.predict(states)
        ax1.scatter(states[:, 0], states[:, 1], states[:, 2], s=10, color=colors[(y_pred + 1) // 2])
        # for ep in range(10):
        #     new_states = np.array([], dtype=np.int64).reshape(0, env.nbFeatures)
        #     env.reset()
        #     obj = np.random.choice(env.nbObjects)
        #     new_states = np.vstack([new_states, np.expand_dims(env.objects[obj].state, axis=0)])
        #     # cs.append(cm.hot(ep/10))
        #     for step in range(100):
        #         # p = [0.2] * 5
        #         p = [0.1, 0.6, 0.1, 0.1, 0.1]
        #         act = (obj, np.random.choice(range(5),p=p))
        #         env.step(act)
        #         state = env.objects[obj].state
        #         new_states = np.vstack([new_states, np.expand_dims(state, axis=0)])
        #         # cs.append(cm.hot(ep/10))
        #     y_pred = detector.predict(new_states)
        #     ax1.scatter(new_states[:, 0], new_states[:, 1], new_states[:, 2], s=10, color=colors[(y_pred + 1) // 2])


    plt.show(block=True)
