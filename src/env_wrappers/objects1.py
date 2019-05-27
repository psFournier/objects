import numpy as np
from gym import Wrapper

class Objects1(Wrapper):
    def __init__(self, env, args):
        super(Objects1, self).__init__(env)
        self.gamma = 0.99
        self.rNotTerm = -1 + (self.gamma - 1) * float(args['--initq'])
        self.rTerm = 0 - float(args['--initq'])

    def get_state(self, object, state):
        start = object * self.env.nbFeatures
        object_state = state[start:start + self.env.nbFeatures]
        return object_state

    def get_r(self, s, g):
        s = s.reshape(-1, self.env.nbFeatures)[:, self.goal_idxs]
        g = g.reshape(-1, self.goal_dim)
        # idxs = self.goal_idxs
        # avgs = self.env.avgs[idxs]
        # spans = self.env.spans[idxs]
        # s = s * spans + avgs
        # g = g * spans + avgs
        diff = s - g
        d = np.linalg.norm(diff, axis=-1)
        t = (d < 0.01)
        r = t * self.rTerm + (1 - t) * self.rNotTerm
        return r, t

    @property
    def state_dim(self):
        return self.env.nbFeatures

    @property
    def goal_dim(self):
        return 1

    @property
    def action_dim(self):
        return self.env.nbActions

    @property
    def goal_idxs(self):
        return np.array([2])