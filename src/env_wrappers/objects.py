import numpy as np
from gym import Wrapper

class Objects(Wrapper):
    def __init__(self, env, args):
        super(Objects, self).__init__(env)
        self.gamma = 0.99
        # self.rNotTerm = -1 + (self.gamma - 1) * float(args['--initq'])
        # self.rTerm = 0 - float(args['--initq'])
        self.rNotTerm = -1  # Careful with target clipping when changing these
        self.rTerm = 0

    def get_state(self, object, state):
        start = object * self.env.nbFeatures
        object_state = state[start:start + self.env.nbFeatures]
        return object_state

    def step(self, pairObjAction):
        self.env.step(pairObjAction)

    def get_r(self, s, g):
        diff = s.reshape(-1, self.env.nbFeatures)[:,1:2] - g.reshape(-1, 1)
        d = np.linalg.norm(diff, axis=-1)
        r = (d < 0.05) * self.rTerm + (1 - (d < 0.05)) * self.rNotTerm
        return r, np.zeros_like(r)

    # def reset(self, state):
    #     exp = {}
    #     exp['s0'] = np.expand_dims(state, axis=0)
    #     exp['g'] = self.get_g()
    #     exp['w'] = self.get_w()
    #     return exp
    #
    # def get_w(self):
    #     w = 0.01 * np.ones(self.N)
    #     obj = np.random.randint(self.N)
    #     w[obj] = 1
    #     return w / sum(w)
    #
    # def get_g(self):
    #     g = np.random.randint(1, self.env.L + 1, size=self.N)
    #     # g = np.ones(self.N)
    #     return g / self.env.L
    #
    # def get_r(self, s, g, w):
    #     pos, objs = np.split(s, [2], axis=-1)
    #     d = np.linalg.norm(np.multiply(w, objs-g), axis=-1)
    #     t = d < 0.001
    #     r = t * self.rTerm + (1 - t) * self.rNotTerm
    #     return r, t

    # def process_trajectory(self, trajectory):
    #     rParams = np.expand_dims(trajectory[-1]['rParams'], axis=0)
    #     new_trajectory = []
    #     n_changes = 0

        # Reservoir sampling for HER
        # if self.her != 0:
        #     virtual_idx = []
        #     for i, exp in enumerate(reversed(trajectory)):
        #         changes = np.where(exp['s0'][2:] != exp['s1'][2:])[0]
        #         for change in changes:
        #             n_changes += 1
        #             if len(virtual_idx) < self.her:
        #                 virtual_idx.append((i, change))
        #             else:
        #                 j = np.random.randint(0, n_changes)
        #                 if j < self.her:
        #                     virtual_idx[j] = (i, change)

        # for i, exp in enumerate(reversed(trajectory)):
        #     if i == 0:
        #         exp['next'] = None
        #     else:
        #         exp['next'] = trajectory[-i]
        #
        #     # if self.her != 0:
        #     #     virtual_goals = [np.hstack([trajectory[idx]['s1'], self.vs[c]]) for idx, c in virtual_idx if idx >= i]
        #     #     exp['goal'] = np.vstack([trajectory[-1]['goal']] + virtual_goals)
        #     # else:
        #     #     exp['goal'] = np.expand_dims(trajectory[-1]['goal'], axis=0)
        #
        #     # Reservoir sampling for HER
        #     if self.her != 0:
        #         changes = np.where(exp['s0'][2:] != exp['s1'][2:])[0]
        #         for change in changes:
        #             n_changes += 1
        #             v = self.vs[change]
        #             if goals.shape[0] <= self.her:
        #                 goals = np.vstack([goals, np.hstack([exp['s1'], v])])
        #             else:
        #                 j = np.random.randint(1, n_changes + 1)
        #                 if j <= self.her:
        #                     goals[j] = np.hstack([exp['s1'], v])
        #
        #     exp['rParams'] = rParams
        #     # exp['reward'], exp['terminal'] = self.get_r(exp['s1'], exp['rParams'])
        #     new_trajectory.append(exp)
        #
        # return new_trajectory

    @property
    def state_dim(self):
        return self.env.nbFeatures

    @property
    def goal_dim(self):
        return 1

    @property
    def action_dim(self):
        return self.env.nbActions
