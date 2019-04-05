import numpy as np
from gym import Wrapper, spaces

class Base(Wrapper):
    def __init__(self, env, args=None):
        super(Base, self).__init__(env)
        assert int(args['--targetClip']) == 0
        self.gamma = float(args['--gamma'])
        self.rNotTerm = -1 + (self.gamma - 1) * float(args['--initq'])
        self.rTerm = 0 - float(args['--initq'])
        self.nbObjects = 1

    def step(self, pairObjAction):
        self.env.step(pairObjAction[1])

    @property
    def state_dim(self):
        return self.env.observation_space.low.shape[0],

    @property
    def action_dim(self):
        return self.env.action_space.n
