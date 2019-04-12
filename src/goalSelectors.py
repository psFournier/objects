import numpy as np

class Uniform_goal_selector(object):
    def __init__(self, nbFeatures):
        self.nbFeatures = nbFeatures

    def select(self, object):
        return np.random.uniform(-1, 1, self.nbFeatures)

class Buffer_goal_selector(object):
    def __init__(self, buffer):
        self.buffer = buffer

    def select(self, object):
        rnd_exp_from_object = self.buffer.sample(1, object)
        goal = rnd_exp_from_object[0]['s1']
        return goal