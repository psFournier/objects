import numpy as np

class Uniform_goal_selector(object):
    def __init__(self, agent):
        self.nbFeatures = agent.env.nbFeatures

    def select(self, object):
        return np.random.uniform(-1, 1, self.nbFeatures)

class No_goal_selector(object):
    def __init__(self, agent):
        self.nbFeatures = agent.env.nbFeatures

    def select(self, object):
        return np.array([])

class Buffer_goal_selector(object):
    def __init__(self, agent):
        self.buffer = agent.buffer
        self.nbFeatures = agent.env.nbFeatures

    def select(self, object):
        try:
            rnd_exp_from_object = self.buffer.sample(1, object)
            goal = rnd_exp_from_object[0]['s1']
        except:
            goal = np.random.uniform(-1, 1, self.nbFeatures)
        return goal
