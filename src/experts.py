import numpy as np

class Reached_states_variance_maximizer_expert(object):
    def __init__(self, agent):
        self.K = agent.env.nbObjects
        self.buffer = agent.buffer
        self.probs = None
        self.name = 'rsv'

    def update_probs(self):
        reached_states_all = []
        for buffer in self.buffer._buffers:
            if buffer._numsamples != 0:
                reached_states_all.append(
                    np.vstack([self.buffer._storage[idx]['s1'] for idx in buffer._storage])
                )
            else:
                reached_states_all.append(0)
        vars = [np.var(reached_states) for reached_states in reached_states_all]
        if np.sum(vars) != 0:
            vars /= np.sum(vars)
        else:
            vars = [1 / self.K] * self.K
        self.probs = np.array(vars)

class Uniform_expert(object):
    def __init__(self, agent):
        self.K = agent.env.nbObjects
        self.probs = np.array(1 / self.K * np.ones(self.K))
        self.name = 'uni'

    def update_probs(self):
        pass