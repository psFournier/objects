import numpy as np

class Uniform_goal_selector(object):
    def __init__(self, agent):
        self.agent = agent
        self.name = 'random_goal'

    def select(self, object):
        while True:
            goal = np.random.uniform(self.agent.wrapper.goal_space[0, 0],
                                     self.agent.wrapper.goal_space[0, 1],
                                     self.agent.wrapper.goal_dim)
            state = self.agent.wrapper.get_state(object, self.agent.env.state)
            if not self.agent.wrapper.get_r(state, goal)[1]:
                break
        return goal

    @property
    def stats(self):
        d = {}
        return d

class Constant_goal_selector(object):
    def __init__(self, agent):
        self.agent = agent
        self.name = 'constant_goal'

    def select(self, object):
        goal = np.array([0.5])
        return goal

    @property
    def stats(self):
        d = {}
        return d

class No_goal_selector(object):
    def __init__(self, agent):
        self.agent = agent
        self.name = 'no_goal'

    def select(self, object):
        return np.array([])

    @property
    def stats(self):
        d = {}
        return d

class Buffer_goal_selector(object):
    def __init__(self, agent):
        self.agent = agent
        self.dist_to_goal = 0
        self.stat_steps = 0
        self.name = 'buffer_goal'

    def select(self, object):
        attempts = 0
        state = self.agent.wrapper.get_state(object, self.agent.env.state)
        while True and attempts < 100:
            rnd_exp_from_object = self.agent.buffer.sample(1, object)
            goal = rnd_exp_from_object[0]['s1'][0:2]
            attempts += 1
            if self.agent.wrapper.get_r(state, goal)[0] == self.agent.wrapper.rNotTerm:
                break
        if attempts == 100:
            goal = np.random.uniform(-1, 1, 1)
        self.dist_to_goal += np.linalg.norm(state[0] - goal)
        self.stat_steps += 1
        return goal

    @property
    def stats(self):
        d = {'dist': self.dist_to_goal / self.stat_steps}
        self.stat_steps = 0
        self.dist_to_goal = 0
        return d
