import numpy as np
from utils import softmax

class Random_action_selector(object):
    def __init__(self, agent):
        self.nbActions = agent.env.nbActions

    def select(self, state, goal):
        return np.random.choice(self.nbActions), None, np.ones(self.nbActions)/self.nbActions

class State_goal_max_action_selector(object):
    def __init__(self, agent):
        self.agent = agent
        self.min_max = 0
        self.min_max_prob = 0
        self.max_max_prob = 0
        self.stat_steps = 0
        self.name = 'sgmax_action'

    def select(self, state, goal):
        input = [np.expand_dims(state, axis=0), np.expand_dims(goal, axis=0)]
        qvals = self.agent.model._qvals(input)[0].squeeze()
        probs = np.zeros(qvals.shape)
        self.min_max += max(qvals) - min(qvals)
        action = np.argmax(qvals)
        # print(state, goal)
        probs[action] = 1
        self.stat_steps += 1
        return action, qvals, probs

    def stats(self):
        d = {'min_max': self.min_max / self.stat_steps}
        self.min_max = 0
        self.min_max_prob = 0
        self.stat_steps = 0
        return d

class Epsilon_greedy_action_selector(object):
    def __init__(self, agent):
        self.agent = agent
        self.min_max = 0
        self.min_max_prob = 0
        self.max_max_prob = 0
        self.stat_steps = 0
        self.name = 'sgmax_action'

    def select(self, state, goal):
        input = [np.expand_dims(state, axis=0), np.expand_dims(goal, axis=0)]
        qvals = self.agent.model._qvals(input)[0].squeeze()
        probs = np.zeros(qvals.shape)
        self.min_max += max(qvals) - min(qvals)
        eps = (1 - 0.9*min(1, sum(self.agent.train_steps)/10000))
        if np.random.rand() < eps:
            # print(eps)
            action = np.random.randint(self.agent.env.nbActions)
        else:
            action = np.argmax(qvals)
        # print(state, goal, qvals)
        probs[action] = 1
        self.stat_steps += 1
        return action, qvals, probs

    def stats(self):
        d = {'min_max': self.min_max / self.stat_steps}
        self.min_max = 0
        self.min_max_prob = 0
        self.stat_steps = 0
        return d

class State_goal_soft_action_selector(object):
    def __init__(self, agent):
        self.agent = agent
        self.min_max = 0
        self.min_max_prob = 0
        self.max_max_prob = 0
        self.stat_steps = 0
        self.name = 'sgsoft_action'

    def select(self, state, goal):
        input = [np.expand_dims(state, axis=0), np.expand_dims(goal, axis=0)]
        qvals = self.agent.model._qvals(input)[0].squeeze()
        self.min_max += max(qvals) - min(qvals)
        # probs = softmax(qvals, theta=0.5 + 1.5*min(1, sum(self.agent.train_steps)/10000))
        # print(state, goal)
        probs = softmax(qvals, theta=2)

        sorted_probs = np.sort(probs)
        self.min_max_prob += sorted_probs[-1] - sorted_probs[0]
        self.max_max_prob += sorted_probs[-1] - sorted_probs[-2]
        action = np.random.choice(len(probs), p=probs)
        self.stat_steps += 1
        return action, qvals, probs

    def stats(self):
        d = {'min_max': self.min_max / self.stat_steps,
             'min_max_prob': self.min_max_prob / self.stat_steps,
             'max_max_prob': self.max_max_prob / self.stat_steps}
        self.min_max = 0
        self.min_max_prob = 0
        self.max_max_prob = 0
        self.stat_steps = 0
        return d

class State_action_selector(object):
    def __init__(self, agent):
        self.model = agent.model

    def select(self, state, goal):
        input = [np.expand_dims(state, axis=0)]
        qvals = self.model._qvals(input)[0].squeeze()
        probs = softmax(qvals, theta=1)
        action = np.random.choice(len(probs), p=probs)
        return action, probs