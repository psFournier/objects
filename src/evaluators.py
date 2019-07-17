import numpy as np
from actionSelectors import State_goal_max_action_selector
from players import Player
from environments.objects import Obj

class TDerror_evaluator():
    def __init__(self, agent):
        self.agent = agent
        self.name = 'tderror_eval'
        self.history = []

    def get_reward(self):
        transitions = self.agent.buffer.sample(200)
        states0 = np.vstack([t['s0'] for t in transitions])
        goals = np.vstack([t['g'] for t in transitions])
        actions = np.vstack([t['a0'] for t in transitions])
        states1 = np.vstack([t['s1'] for t in transitions])
        rewards, terminals = self.agent.wrapper.get_r(states1, goals)
        qvals = self.agent.model._qval([states0, goals, actions])[0]
        targetqvals = self.agent.model._targetqvals([states1, goals])[0]
        targetqvals = rewards + (1 - terminals) * np.max(targetqvals, axis=1)
        reward = np.sqrt(np.mean(np.square(qvals - targetqvals)))
        return reward

class Test_generalisation():
    def __init__(self, agent):
        self.agent = agent
        self.name = 'test_generalization'
        self.action_selector = State_goal_max_action_selector(agent)
        self.player = Player(agent)
        self.rewards = agent.ep_env_steps * agent.wrapper.rNotTerm * np.ones(2)
        self.stat_steps = np.zeros(2)

    def get_reward(self):

        nbep = 10
        # Beware : here using Obj from (potentially) another env def

        for _ in range(nbep):
            obj = Obj(self.agent.env,
                      -1,
                      np.random.uniform(1.1,1.3))
            _, r, episodes = self.player._play(obj, self.agent.goal_selector, self.action_selector)
            if self.stat_steps[0] == 0:
                self.rewards[0] = r
            else:
                self.rewards[0] += r
            self.stat_steps[0] += episodes

        for _ in range(nbep):
            obj = Obj(self.agent.env,
                      -1,
                      np.random.uniform(1.7, 1.9))
            _, r, episodes = self.player._play(obj, self.agent.goal_selector, self.action_selector)
            if self.stat_steps[1] == 0:
                self.rewards[1] = r
            else:
                self.rewards[1] += r
            self.stat_steps[1] += episodes

    def stats(self):
        d = {}
        for i, s in enumerate(self.stat_steps):
            if s != 0:
                self.rewards[i] /= s
            self.stat_steps[i] = 0
        d['rewards'] = self.rewards
        return d

class Test_generalisation_one_evaluator():
    def __init__(self, agent):
        self.agent = agent
        self.name = 'test_gene_one_eval'
        self.action_selector = State_goal_max_action_selector(agent)
        self.player = Player(agent)

    def get_reward(self):
        reward = 0
        nbep = 1
        # Beware : here using Obj from another env def
        for _ in range(nbep):
            obj = Obj(self.agent.env,
                      self.agent.env.nbObjects,
                      self.agent.env.fixed_feature_ranges[:, 1])
            _, r = self.player.play(obj, self.agent.goal_selector, self.action_selector)
            reward += r
        reward /= nbep
        return reward

    def stats(self):
        return self.player.stats()

class Test_generalisation_all_evaluator():
    def __init__(self, agent):
        self.agent = agent
        self.name = 'test_gene_all_eval'
        self.action_selector = State_goal_max_action_selector(agent)
        self.player = Player(agent)

    def get_reward(self):
        reward = 0
        nbep = 5
        for i in range(nbep):
            obj = Obj(self.agent.env,
                      self.agent.env.nbObjects,
                      np.random.uniform(low=self.agent.env.fixed_feature_ranges[:, 0],
                                        high=self.agent.env.fixed_feature_ranges[:, 1]))
            _, r = self.player.play(obj, self.agent.goal_selector, self.action_selector)
            reward += r
        reward /= nbep
        return reward

    def stats(self):
        return self.player.stats()

class qval_evaluator():
    def __init__(self, agent):
        self.agent = agent
        self.name = 'qval_eval'
        self.action_selector = State_goal_max_action_selector(agent)
        self.player = Player(agent)

    def get_reward(self):
        reward = 0
        nbep = 1
        for _ in range(nbep):
            obj = Obj(self.agent.env,
                      self.agent.env.nbObjects,
                      np.random.uniform(low=self.agent.env.fixed_feature_ranges[:, 0],
                                        high=self.agent.env.fixed_feature_ranges[:,1]))
            _, r = self.player.play(obj, self.agent.goal_selector, self.action_selector)
            reward += r
        reward /= nbep
        return reward

    def stats(self):
        return self.player.stats()

class Test_episode_evaluator():
    def __init__(self, agent):
        self.agent = agent
        self.name = 'test_ep_eval'
        self.action_selector = State_goal_max_action_selector(agent)
        self.player = Player(agent)

    def get_reward(self):
        reward = 0
        if self.agent.object_selector.K < self.agent.env.nbObjects:
            for _ in range(10):
                object = np.random.randint(self.agent.object_selector.K, self.agent.env.nbObjects)
                _, r = self.player.play(object, self.agent.goal_selector, self.action_selector)
                reward += r
            reward /= (self.agent.env.nbObjects - self.agent.object_selector.K)
        else:
            reward = 0
        return reward

    def stats(self):
        return self.player.stats()

class Train_episode_evaluator(object):
    def __init__(self, agent):
        self.agent = agent
        self.name = 'train_ep_eval'
        self.reward = 0
        self.action_selector = State_goal_max_action_selector(agent)
        self.player = Player(agent)

    def get_reward(self):
        reward = 0
        for _ in range(10):
            object = np.random.randint(self.agent.object_selector.K)
            _, r = self.player.play(object, self.agent.goal_selector, self.action_selector)
            reward += r
        reward /= self.agent.object_selector.K
        return reward

    def stats(self):
        return self.player.stats()


class Reached_states_variance_evaluator(object):
    def __init__(self, agent):
        self.last_eval = 0
        self.buffer = agent.buffer

    def evaluate(self):
        reached_states_all = []
        for buffer in self.buffer._buffers:
            if buffer._numsamples != 0:
                reached_states_all.append(
                    np.vstack([self.buffer._storage[idx]['s1'] for idx in buffer._storage])
                )
            else:
                reached_states_all.append(0)
        vars = [np.var(reached_states) for reached_states in reached_states_all]
        return np.sum(vars)

    def get_reward(self):
        eval = self.evaluate()
        reward = eval - self.last_eval
        self.last_eval = eval
        return reward

    def stats(self):
        return {'eval': self.last_eval}

class Reached_goals_variance_evaluator(object):
    def __init__(self, agent):
        self.last_eval = 0
        self.agent = agent
        self.reached_goals = [np.reshape(np.array([]), (0, agent.env.nbFeatures))
                              for _ in range(agent.env.nbObjects)]

    def get_reward(self, transitions):
        for t in transitions:
            r, t = self.agent.wrapper.get_r(t['s1'], t['g'])
            if r == self.agent.wrapper.rTerm:
                self.reached_goals[t['object']] = np.vstack([self.reached_goals[t['object']], t['g']])
        vars = [np.var(reached_goals) if reached_goals.size!=0 else 0 for reached_goals in self.reached_goals]
        eval = np.sum(vars)
        reward = eval - self.last_eval
        self.last_eval = eval
        return reward

    @property
    def stats(self):
        return {'eval': self.last_eval}

# class Reward_evaluator(object):
#     def __init__(self, agent):
#         self.last_eval = 0
#         self.agent = agent
#         self.name = 'train_reward'
#
#     def get_reward(self):
#         eval = 0
#         for t in self.agent.last_ep:
#             r, t = self.agent.wrapper.get_r(t['s1'], t['g'])
#             eval += r[0]
#         reward = eval - self.last_eval
#         self.last_eval = eval
#         return reward
#
#     @property
#     def stats(self):
#         return {'eval': self.last_eval}

class Qval_evaluator(object):
    def __init__(self, agent):
        self.agent = agent
        self.last_eval = 0

    def get_reward(self, transitions):
        tr = self.agent.buffer.sample(200)
        states0 = np.vstack([t['s0'] for t in tr])
        goals = np.vstack([t['g'] for t in tr])
        #Normalize
        qvals = self.agent.model._qvals([states0, goals])[0]
        eval = np.mean(qvals)
        reward = eval - self.last_eval
        self.last_eval = eval
        return reward

    @property
    def stats(self):
        return {'eval': self.last_eval}



class ApproxError_buffer_evaluator(object):
    def __init__(self, buffer, model):
        self.buffer = buffer
        self.model = model

    def evaluate(self):
        transitions = self.buffer.sample(5000)
        states0 = np.vstack([t['s0'] for t in transitions])
        actions = np.vstack([t['a0'] for t in transitions])
        y_preds = self.model._pred([states0, actions])[0]
        y_true = np.vstack([t['s1'] for t in transitions])
        return np.mean(np.square(y_preds - (y_true - states0)))

class ApproxError_objects_evaluator(object):
    def __init__(self, env, model):
        self.env = env
        self.model = model

    def evaluate(self):
        states0 = []
        for object in self.env.objects:
            for _ in range(5):
                state = object.state
                for _ in range(20):
                    state = np.clip(np.dot(self.env.As[np.random.randint(self.env.nbActions)], state), -1, 1)
                states0 += [state]
        states0 = np.vstack(states0)
        states1 = np.reshape(np.array([]), (0, self.env.nbFeatures))
        for state in states0:
            state1 = np.clip(np.dot(self.env.As, state), -1, 1)
            states1 = np.vstack([states1, state1])
        states0 = np.repeat(states0, self.env.nbActions, axis=0)
        actions = np.tile(np.reshape(np.arange(self.env.nbActions), (-1, 1)), (self.env.nbObjects * 5, 1))
        y_preds = self.model._pred([states0, actions])[0]
        return np.mean(np.square(y_preds - states1))

class ApproxError_global_evaluator(object):
    def __init__(self, env, model):
        self.env = env
        self.model = model
        self.states0 = np.random.uniform(-1, 1, size=(1000, self.env.nbFeatures))
        self.actions = np.tile(np.reshape(np.arange(self.env.nbActions), (-1, 1)), (1000, 1))
        self.states1 = np.reshape(np.array([]), (0, self.env.nbFeatures))
        for state in self.states0:
            state1 = np.clip(self.env.next_state(state), -1, 1)
            self.states1 = np.vstack([self.states1, state1])
        self.states0 = np.repeat(self.states0, self.env.nbActions, axis = 0)

    def evaluate(self):
        y_preds = self.model._pred([self.states0, self.actions])[0]
        return np.mean(np.square(y_preds - (self.states1 - self.states0)))

class ApproxError_changes_evaluator(object):
    def __init__(self, env, model):
        self.env = env
        self.model = model
        self.states0 = np.random.uniform(-1, 1, size=(10000, self.env.nbFeatures))
        self.actions = np.tile(np.reshape(np.arange(self.env.nbActions), (-1, 1)), (10000, 1))
        self.states1 = np.reshape(np.array([]), (0, self.env.nbFeatures))
        for state in self.states0:
            for action in range(self.env.nbActions):
                state1 = np.clip(self.env.next_state(state, action), -1, 1)
                self.states1 = np.vstack([self.states1, state1])
        self.states0 = np.repeat(self.states0, self.env.nbActions, axis = 0)
        changes = np.where(np.any(self.states0 - self.states1 != 0, axis=1))
        self.states0 = self.states0[changes]
        self.states1 = self.states1[changes]
        self.actions = self.actions[changes]

    def evaluate(self):
        y_preds = self.model._pred([self.states0, self.actions])[0]
        return np.mean(np.square(y_preds - (self.states1 - self.states0)))


# class Control_evaluator(object)