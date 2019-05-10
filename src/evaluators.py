import numpy as np
from actionSelectors import State_goal_max_action_selector

class Test_episode_evaluator(object):
    def __init__(self, agent):
        self.agent = agent
        self.last_eval = 0
        self.name = 'test_ep_eval'
        self.reward = 0
        self.stat_steps = 0
        self.action_selector = State_goal_max_action_selector(agent)

    def get_reward(self):
        eval = 0
        for object in range(self.agent.object_selector.K, self.agent.env.nbObjects):
            self.agent.env.reset()
            test_ep = self.agent.player.play(object, self.agent.goal_selector, self.action_selector)
            for t in test_ep[object]:
                r, t = self.agent.wrapper.get_r(t['s1'], t['g'])
                eval += r[0]
        if self.agent.object_selector.K < self.agent.env.nbObjects:
            eval /= (self.agent.env.nbObjects - self.agent.object_selector.K)
            reward = eval - self.last_eval
            self.last_eval = eval
            self.reward += eval
        else:
            reward = 0
        self.stat_steps += 1
        return reward

    @property
    def stats(self):
        d = {'reward': self.reward / self.stat_steps}
        self.reward = 0
        self.stat_steps = 0
        return d

class Train_episode_evaluator(object):
    def __init__(self, agent):
        self.agent = agent
        self.last_eval = 0
        self.name = 'train_ep_eval'
        self.reward = 0
        self.stat_steps = 0
        self.action_selector = State_goal_max_action_selector(agent)

    def get_reward(self):
        eval = 0
        for object in range(0, self.agent.object_selector.K):
            self.agent.env.reset()
            test_ep = self.agent.player.play(object, self.agent.goal_selector, self.action_selector)
            for t in test_ep[object]:
                r, t = self.agent.wrapper.get_r(t['s1'], t['g'])
                eval += r[0]
        eval /= self.agent.object_selector.K
        reward = eval - self.last_eval
        self.last_eval = eval
        self.reward += eval
        self.stat_steps += 1
        return reward

    @property
    def stats(self):
        d = {'reward': self.reward / self.stat_steps}
        self.reward = 0
        self.stat_steps = 0
        return d


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

    @property
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
        qvals = self.agent.model._qvals([states0, goals])[0]
        eval = np.mean(qvals)
        reward = eval - self.last_eval
        self.last_eval = eval
        return reward

    @property
    def stats(self):
        return {'eval': self.last_eval}

class TDerror_evaluator(object):
    def __init__(self, buffer, model, wrapper):
        self.buffer = buffer
        self.model = model
        self.wrapper = wrapper

    def evaluate(self):
        transitions = self.buffer.sample(200)
        states0 = np.vstack([t['s0'] for t in transitions])
        goals = np.vstack([t['g'] for t in transitions])
        actions = np.vstack([t['a0'] for t in transitions])
        qvals = self.model._qval([states0, goals, actions])[0]
        states1 = np.vstack([t['s1'] for t in transitions])
        targetqvals = self.model._targetqvals([states1, goals])[0]
        rewards, terminals = self.wrapper.get_r(states1, goals)
        targetqvals = rewards + (1 - terminals) * np.max(targetqvals, axis=1)
        return np.mean(np.square(qvals - targetqvals))

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