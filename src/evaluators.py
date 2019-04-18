import numpy as np

class State_variance_evaluator(object):
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

class Qval_evaluator(object):
    def __init__(self, buffer, model):
        self.buffer = buffer
        self.model = model

    def evaluate(self):
        transitions = self.buffer.sample(200)
        states0 = np.vstack([t['s0'] for t in transitions])
        goals = np.vstack([t['g'] for t in transitions])
        qvals = self.model.compute_qvals(states0, goals)
        return np.mean(qvals)

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