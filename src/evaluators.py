import numpy as np



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
        transitions = self.buffer.sample(1000)
        states0 = np.vstack([t['s0'] for t in transitions])
        actions = np.vstack([t['a0'] for t in transitions])
        y_preds = self.model._pred([states0, actions])[0]
        y_true = np.vstack([t['s1'] for t in transitions])
        return np.mean(np.square(y_preds - y_true))

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

    def evaluate(self):
        states0 = np.random.uniform(-1, 1, size=(100, self.env.nbFeatures))
        states1 = np.reshape(np.array([]), (0, self.env.nbFeatures))
        for state in states0:
            state1 = np.clip(self.env.next_state(state), -1, 1)
            states1 = np.vstack([states1, state1])
        states0 = np.repeat(states0, self.env.nbActions, axis=0)
        actions = np.tile(np.reshape(np.arange(self.env.nbActions), (-1, 1)), (100, 1))
        y_preds = self.model._pred([states0, actions])[0]
        return np.mean(np.square(y_preds - states1))

# class Control_evaluator(object)