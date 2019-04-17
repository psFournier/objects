import numpy as np
from utils import softmax
from collections import deque

class RandomObjectSelector(object):
    def __init__(self, K, evaluator, eta):
        self.K = K
        self.evaluator = evaluator
        self.LPs = np.zeros(K)
        self.normalized_LPs = np.zeros(K)
        self.eta = eta

    def evaluate(self):
        self.last_eval = self.evaluator.evaluate()

    def update_experts(self, object, learning_progress):
        self.LPs[object] += self.eta * (learning_progress - self.LPs[object])
        self.normalized_LPs = (self.LPs - min(self.LPs)) / (max(self.LPs) - min(self.LPs))

    def update(self, object):
        eval = self.evaluator.evaluate()
        learning_progress = self.last_eval - eval
        self.update_experts(object, learning_progress)

    @property
    def stats(self):
        stats = {}
        for i, lp in enumerate(self.LPs):
            stats['lp_{}'.format(i)] = lp
        for i, prob in enumerate(self.probs):
            stats['prob_{}'.format(i)] = prob
        stats['last_eval'] = self.last_eval
        return stats

    @property
    def probs(self):
        return 1 / self.K * np.ones(self.K)


class EXP4(object):
    def __init__(self, K, evaluator, gamma, beta, eta):
        self.gamma = gamma
        self.K = K
        self.eta = eta
        self.beta = beta
        self.experts_weights = np.array([1, 1], dtype='float32')
        self.LPs = np.zeros(K)
        self.normalized_LPs = np.zeros(K)
        self.expert_probs = np.vstack([softmax(self.LPs, theta=beta),
                                       1 / K * np.ones(K)])
        self.evaluator = evaluator
        self.last_eval = 0
        self.lp_list = deque(maxlen=10)

    def evaluate(self):
        self.last_eval = self.evaluator.evaluate()

    def update(self, object):
        eval = self.evaluator.evaluate()
        learning_progress = self.last_eval - eval
        self.lp_list.append(learning_progress)
        self.update_experts(object, learning_progress)

    def update_experts(self, object, learning_progress):
        self.LPs[object] += self.eta * (learning_progress - self.LPs[object])
        self.normalized_LPs = (self.LPs - min(self.LPs)) / (max(self.LPs) - min(self.LPs))
        for i, w in enumerate(self.experts_weights):
            a = self.gamma * (learning_progress/np.max(self.lp_list)) * self.expert_probs[i][object]
            b = self.K * self.probs[object]
            self.experts_weights[i] *= np.exp(a / b)
        self.expert_probs[0] = softmax(self.normalized_LPs - min(self.normalized_LPs), theta=self.beta)

    @property
    def probs(self):
        weights_sum = np.sum(self.experts_weights)
        probs = (1 - self.gamma) * np.dot(self.experts_weights, self.expert_probs) / weights_sum
        probs += self.gamma / self.K
        probs /= probs.sum()
        return probs

    @property
    def stats(self):
        stats = {}
        for i, lp in enumerate(self.LPs):
            stats['lp_{}'.format(i)] = lp
        for i, prob in enumerate(self.probs):
            stats['prob_{}'.format(i)] = prob
        for name, weight in zip(['lp', 'uni'], self.experts_weights):
            stats['{}_weights'.format(name)] = weight
        stats['last_eval'] = self.last_eval
        return stats

class EXP4_SSP(object):
    def __init__(self, K, evaluator, gamma, beta, eta):
        self.gamma = gamma
        self.K = K
        self.eta = eta
        self.beta = beta
        self.experts_weights = np.array([1, 1], dtype='float32')
        self.LPs = np.zeros(K)
        self.normalized_LPs = np.zeros(K)
        self.expert_probs = np.vstack([softmax(self.LPs, theta=beta),
                                       1 / K * np.ones(K)])
        self.evaluator = evaluator
        self.last_eval = 0

    def evaluate(self):
        self.last_eval = self.evaluator.evaluate()

    def update(self, object):
        eval = self.evaluator.evaluate()
        learning_progress = self.last_eval - eval
        self.update_experts(object, learning_progress)

    def update_experts(self, object, learning_progress):
        normalized_lp = (learning_progress - min(self.LPs)) / (max(self.LPs) - min(self.LPs))
        for i, w in enumerate(self.experts_weights):
            a = self.gamma * normalized_lp * self.expert_probs[i][object]
            b = self.K * self.probs[object]
            self.experts_weights[i] *= np.exp(a / b)
        self.LPs[object] += self.eta * (learning_progress - self.LPs[object])
        self.normalized_LPs = (self.LPs - min(self.LPs)) / (max(self.LPs) - min(self.LPs))
        self.expert_probs[0] = softmax(self.normalized_LPs - min(self.normalized_LPs), theta=self.beta)

    @property
    def probs(self):
        weights_sum = np.sum(self.experts_weights)
        probs = (1 - self.gamma) * np.dot(self.experts_weights, self.expert_probs) / weights_sum
        probs += self.gamma / self.K
        probs /= probs.sum()
        return probs

    @property
    def stats(self):
        stats = {}
        for i, lp in enumerate(self.LPs):
            stats['lp_{}'.format(i)] = lp
        for i, prob in enumerate(self.probs):
            stats['prob_{}'.format(i)] = prob
        for name, weight in zip(['lp', 'uni'], self.experts_weights):
            stats['{}_weights'.format(name)] = weight
        stats['last_eval'] = self.last_eval
        return stats