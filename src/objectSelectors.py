import numpy as np
from utils import softmax

class RandomObjectSelector(object):
    def __init__(self, K, evaluator):
        self.K = K
        self.evaluator = evaluator

    def evaluate(self):
        self.last_eval = self.evaluator.evaluate()

    def update(self, object):
        pass

    @property
    def stats(self):
        return {'last_eval': self.last_eval}

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
        self.expert_probs = np.vstack([softmax(self.LPs, theta=beta),
                                       1 / K * np.ones(K)])
        self.evaluator = evaluator
        self.last_eval = 0

    def evaluate(self):
        self.last_eval = self.evaluator.evaluate()

    def update(self, object):
        eval = self.evaluator.evaluate()
        learning_progress = np.abs(eval - self.last_eval)
        self.update_experts(object, learning_progress)

    def update_experts(self, object, learning_progress):
        for i, w in enumerate(self.experts_weights):
            a = self.gamma * learning_progress * self.expert_probs[i][object]
            b = self.K * self.probs[object]
            self.experts_weights[i] *= np.exp(a / b)
        self.LPs[object] += self.eta * (learning_progress - self.LPs[object])
        self.expert_probs[0] = softmax(self.LPs - min(self.LPs), theta=self.beta)

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
        for i, prob in enumerate(self.expert_probs[0]):
            stats['lp_prob_{}'.format(i)] = prob
        for name, weight in zip(['lp', 'uni'], [self.experts_weights]):
            stats['{}_weights'.format(name)] = weight
        stats['last_eval'] = self.last_eval
        return stats