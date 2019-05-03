import numpy as np
from scipy import sparse
def softmax(X, theta=1.0, axis=None):
    """
    Compute the softmax of each element along an axis of X.

    Parameters
    ----------
    X: ND-Array. Probably should be floats.
    theta (optional): float parameter, used as a multiplier
        prior to exponentiation. Default = 1.0
    axis (optional): axis to compute values along. Default is the
        first non-singleton axis.

    Returns an array the same size as X. The result will sum to 1
    along the specified axis.
    """

    # make X at least 2d
    y = np.atleast_2d(X)

    # find axis
    if axis is None:
        axis = next(j[0] for j in enumerate(y.shape) if j[1] > 1)

    # multiply y against the theta parameter,
    y = y * float(theta)

    # subtract the max for numerical stability
    y = y - np.expand_dims(np.max(y, axis=axis), axis)

    # exponentiate y
    y = np.exp(y)

    # take the sum along the specified axis
    ax_sum = np.expand_dims(np.sum(y, axis=axis), axis)

    # finally: divide elementwise
    p = y / ax_sum

    # flatten if X was 1D
    if len(X.shape) == 1: p = p.flatten()

    return p

def merge_two_dicts(x, y):
    z = x.copy()   # start with x's keys and values
    z.update(y)    # modifies z with y's keys and values & returns None
    return z

class FeedForwardNetwork:

    def __init__(self, nIn, nHiddens, nOut, rng, density, amplitude):

        self.nPerLayer = [nIn] + nHiddens + [nOut]
        self.weights = []
        self.activations = []
        self.outputs = []
        for nI, nO in zip(self.nPerLayer[:-1], self.nPerLayer[1:]):
            self.weights.append(sparse.random(nO, nI + 1,
                                              density=density,
                                              random_state=rng,
                                              data_rvs=lambda s: rng.uniform(-amplitude, amplitude, size=s)).toarray())
            self.activations.append(np.zeros((nO, 1), dtype=float))
            self.outputs.append(np.zeros((nI + 1, 1), dtype=float))
        self.outputs.append(np.zeros((nOut,), dtype=float))

    def forward(self, input):

        self.outputs[0][:-1, 0] = input
        self.outputs[0][-1:, 0] = .1

        l = 0
        for W in self.weights[:-1]:
            self.activations[l] = np.dot(W, self.outputs[l])
            self.outputs[l + 1][:-1, :] = np.tanh(self.activations[l])
            self.outputs[l + 1][-1:, :] = .1
            l += 1

        self.activations[l] = np.dot(self.weights[l], self.outputs[l])
        self.outputs[l + 1] = np.tanh(self.activations[l])
        return self.outputs[-1]

class CompetenceQueue():
    def __init__(self, window = 100, maxlen=250):
        self.window = window
        self.C = deque(maxlen=maxlen)
        self.C_avg = deque(maxlen=10)
        self.C_avg.append(0)
        self.CP = 0
        self.init_stat()

    def init_stat(self):
        self.envstep = 0
        # self.trainstep = 0
        # self.trainstepT = 0
        # self.attempt = 0
        # self.tutorsample = 0
        # self.terminal = 0

    def process_ep(self, episode, term):
        self.C.append(term)
        self.envstep += len(episode)
        # self.attempt += 1

    # def process_samples(self, samples):
    #     self.trainstep += 1
    #     self.terminal += np.mean(samples['t'])
    #     self.tutorsample += np.mean(samples['o'])
    #
    # def process_samplesT(self, samples):
    #     self.trainstepT += 1

    def update(self):
        size = len(self.C)
        if size > 2:
            window = min(size, self.window)
            self.C_avg.append(np.mean(list(self.C)[-window:]))
            self.CP = self.C_avg[-1] - self.C_avg[0]

    def get_stats(self):
        dict = {'C': float("{0:.3f}".format(self.C_avg[-1])),
                'CP': float("{0:.3f}".format(self.CP)),
                'envstep': float("{0:.3f}".format(self.envstep))}
        return dict

if __name__ == '__main__':
    rng = np.random.RandomState(100)
    # np.random.seed(1000)
    fnn = FeedForwardNetwork(3, [5], 3, rng)
    fnn.forward(np.array([-0.1,-0.2,0.3]))
    print(fnn.forward(np.array([-0.1,-0.2,0.3])))