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

class FeedForwardNetwork:

    def __init__(self, nIn, nHiddens, nOut):

        self.nPerLayer = [nIn] + nHiddens + [nOut]
        self.weights = []
        self.activations = []
        self.outputs = []
        for nI, nO in zip(self.nPerLayer[:-1], self.nPerLayer[1:]):
            self.weights.append(sparse.random(nO, nI + 1,
                                              density=0.5,
                                              data_rvs=lambda s: np.random.uniform(-0.1, 0.1, size=s)).toarray())
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

if __name__ == '__main__':

    np.random.seed(1000)
    fnn = FeedForwardNetwork(3, [5], 3)
    fnn.forward(np.array([-0.1,-0.2,0.3]))
    print(fnn.forward(np.array([-0.1,-0.2,0.3])))