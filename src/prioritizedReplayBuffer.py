import numpy as np
from segment_tree import SumSegmentTree, MinSegmentTree

class RingBuffer(object):
    def __init__(self, limit):
        self._storage = []
        self._limit = limit
        self._oldest = 0
        self._next = 0
        self._numsamples = 0

    def append(self, idx):
        if self._next >= len(self._storage):
            self._storage.append(idx)
            self._numsamples += 1
        else:
            self._storage[self._next] = idx
            if self._next == self._oldest and self._numsamples > 0:
                self._oldest = (self._oldest + 1) % self._limit
            else:
                self._numsamples += 1
        self._next = (self._next + 1) % self._limit

    def popOrNot(self, idx):
        if idx == self._storage[self._oldest]:
            self._oldest = (self._oldest + 1) % self._limit
            self._numsamples -= 1

    def sample(self, batch_size):
        res = []
        idxs = [np.random.randint(0, self._numsamples - 1) for _ in range(batch_size)]
        for i in idxs:
            idx = (self._oldest + i) % len(self._storage)
            res.append(self._storage[idx])
        return res

    def __len__(self):
        return len(self._storage)

# class PrioritizedRingBuffer(RingBuffer):
#     def __init__(self, limit):
#         """Create Prioritized Replay buffer.
#
#         Parameters
#         ----------
#         size: int
#             Max number of transitions to store in the buffer. When the buffer
#             overflows the old memories are dropped.
#         alpha: float
#             how much prioritization is used
#             (0 - no prioritization, 1 - full prioritization)
#
#         See Also
#         --------
#         ReplayBuffer.__init__
#         """
#         super(PrioritizedRingBuffer, self).__init__(limit=limit)
#         self.alpha = 0.5
#
#         self.epsilon = 1e-6
#
#
#         it_capacity = 1
#         while it_capacity < limit:
#             it_capacity *= 2
#
#         self._it_sum = SumSegmentTree(it_capacity)
#         self._it_min = MinSegmentTree(it_capacity)
#         self.max_priority = 1.
#
#     def append(self, idx):
#         self._it_sum[self._next] = self.max_priority ** self.alpha
#         self._it_min[self._next] = self.max_priority ** self.alpha
#         super().append(idx)
#
#     def _sample_proportional(self, batch_size):
#         res = []
#         sum = self._it_sum.sum(0, len(self._storage) - 1)
#         for _ in range(batch_size):
#             mass = np.random.random() * sum
#             idx = self._it_sum.find_prefixsum_idx(mass)
#             res.append(idx)
#         return res
#
#     def sample(self, batch_size):
#         """Sample a batch of experiences.
#
#         Parameters
#         ----------
#         batch_size: int
#             How many transitions to sample.
#         beta: float
#             To what degree to use importance weights
#             (0 - no corrections, 1 - full correction)
#         """
#         idxes = self._sample_proportional(batch_size)
#
#         weights = []
#         p_min = self._it_min.min() / self._it_sum.sum()
#         max_weight = (p_min * self._numsamples) ** (-0.4)
#
#         for idx in idxes:
#             p_sample = self._it_sum[idx] / self._it_sum.sum()
#             weight = (p_sample * self._numsamples) ** (-0.4)
#             weights.append(np.expand_dims(weight / max_weight, axis=1))
#
#         for i in idxes:
#             idx = (self._oldest + i) % len(self._storage)
#             res.append(self._storage[idx])
#
#         exps = super(PrioritizedReplayBuffer, self).sample(batch_size, idxes)
#         for exp, idx, weight in zip(exps, idxes, weights):
#             exp['indices'] = np.expand_dims(idx, axis=1)
#             exp['weights'] = np.expand_dims(weight, axis=1)
#
#         return exps


    # def update_priorities(self, idxes, priorities):
    #     """Update priorities of sampled transitions.
    #
    #     sets priority of transition at index idxes[i] in buffer
    #     to priorities[i].
    #
    #     Parameters
    #     ----------
    #     idxes: [int]
    #         List of idxes of sampled transitions
    #     priorities: [float]
    #         List of updated priorities corresponding to
    #         transitions at the sampled idxes denoted by
    #         variable `idxes`.
    #     """
    #     idxes = list(idxes.squeeze())
    #     assert len(idxes) == len(priorities)
    #     for idx, priority in zip(idxes, priorities):
    #         priority = priority + self.epsilon
    #         assert 0 <= idx < len(self._storage)
    #         self._it_sum[idx] = priority ** self.alpha
    #         self._it_min[idx] = priority ** self.alpha
    #         self.max_priority = max(self.max_priority, priority)

class ReplayBuffer(object):
    def __init__(self, limit, N):
        self._storage = []
        self._next_idx = 0
        self._limit = N*limit
        self._buffers = [RingBuffer(limit) for _ in range(N)]

    def __len__(self):
        return len(self._storage)

    def append(self, transition):
        if self._next_idx >= len(self._storage):
            self._storage.append(transition)
        else:
            object = self._storage[self._next_idx]['object']
            self._buffers[object].popOrNot(self._next_idx)
            self._storage[self._next_idx] = transition
        self._buffers[transition['object']].append(self._next_idx)
        self._next_idx = (self._next_idx + 1) % self._limit

    def sample(self, batchsize, object=None):
        idxs = []
        if object is None:
            if len(self._storage) >= batchsize:
                idxs = [np.random.randint(0, len(self._storage) - 1) for _ in range(batchsize)]
        else:
            if self._buffers[object]._numsamples >= batchsize:
                idxs = self._buffers[object].sample(batchsize)
        exps = []
        for i in idxs:
            exps.append(self._storage[i].copy())
        return exps

    @property
    def stats(self):
        stats = {}
        for i, buffer in enumerate(self._buffers):
            idxs = buffer._storage[-max(100, len(buffer)):]
            last_states = []
            for idx in idxs:
                last_states.append(self._storage[idx]['state0'])
            var = np.mean(np.var(np.array(last_states), axis=0))
            stats['var' + str(i)] = var
        return stats
