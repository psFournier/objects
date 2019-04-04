import numpy as np
from prioritizedReplayBuffer import ReplayBuffer

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

class Agent(object):
    def __init__(self, args, wrapper, model):
        self.wrapper = wrapper
        self.model = model
        self._gamma = 0.99
        self._lambda = float(args['--lambda'])
        self.nstep = int(args['--nstep'])
        self.IS = args['--IS']
        self.buffer = ReplayBuffer(limit=int(5e5), N=self.wrapper.nbObjects)
        self.batch_size = 64
        self.train_step = 0
        self.current_exp = {}
        self.current_trajectory = []
        self.update_stats = {'offpolicyness': 0, 'loss': 0, 'target': 0, 'qval': 0, 'utility': 0, 'term': 0, 'tderror':0, 'update': 0, 'min_prob':0, 'max_prob':0, 'regret':0}
        self.env_stats = {'reward'}

    def pick_object(self):
        self.current_object = np.random.choice(self.wrapper.nbObjects)
        return self.current_object

    def choose_action(self):
        self.current_action = np.random.choice(self.wrapper.action_dim)
        return self.current_action

    # def end_episode(self):
    #     l = len(self.current_trajectory)
    #     utilities = np.empty(self.wrapper.N)
    #     n_changes = 0
    #     virtual = np
    #
    #     for i, exp in enumerate(self.current_trajectory):
    #         # if i == 0:
    #         #     exp['prev'] = None
    #         # else:
    #         #     exp['prev'] = trajectory[i - 1]
    #         if i == l - 1:
    #             exp['next'] = None
    #         else:
    #             exp['next'] = self.current_trajectory[i + 1]
    #
    #     # TODO utility = 1 automatiquement pour le but poursuivi ?
    #
    #     for exp in reversed(self.current_trajectory):
    #
    #         # Reservoir sampling for HER
    #         changes = np.where(exp['s0'][2:] != exp['s1'][2:])[0]
    #         for change in changes:
    #             n_changes += 1
    #             v = self.vs[change]
    #             if goals.shape[0] <= self.her:
    #                 goals = np.vstack([goals, np.hstack([exp['s1'], v])])
    #             else:
    #                 j = np.random.randint(1, n_changes + 1)
    #                 if j <= self.her:
    #                     goals[j] = np.hstack([exp['s1'], v])
    #
    #         changes = np.where(exp['s0'][0, 2:] != exp['s1'][0, 2:])[0]
    #         utilities[changes] = 1
    #
    #         goals[changes] = exp['s1'][0, 2:][changes]
    #         exp['u'] = utilities / (sum(utilities) + 0.00001)
    #         exp['vg'] = goals
    #
    #         processed.append(exp)
    #
    # for exp in trajectory:
    #         self.buffer.append(exp)
    #     self.current_trajectory.clear()

    def reset(self, state, w=None, g=None):
        self.current_exp['s0'] = np.expand_dims(state, axis=0)
        if w is None:
            self.current_exp['w'] = np.expand_dims(self.wrapper.get_w(), axis=0)
        else:
            self.current_exp['w'] = np.expand_dims(w, axis=0)
        if g is None:
            self.current_exp['g'] = np.expand_dims(self.wrapper.get_g(), axis=0)
        else:
            self.current_exp['g'] = np.expand_dims(g, axis=0)

    def act(self):
        qvals = self.model.qvals([self.current_exp['s0'],
                                  self.current_exp['g'],
                                  self.current_exp['w']])[0].squeeze()
        # theta = max(0, 10*self.train_step / 5e4)
        probs = softmax(qvals, theta=1)
        self.update_stats['min_prob'] += min(probs)
        self.update_stats['max_prob'] += max(probs)
        action = np.random.choice(range(qvals.shape[0]), p=probs)
        self.update_stats['regret'] += probs[self.wrapper.env.opt_action(0)[0]] / probs[action]
        self.current_exp['mu0'] = probs[action]
        self.current_exp['a0'] = np.expand_dims(action, axis=1)
        return action

    def step(self, state, r, term):
        self.current_exp['s1'] = np.expand_dims(state, axis=0)
        #TODO catch not defined in base env and set r and term to env values
        self.current_exp['reward'], self.current_exp['terminal'] = self.wrapper.get_r(self.current_exp['s1'],
                                                                             self.current_exp['g'],
                                                                             self.current_exp['w'])
        self.current_trajectory.append(self.current_exp.copy())
        self.current_exp['s0'] = np.expand_dims(state, axis=0)
        return self.current_exp['reward'], self.current_exp['terminal']


    def train(self):
        return
        # exps = self.buffer.sample(self.batch_size)
        # nStepExpes = self.getNStepSequences(exps)
        # nStepExpes = self.getQvaluesAndBootstraps(nStepExpes)
        # states, actions, goals, weights, targets, ros = self.getTargetsSumTD(nStepExpes)
        # inputs = [states, actions, goals, weights, targets]
        # loss, qval, td_errors = self.model.train(inputs)
        # self.train_step += 1
        # self.update_stats['update'] += 1
        # self.update_stats['target'] += np.mean(targets)
        # self.update_stats['offpolicyness'] += np.mean(ros)
        # self.update_stats['loss'] += loss
        # self.update_stats['qval'] += np.mean(qval)
        # self.update_stats['tderror'] += np.mean(td_errors)
        # self.model.target_train()

    def getNStepSequences(self, exps):
        nStepSeqs = []
        for exp in exps:
            nStepSeq = [exp]
            i = 1
            while i <= self.nstep - 1 and exp['next'] != None:
                exp = exp['next']
                nStepSeq.append(exp)
                i += 1
            nStepSeqs.append(nStepSeq)
        return nStepSeqs

    def getQvaluesAndBootstraps(self, nStepExpes, g=None, w=None):

        states, gs, ws = [], [], []
        for nStepExpe in nStepExpes:

            first = nStepExpe[0]
            utility = first['u'].any()
            if utility:
                self.update_stats['utility'] += 1
            l = len(nStepExpe)
            if g is None or w is None:
                if utility and np.random.rand() < 0.75:
                    g, w = first['vg'], first['u']
                else:
                    g, w = first['g'], first['w']
            gs += [g] * (l+1)
            ws += [w] * (l+1)

            for exp in nStepExpe:
                states.append(exp['s0'])
            states.append(nStepExpe[-1]['s1'])

        states = np.vstack(states)
        gs = np.vstack(gs)
        ws = np.vstack(ws)

        rewards, terminals = self.wrapper.get_r(states, gs, ws)
        self.update_stats['term'] += np.mean(terminals)
        qvals = self.model.qvals([states, gs, ws])[0]
        target_qvals = self.model.targetqvals([states, gs, ws])[0]
        actionProbs = softmax(qvals, axis=1, theta=10)

        i = 0
        for nStepExpe in nStepExpes:
            nStepExpe[0]['g'], nStepExpe[0]['w'] = gs[i], ws[i]
            for exp in nStepExpe:
                exp['q'] = qvals[i]
                exp['tq'] = target_qvals[i]
                exp['pi'] = actionProbs[i]
                i += 1
                exp['reward'] = rewards[i]
                exp['terminal'] = terminals[i]
            end = {'q': qvals[i], 'tq': target_qvals[i], 'pi': actionProbs[i]}
            nStepExpe.append(end)
            i += 1

        return nStepExpes

    def getTargetsSumTD(self, nStepExpes):
        targets = []
        states = []
        actions = []
        goals = []
        weights = []
        ros = []
        for nStepExpe in nStepExpes:
            tdErrors = []
            cs = []
            for exp0, exp1 in zip(nStepExpe[:-1], nStepExpe[1:]):

                b = np.sum(np.multiply(exp1['pi'], exp1['tq']), keepdims=True)
                b = exp0['reward'] + (1 - exp0['terminal']) * self._gamma * b
                #TODO influence target clipping
                b = np.clip(b, self.wrapper.rNotTerm / (1 - self._gamma), self.wrapper.rTerm)
                tdErrors.append((b - exp0['q'][exp0['a0']]).squeeze())

                ### Calcul des ratios variable selon la méthode
                if self.IS == 'no':
                    cs.append(self._gamma * self._lambda)
                elif self.IS == 'standard':
                    ro = exp0['pi'][exp0['a0']] / exp0['mu0']
                    cs.append(ro * self._gamma * self._lambda)
                elif self.IS == 'retrace':
                    ro = exp0['pi'][exp0['a0']] / exp0['mu0']
                    cs.append(min(1, ro) * self._gamma * self._lambda)
                elif self.IS == 'tb':
                    cs.append(exp0['pi'][exp0['a0']] * self._gamma * self._lambda)
                else:
                    raise RuntimeError

            cs[0] = 1
            exp = nStepExpe[0]
            states.append(exp['s0'])
            actions.append(exp['a0'])
            goals.append(exp['g'])
            weights.append(exp['w'])
            ros.append(np.mean(cs))
            delta = np.sum(np.multiply(tdErrors, np.cumprod(cs)))
            targets.append(exp['q'][exp['a0']] + delta)

        res = [np.vstack(x) for x in [states, actions, goals, weights, targets, ros]]
        return res