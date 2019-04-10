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
        self.buffer = ReplayBuffer(limit=int(5e5), N=self.wrapper.env.nbObjects)
        self.batch_size = 64
        self.train_step = 0
        self.update_stats = {'offpolicyness': 0, 'loss': 0, 'target': 0, 'qval': 0, 'utility': 0, 'mean_r': 0, 'tderror':0, 'update': 0, 'min_prob':0, 'max_prob':0, 'regret':0}
        self.env_stats = {'reward'}
        self.epsilon = 0.2
        self.eta = 0.2
        self.beta = 3
        self.nb_attempts = 10
        self.experts_weights = np.array([1,1])
        self.LPs = np.zeros(self.wrapper.env.nbObjects)
        uni = 1/self.wrapper.env.nbObjects * np.ones(self.wrapper.env.nbObjects)
        self.expert_probs = np.vstack([softmax(self.LPs, theta=self.beta),
                                       uni])

    def learn(self):
        for object in range(self.wrapper.env.nbObjects):
            self.wrapper.env.reset()
            goal = np.random.uniform(-1, 1, self.wrapper.env.nbFeatures)
            for _ in range(100):
                state0 = self.wrapper.get_state(object)
                qvals = self.model.compute_qvals(state0, goal)
                probs = softmax(qvals, theta=1)
                action = np.random.choice(self.wrapper.env.nbActions, p=probs)
                mu0 = probs[action]
                self.wrapper.step((object, action))
                state1 = self.wrapper.get_state(object)
                transition = {'s0': state0,
                              'a0': action,
                              's1': state1,
                              'g': goal,
                              'mu0': mu0,
                              'object': object,
                              'next': None}
                self.buffer.append(transition)

        for ep in range(1000):
            weights_sum = np.sum(self.experts_weights)
            probs = (1 - self.epsilon) * np.dot(self.experts_weights, self.expert_probs) / weights_sum
            probs += self.epsilon / self.wrapper.env.nbObjects
            object = np.random.choice(self.wrapper.env.nbObjects, p=probs)
            learning_progress = self.play(object)
            for i, w in enumerate(self.experts_weights):
                a = self.epsilon * learning_progress * self.expert_probs[i][object]
                b = self.wrapper.env.nbObjects * probs[object]
                self.experts_weights[i] *= np.exp(a/b)
            self.LPs[object] += self.eta * (learning_progress - self.LPs[object])
            self.expert_probs[0] = softmax(self.LPs - min(self.LPs), theta=self.beta)

    def play(self, object):

        transitions_for_eval = self.buffer.sample(100 * self.wrapper.env.nbObjects)
        states = np.vstack([t['s0'] for t in transitions_for_eval])
        goals = np.vstack([t['g'] for t in transitions_for_eval])
        avg_qvals_before_play = np.mean(self.model.compute_qvals(states, goals))

        for _ in range(self.nb_attempts):
            self.wrapper.env.reset()
            rnd_exp_from_object = self.buffer.sample(1, object)
            goal = rnd_exp_from_object[0]['s1']
            trajectory = []
            for _ in range(100):
                state0 = self.wrapper.get_state(object)
                qvals = self.model.compute_qvals(state0, goal)
                probs = softmax(qvals, theta=1)
                self.update_stats['min_prob'] += min(probs)
                self.update_stats['max_prob'] += max(probs)
                action = np.random.choice(self.wrapper.env.nbActions, p=probs)
                mu0 = probs[action]
                self.wrapper.step((object, action))
                state1 = self.wrapper.get_state(object)
                transition = {'s0': state0,
                              'a0': action,
                              's1': state1,
                              'g': goal,
                              'mu0': mu0,
                              'object': object,
                              'next': None}
                self.buffer.append(transition)
                #TODO check everything is fine here

        for _ in range(100 * self.nb_attempts):
            exps = self.buffer.sample(self.batch_size, object)
            nStepExpes = self.getNStepSequences(exps)
            nStepExpes = self.getQvaluesAndBootstraps(nStepExpes)
            states, actions, goals, targets, ros = self.getTargetsSumTD(nStepExpes)
            inputs = [states, actions, goals, targets]
            loss, qval, td_errors = self.model.train(inputs)
            self.train_step += 1
            self.update_stats['update'] += 1
            self.update_stats['target'] += np.mean(targets)
            self.update_stats['offpolicyness'] += np.mean(ros)
            self.update_stats['loss'] += loss
            self.update_stats['qval'] += np.mean(qval)
            self.update_stats['tderror'] += np.mean(td_errors)
            self.model.target_train()

        avg_qvals_after_play = np.mean(self.model.compute_qvals(states, goals))

        learning_progress = avg_qvals_after_play - avg_qvals_before_play

        return learning_progress

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

    def getQvaluesAndBootstraps(self, nStepExpes):

        states, goals = [], []
        for nStepExpe in nStepExpes:
            first = nStepExpe[0]
            l = len(nStepExpe)
            g = first['g']
            goals += [g] * (l+1)
            for exp in nStepExpe:
                states.append(exp['s0'])
            states.append(nStepExpe[-1]['s1'])

        states = np.vstack(states)
        goals = np.vstack(goals)

        rewards, terminals = self.wrapper.get_r(states, goals)
        self.update_stats['mean_r'] += np.mean(rewards)
        qvals = self.model.compute_qvals(states, goals)
        target_qvals = self.model.compute_target_qvals(states, goals)
        actionProbs = softmax(qvals, axis=1, theta=10)

        i = 0
        for nStepExpe in nStepExpes:
            nStepExpe[0]['g'] = goals[i]
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
            ros.append(np.mean(cs))
            delta = np.sum(np.multiply(tdErrors, np.cumprod(cs)))
            targets.append(exp['q'][exp['a0']] + delta)

        res = [np.vstack(x) for x in [states, actions, goals, targets, ros]]
        return res