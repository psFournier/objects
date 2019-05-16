import numpy as np

class Player(object):
    def __init__(self, agent):
        self.agent = agent
        self.rewards = agent.env_steps * agent.wrapper.rNotTerm * np.ones(agent.env.nbObjects)
        self.tderrors = np.zeros(agent.env.nbObjects)
        self.name = 'player'
        self.stat_steps = np.zeros(agent.env.nbObjects)
        # self.to_reinit = np.ones(agent.env.nbObjects)

    def play(self, object, goal_selector, action_selector):
        self.agent.env.reset()
        goal = goal_selector.select(object)
        # print(self.name, goal)
        transitions = [[] for o in range(self.agent.env.nbObjects)]
        r = 0.
        tderror = 0.
        lastqval = None
        episodes = 1
        for _ in range(self.agent.env_steps):
            fullstate0 = self.agent.env.state
            states0 = [self.agent.wrapper.get_state(o, fullstate0) for o in range(self.agent.env.nbObjects)]
            state0 = states0[object]
            # print(self.name, state0)
            action, qvals, probs = action_selector.select(state0, goal)
            mu0 = probs[action]
            self.agent.env.step(action)
            fullstate1 = self.agent.env.state
            states1 = [self.agent.wrapper.get_state(o, fullstate1) for o in range(self.agent.env.nbObjects)]
            for o in range(self.agent.env.nbObjects):
                transition = {'s0': states0[o],
                              'a0': action,
                              's1': states1[o],
                              'g': goal,
                              'mu0': mu0,
                              'object': o,
                              'next': None}
                transitions[o].append(transition)
            if goal.size == self.agent.wrapper.goal_dim:
                rs, ts = self.agent.wrapper.get_r(states1[object], goal)
                # print(self.name, rs, ts)
                r += rs[0]
                if lastqval is not None:
                    tderror += (lastqval - rs[0] - self.agent.model._gamma * max(qvals))**2
                lastqval = qvals[action]
                if ts[0]:
                    # print('reward')
                    self.agent.env.reset()
                    goal = goal_selector.select(object)
                    episodes += 1
        # if self.to_reinit[object]:
        #     self.stat_steps[object] = 0
        #     self.rewards[object] = 0
        #     self.tderrors[object] = 0
        #     self.to_reinit[object] = 0
        if self.stat_steps[object] == 0:
            self.rewards[object] = r
            self.tderrors[object] = tderror
        else:
            self.rewards[object] += r
            self.tderrors[object] += tderror
        self.stat_steps[object] += episodes

        # print(object, r, episodes)
        # print('reward', self.rewards[0])
        # print('ep', self.stat_steps[0])
        return transitions, r/episodes

    def stats(self):
        d = {}
        for i, s in enumerate(self.stat_steps):
            if s !=0:
                self.rewards[i] /= s
                self.tderrors[i] /= s
                # self.to_reinit[i] = 1
            self.stat_steps[i] = 0
        d['rewards'] = self.rewards
        d['tderrors'] = self.tderrors
        return d