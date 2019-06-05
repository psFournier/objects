import numpy as np

class Player(object):
    def __init__(self, agent):
        self.agent = agent
        self.rewards = agent.ep_env_steps * agent.wrapper.rNotTerm * np.ones(agent.env.nbObjects)
        # self.tderrors = np.zeros(agent.env.nbObjects)
        self.name = 'player'
        self.stat_steps = np.zeros(agent.env.nbObjects)
        # self.to_reinit = np.ones(agent.env.nbObjects)

    def play(self, object, goal_selector, action_selector):

        avgs = self.agent.env.objects[object].avgs
        spans = self.agent.env.objects[object].spans
        idxs = self.agent.wrapper.goal_idxs

        self.agent.env.reset()
        goal = goal_selector.select(object)
        goal_norm = (goal - avgs[idxs]) / spans[idxs]

        transitions = [[]]
        r = 0.
        t = False
        episodes = 1

        for _ in range(self.agent.ep_env_steps):
            if t:
                self.agent.env.reset()
                goal = goal_selector.select(object)
                goal_norm = (goal - avgs[idxs]) / spans[idxs]
                episodes += 1
                transitions.append([])
            fullstate0 = self.agent.env.state
            states0 = [self.agent.wrapper.get_state(o, fullstate0) for o in range(self.agent.env.nbObjects)]
            state0 = states0[object]
            state0_norm = (state0 - avgs) / spans
            action, qvals, probs = action_selector.select(state0_norm, goal_norm)
            mu0 = probs[action]
            self.agent.env.step(action)
            fullstate1 = self.agent.env.state
            states1 = [self.agent.wrapper.get_state(o, fullstate1) for o in range(self.agent.env.nbObjects)]
            state1 = states1[object]
            state1_norm = (state1 - avgs) / spans
            rs, ts = self.agent.wrapper.get_r(state1_norm, goal_norm)
            r += rs[0]
            t = ts[0]
            # print(object, goal, state1, ts[0])
            transition = {'s0': state0_norm,
                          'a0': action,
                          's1': state1_norm,
                          'g': goal_norm,
                          'mu0': mu0,
                          'object': object,
                          'next': None,
                          'reward': rs[0],
                          'terminal': t}
            transitions[-1].append(transition)


        if self.stat_steps[object] == 0:
            self.rewards[object] = r
        else:
            self.rewards[object] += r
        self.stat_steps[object] += episodes

        return transitions, r/episodes

    def stats(self):
        d = {}
        for i, s in enumerate(self.stat_steps):
            if s !=0:
                self.rewards[i] /= s
                # self.tderrors[i] /= s
            self.stat_steps[i] = 0
        d['rewards'] = self.rewards
        # d['tderrors'] = self.tderrors
        return d