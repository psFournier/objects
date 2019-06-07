import numpy as np

class Player():
    def __init__(self, agent):
        self.agent = agent
        self.rewards = agent.ep_env_steps * agent.wrapper.rNotTerm * np.ones(agent.env.nbObjects+1)
        self.name = 'player'
        self.stat_steps = np.zeros(agent.env.nbObjects+1)

    def play(self, obj, goal_selector, action_selector):

        avgs = self.agent.env.avgs
        spans = self.agent.env.spans
        idxs = self.agent.wrapper.goal_idxs

        self.agent.env.reset()
        goal = goal_selector.select(obj)
        goal_norm = (goal - avgs[idxs]) / spans[idxs]

        transitions = [[]]
        r = 0.
        t = False
        episodes = 1

        for _ in range(self.agent.ep_env_steps):
            if t:
                self.agent.env.reset()
                goal = goal_selector.select(obj)
                goal_norm = (goal - avgs[idxs]) / spans[idxs]
                episodes += 1
                transitions.append([])
            state0_norm = (obj.state - avgs) / spans
            action, qvals, probs = action_selector.select(state0_norm, goal_norm)
            mu0 = probs[action]
            obj.step(action)
            state1_norm = (obj.state - avgs) / spans
            rs, ts = self.agent.wrapper.get_r(state1_norm, goal_norm)
            r += rs[0]
            t = ts[0]
            transition = {'s0': state0_norm,
                          'a0': action,
                          's1': state1_norm,
                          'g': goal_norm,
                          'mu0': mu0,
                          'object': obj.nb,
                          'next': None,
                          'reward': rs[0],
                          'terminal': t}
            transitions[-1].append(transition)

        if self.stat_steps[obj.nb] == 0:
            self.rewards[obj.nb] = r
        else:
            self.rewards[obj.nb] += r
        self.stat_steps[obj.nb] += episodes

        return transitions, r/episodes

    def stats(self):
        d = {}
        for i, s in enumerate(self.stat_steps):
            if s !=0:
                self.rewards[i] /= s
            self.stat_steps[i] = 0
        d['rewards'] = self.rewards
        return d