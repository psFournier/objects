class Player(object):
    def __init__(self, agent):
        self.agent = agent
        self.reward = 0
        self.name = 'player'
        self.stat_steps = 0

    def play(self, object, goal_selector, action_selector):
        goal = goal_selector.select(object)
        transitions = [[] for o in range(self.agent.env.nbObjects)]
        r = 0
        self.agent.env.reset()
        episodes = 1
        for _ in range(self.agent.env_steps):
            fullstate0 = self.agent.env.state
            states0 = [self.agent.wrapper.get_state(o, fullstate0) for o in range(self.agent.env.nbObjects)]
            state0 = states0[object]
            action, probs = action_selector.select(state0, goal)
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
                r += rs[0]
                if ts[0]:
                    self.agent.env.reset()
                    goal = goal_selector.select(object)
                    episodes += 1
        self.reward += r
        self.stat_steps += episodes
        return transitions, r/episodes

    @property
    def stats(self):
        d = {'reward': self.reward / self.stat_steps}
        self.reward = 0
        self.stat_steps = 0
        return d