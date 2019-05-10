class Player(object):
    def __init__(self, agent):
        self.agent = agent
        self.reward = 0
        self.name = 'player'

    def play(self, object, goal_selector, action_selector):
        goal = goal_selector.select(object)
        transitions = [[] for o in range(self.agent.env.nbObjects)]
        r = 0
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
                r += self.agent.wrapper.get_r(states1[object], goal)[0][0]
        self.reward = r
        return transitions

    @property
    def stats(self):
        return {'reward': self.reward}