import numpy as np
from prioritizedReplayBuffer import ReplayBuffer
from goalSelectors import Uniform_goal_selector, Buffer_goal_selector, No_goal_selector
from actionSelectors import Random_action_selector

class Agent(object):
    def __init__(self, args, env, wrapper, model, loggers):
        self.env = env
        self.wrapper = wrapper
        self.model = model
        self.buffer = ReplayBuffer(limit=int(1e4), N=self.env.nbObjects)
        self.loggers = loggers
        self.env_step = 0
        self.train_step = 0

        self.stats = {}
        for object in range(self.env.nbObjects):
            self.stats['train{}'.format(object)] = {'divide': 0, 'loss':0, 'qval': 0, 'tderror':0}
            self.stats['play{}'.format(object)] = {}
        self.stats['train'] = {'loss': 0, 'divide':0, 'qval': 0, 'tderror':0}
        self.stats['play'] = {}

        self.batch_size = 64
        self.env_steps = 50
        self.train_steps = 10
        self.random_play_episodes = int(args['--rndepisodes'])
        self.episodes = int(args['--episodes'])
        self.log_freq = 100

    def log(self):
        for logger in self.loggers:
            logger.logkv('trainstep', self.train_step)
            logger.logkv('playstep', self.env_step)
            for name, dic in self.stats.items():
                for key, val in dic.items():
                    if 'divide' in dic.keys():
                        if dic['divide'] != 0 and key != 'divide':
                            logger.logkv(name + '_' + key, val / dic['divide'])
                    else:
                        logger.logkv(name + '_' + key, val)
            logger.dumpkvs()
        for key1, stat in self.stats.items():
            for key2, val in stat.items():
                stat[key2] = 0

    def play1(self, object, goal_selector, action_selector):
        goal = goal_selector.select(object)
        for _ in range(self.env_steps):
            state0 = self.wrapper.get_state(object)
            action, probs = action_selector.select(state0, goal)
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
            self.env_step += 1

    def play2(self, object, goal_selector, action_selector):
        goal = None
        reached = 0
        transitions = []
        for _ in range(self.env_steps):
            if goal is None or reached:
                goal = goal_selector.select(object)
            state0 = self.wrapper.get_state(object)
            action, probs = action_selector.select(state0, goal)
            mu0 = probs[action]
            self.wrapper.step((object, action))
            state1 = self.wrapper.get_state(object)
            r, t = self.wrapper.get_r(state1, goal)
            if r == self.wrapper.rTerm:
                reached = 1
            transition = {'s0': state0,
                          'a0': action,
                          's1': state1,
                          'g': goal,
                          'mu0': mu0,
                          'object': object,
                          'next': None}
            self.buffer.append(transition)
            transitions.append(transition)
            self.env_step += 1
        return transitions

    # def bootstrap(self, action_selector):
    #
    #     bootstrap_goal_selector = Uniform_goal_selector(self.env.nbFeatures)
    #
    #     for object in range(self.env.nbObjects):
    #
    #         self.play(object=object,
    #                   goal_selector=bootstrap_goal_selector,
    #                   action_selector=action_selector)

    def train(self, object):

        if self.buffer._buffers[object]._numsamples > self.train_steps * self.batch_size:
            for _ in range(self.train_steps):
                exps = self.buffer.sample(self.batch_size, object)
                dico = self.model.train(exps)
                self.train_step += 1
                for key, val in dico.items():
                    self.stats['train'][key] += val
                    self.stats['train'+str(object)][key] += val
                self.stats['train']['divide'] += 1
                self.stats['train' + str(object)]['divide'] += 1

    # def learn1(self, object_selector, goal_selector, action_selector):
    #
    #     for ep in range(self.episodes):
    #
    #         object_selector.evaluate()
    #
    #         object = np.random.choice(object_selector.K, p=object_selector.probs)
    #         object_selector.attempts[object] += 1
    #
    #         self.play(object, goal_selector, action_selector)
    #
    #         self.train(object)
    #
    #         object_selector.update(object)
    #
    #         if ep % self.log_freq == 0:
    #             self.stats['objs'] = object_selector.stats
    #             self.log()
    #
    # def learn2(self, object_selector, goal_selector, action_selector):
    #
    #     for ep in range(self.episodes):
    #
    #         object = np.random.randint(self.env.nbObjects)
    #
    #         self.play(object, goal_selector, action_selector)
    #
    #     for ep in range(self.train_episodes):
    #
    #         object_selector.evaluate()
    #
    #         object = np.random.choice(object_selector.K, p=object_selector.probs)
    #         # object = np.random.randint(2)
    #
    #         self.train(object)
    #
    #         object_selector.update(object)
    #         if ep % self.log_freq == 0:
    #             self.stats['objs'] = object_selector.stats
    #             self.log()

    def learn3(self, object_selector, goal_selector, action_selector, evaluator):

        random_goals = No_goal_selector(self)
        random_actions = Random_action_selector(self)
        for ep in range(self.random_play_episodes):
            object = np.random.randint(self.env.nbObjects)
            _ = self.play2(object, random_goals, random_actions)

        for ep in range(self.episodes):

            object_selector.update_probs()
            p = object_selector.get_probs()
            object = np.random.choice(object_selector.K, p=p)
            object_selector.attempts[object] += 1

            self.train(object)
            self.env.reset()
            transitions = self.play2(object, goal_selector, action_selector)

            reward = evaluator.get_reward(transitions)

            object_selector.update_weights(object, reward)

            if ep % self.log_freq == 0:
                self.stats['objselector'] = object_selector.stats
                self.stats['evaluator'] = evaluator.stats
                self.log()

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