import numpy as np
from prioritizedReplayBuffer import ReplayBuffer
from goalSelectors import Uniform_goal_selector, Buffer_goal_selector, No_goal_selector
from actionSelectors import Random_action_selector

class Agent(object):
    def __init__(self, args, env, wrapper, model, loggers):
        self.env = env
        self.wrapper = wrapper
        self.model = model
        self.buffer = ReplayBuffer(limit=int(1e5), N=self.env.nbObjects)
        self.goal_selector = None
        self.action_selector = None
        self.object_selector = None
        self.loggers = loggers
        self.env_step = 0
        self.train_step = 0
        self.last_ep = []
        self.last_eval = None

        self.stats = {}
        for object in range(self.env.nbObjects):
            self.stats['train{}'.format(object)] = {'divide': 0, 'loss':0, 'qval': 0, 'tderror':0, 'rho':0, 'target_mean':0, 'mean_reward':0}
            self.stats['play{}'.format(object)] = {'reward':0, 'divide':0}
        self.stats['train'] = {'loss': 0, 'divide':0, 'qval': 0, 'tderror':0, 'rho':0, 'target_mean':0, 'mean_reward':0}
        self.stats['play'] = {}

        self.batch_size = 64
        self.env_steps = 200
        self.train_steps = 1000
        self.random_play_episodes = int(args['--rndepisodes'])
        self.episodes = int(args['--episodes'])
        self.log_freq = 1

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

    def play(self, object, goal_selector, action_selector):
        goal = goal_selector.select(object)
        transitions = [[] for o in range(self.env.nbObjects)]
        for _ in range(self.env_steps):
            fullstate0 = self.env.state
            states0 = [self.wrapper.get_state(o, fullstate0) for o in range(self.env.nbObjects)]
            state0 = states0[object]
            action, probs = action_selector.select(state0, goal)
            mu0 = probs[action]
            self.env.step(action)
            fullstate1 = self.env.state
            states1 = [self.wrapper.get_state(o, fullstate1) for o in range(self.env.nbObjects)]
            for o in range(self.env.nbObjects):
                transition = {'s0': states0[o],
                              'a0': action,
                              's1': states1[o],
                              'g': goal,
                              'mu0': mu0,
                              'object': o,
                              'next': None}
                transitions[o].append(transition)
            self.env_step += 1
        return transitions

    def memorize1(self, object, transitions):
        reward = 0
        for tr in transitions[object]:
            r, t = self.wrapper.get_r(tr['s1'], tr['g'])
            reward += r[0]
            self.buffer.append(tr)
        self.stats['play{}'.format(object)]['reward'] += reward
        self.stats['play{}'.format(object)]['divide'] += 1

    def memorize2(self, object, transitions):
        rewards = [0 for _ in self.env.objects]
        for o in range(self.env.nbObjects):
            for tr in transitions[o]:
                r, t = self.wrapper.get_r(tr['s1'], tr['g'])
                rewards[o] += r[0]
                self.buffer.append(tr)
        self.stats['play{}'.format(object)]['reward'] += rewards[object]
        self.stats['play{}'.format(object)]['divide'] += 1

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

        if self.buffer._buffers[object]._numsamples > 10*self.batch_size:
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

    def learn3(self, evaluators):

        random_goals = No_goal_selector(self)
        random_actions = Random_action_selector(self)
        for ep in range(self.random_play_episodes):
            object = np.random.randint(self.env.nbObjects)
            self.env.reset()
            transitions = self.play(object, random_goals, random_actions)
            self.memorize1(object, transitions)

        for ep in range(self.episodes):

            object = self.object_selector.select()

            self.train(object)
            self.env.reset()
            transitions = self.play(object, self.goal_selector, self.action_selector)
            self.memorize1(object, transitions)

            for evaluator in evaluators:
                _ = evaluator.get_reward()

            # self.object_selector.update_weights(object, reward)

            if ep % self.log_freq == 0:
                self.stats['objselector'] = self.object_selector.stats
                for evaluator in evaluators:
                    self.stats[evaluator.name + '_eval'] = evaluator.stats
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