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
        self.evaluators = None
        self.player = None
        self.loggers = loggers
        self.env_step = 0
        self.train_step = 0
        self.last_ep = []
        self.last_eval = 0

        self.batch_size = 64
        self.env_steps = 50
        self.train_steps = 50
        self.random_play_episodes = int(args['--rndepisodes'])
        self.episodes = int(args['--episodes'])
        self.log_freq = 5
        self.her = int(args['--her'])

    def log(self):
        # Careful: stats are reinitialized when called
        stats = {module.name: module.stats for module in [self.goal_selector,
                                                          self.action_selector,
                                                          self.object_selector,
                                                          self.player,
                                                          self.model] + self.evaluators}
        for logger in self.loggers:
            logger.logkv('trainstep', self.train_step)
            logger.logkv('envstep', self.env_step)
            for name, stat in stats.items():
                for key, val in stat.items():
                    logger.logkv(name+'_'+key, val)
            logger.dumpkvs()

    def memorize_her(self, object, transitions):
        on_policy_transitions = transitions[object]
        length = len(on_policy_transitions)
        her_goals_idx = np.random.choice(length - 1, 3)
        her_goals_idx= np.append(her_goals_idx, length - 1)
        for i, tr in enumerate(on_policy_transitions):
            if tr['g'].size == self.wrapper.goal_dim:
                if i == length - 1:
                    tr['next'] = None
                else:
                    tr['next'] = on_policy_transitions[i + 1]
                self.buffer.append(tr)
            for j, idx in enumerate(her_goals_idx):
                if idx >= i:
                    tr_her = tr.copy()
                    tr_her['g'] = on_policy_transitions[idx]['s1'][0:2]
                    self.buffer.append(tr_her)

    def memorize(self, object, transitions):
        on_policy_transitions = transitions[object]
        for i, tr in enumerate(on_policy_transitions):
            # if i == length - 1:
            #     tr['next'] = None
            # else:
            #     tr['next'] = on_policy_transitions[i + 1]
            self.buffer.append(tr)

    def train(self, object):
        if self.buffer._buffers[object]._numsamples > 100 * self.batch_size:
            for _ in range(self.train_steps):
                exps = self.buffer.sample(self.batch_size, object)
                self.model.train(exps)
                self.train_step += 1

    def learn(self):

        # random_goals = No_goal_selector(self)
        # random_actions = Random_action_selector(self)
        # for ep in range(self.random_play_episodes):
        #     object = self.object_selector.select()
        #     transitions,_ = self.player.play(object, random_goals, random_actions)
        #     # print(self.player.reward)
        #     self.env_step += self.env_steps
        #     self.memorize_her(object, transitions)

        for ep in range(self.episodes):

            object = self.object_selector.select()

            self.train(object)
            transitions,_ = self.player.play(object, self.goal_selector, self.action_selector)
            # print(self.player.reward)
            self.env_step += self.env_steps
            if self.her == 1:
                self.memorize_her(object, transitions)
            else:
                self.memorize(object, transitions)

            # self.object_selector.update_weights(object, reward)

            if ep % self.log_freq == 0 and ep > 0:
                for evaluator in self.evaluators:
                    _ = evaluator.get_reward()
                self.log()

