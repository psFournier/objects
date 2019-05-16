import numpy as np
from prioritizedReplayBuffer import ReplayBuffer
from goalSelectors import Uniform_goal_selector, Buffer_goal_selector, No_goal_selector
from actionSelectors import Random_action_selector

class Agent(object):
    def __init__(self, args, env, wrapper, loggers):
        self.env = env
        self.wrapper = wrapper
        self.model = None
        self.buffer = ReplayBuffer(limit=int(1e5), N=self.env.nbObjects)
        self.goal_selector = None
        self.action_selector = None
        self.object_selector = None
        self.evaluators = None
        self.experts = None
        self.player = None
        self.loggers = loggers
        self.env_steps = np.zeros(env.nbObjects)
        self.train_steps = np.zeros(env.nbObjects)
        self.last_ep = []

        self.batch_size = 64
        self.ep_env_steps = 50
        self.ep_train_steps = 50
        self.random_play_episodes = int(args['--rndepisodes'])
        self.episodes = int(args['--episodes'])
        self.log_freq = 10
        self.her = int(args['--her'])

        self.last_play_reward = -self.ep_env_steps

    def log(self):
        # Careful: stats are reinitialized when called
        stats = {module.name: module.stats() for module in [self.goal_selector,
                                                          self.action_selector,
                                                          self.object_selector,
                                                          self.player,
                                                          self.model] + self.evaluators + list(self.experts.values())}
        for logger in self.loggers:
            logger.logkv('trainstep', sum(self.train_steps))
            logger.logkv('envstep', sum(self.env_steps))
            logger.logkv('trainsteps', self.train_steps)
            logger.logkv('envsteps', self.env_steps)
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
                g = on_policy_transitions[idx]['s1'][self.wrapper.goal_idxs]
                if idx >= i and g != on_policy_transitions[0]['s0'][self.wrapper.goal_idxs]:
                    tr_her = tr.copy()
                    tr_her['g'] = g
                    self.buffer.append(tr_her)

    def memorize(self, object, transitions):
        on_policy_transitions = transitions[object]
        for i, tr in enumerate(on_policy_transitions):
            # if i == length - 1:
            #     tr['next'] = None
            # else:
            #     tr['next'] = on_policy_transitions[i + 1]
            self.buffer.append(tr)

    def learn(self):

        for ep in range(self.episodes):

            object = self.object_selector.select()

            if sum(self.env_steps) > 10000:
                self.model.train(object)
            transitions, play_reward = self.player.play(object, self.goal_selector, self.action_selector)
            # print(self.player.reward)
            self.env_steps[object] += self.ep_env_steps
            if self.her == 1:
                self.memorize_her(object, transitions)
            else:
                self.memorize(object, transitions)

            reward = play_reward - self.last_play_reward
            self.last_play_reward = play_reward
            self.object_selector.update_weights(object, reward)
            for expert in self.experts.values():
                expert.update_probs(object, play_reward)
            # print(reward)
            # print(self.object_selector.experts_weights)
            # print(self.object_selector.experts['lp'].probs)

            if ep % self.log_freq == 0 and ep > 0:
                for evaluator in self.evaluators:
                    evaluator.get_reward()
                self.log()

