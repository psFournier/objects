import numpy as np
from prioritizedReplayBuffer import ReplayBuffer
from goalSelectors import Uniform_goal_selector, Buffer_goal_selector, No_goal_selector
from actionSelectors import Random_action_selector
import time

class Agent():
    def __init__(self, args, env, wrapper, loggers):
        self.env = env
        self.wrapper = wrapper
        self.model = None
        self.buffer = ReplayBuffer(limit=int(1e5), N=self.env.nbObjects)
        self.goal_selector = None
        self.action_selector = None
        self.global_evaluator = None
        self.object_selector = None
        self.evaluator = None
        self.experts = None
        self.player = None
        self.loggers = loggers
        self.env_steps = np.zeros(env.nbObjects)
        self.train_steps = np.zeros(env.nbObjects)
        self.last_ep = []

        self.batch_size = 64
        self.ep_env_steps = int(args['--episodeSteps'])
        self.ep_train_steps = int(args['--episodeSteps'])
        self.episodes = int(args['--episodes'])
        self.log_freq = 10
        self.her = int(args['--her'])
        self.args = args
        self.eta = float(self.args['--agentEta'])

        # self.last_play_rewards = [[self.ep_env_steps * wrapper.rNotTerm] for _ in range(env.nbObjects)]
        # self.progresses = [0] * env.nbObjects
        self.last_obj_reward = 0
        self.progress_reservoir = [0]
        self.low_r, self.high_r = -1, 1
        self.name = 'agent'

    def log(self):
        # Careful: stats are reinitialized when called
        stats = {module.name: module.stats() for module in [self,
                                                            self.goal_selector,
                                                            self.action_selector,
                                                            self.object_selector,
                                                            self.player,
                                                            self.model,
                                                            self.evaluator]}
        for logger in self.loggers:
            logger.logkv('trainstep', sum(self.train_steps))
            logger.logkv('envstep', sum(self.env_steps))
            logger.logkv('trainsteps', self.train_steps)
            logger.logkv('envsteps', self.env_steps)
            for name, stat in stats.items():
                for key, val in stat.items():
                    logger.logkv(name+'_'+key, val)
            logger.dumpkvs()

    def memorize_her(self, transitions):
        for ep in transitions:
            length = len(ep)
            nb_her_samples = min(self.her-1, length - 1)
            her_goals_idx = np.random.choice(length - 1, nb_her_samples, replace=False)
            her_goals_idx= np.append(her_goals_idx, length - 1)
            for i, tr in enumerate(ep):
                # if tr['g'].size == self.wrapper.goal_dim:
                    # if i == length - 1:
                    #     tr['next'] = None
                    # else:
                    #     tr['next'] = transitions[i + 1]
                self.buffer.append(tr)
                for j, idx in enumerate(her_goals_idx):
                    g = ep[idx]['s1'][self.wrapper.goal_idxs]
                    if idx >= i and np.any(g != ep[0]['s0'][self.wrapper.goal_idxs]):
                        tr_her = tr.copy()
                        tr_her['g'] = g
                        self.buffer.append(tr_her)

    def memorize(self, transitions):
        for ep in transitions:
            for j, tr in enumerate(ep):
                # if i == length - 1:
                #     tr['next'] = None
                # else:
                #     tr['next'] = on_policy_transitions[i + 1]
                self.buffer.append(tr)

    def learn(self):

        for ep in range(self.episodes):

            if sum(self.env_steps) > 10000:
                obj = self.env.objects[self.object_selector.select()]
                for _ in range(self.ep_train_steps):
                    exps = self.buffer.sample(self.batch_size, obj.nb)
                    if exps:
                        loss_train_before, qval_train_before = self.model.train(exps, obj.nb)
                        self.train_steps[obj.nb] += 1

                # obj_reward = self.evaluator.get_reward()

                # #Paragraph for curriculum building
                # progress = obj_reward - self.last_obj_reward
                # self.last_obj_reward = obj_reward
                # if len(self.progress_reservoir) < 100:
                #     self.progress_reservoir.append(progress)
                #     self.low_progress, self.high_progress = np.quantile(self.progress_reservoir, q=[0.2, 0.8])
                # else:
                #     idx = np.random.randint(ep)
                #     if idx < 100:
                #         self.progress_reservoir[idx] = progress
                #         self.low_progress, self.high_progress = np.quantile(self.progress_reservoir, q=[0.2,0.8])
                # if self.high_progress != self.low_progress:
                #     progress = 2 * (np.clip(progress, self.low_progress, self.high_progress) - self.low_progress) /\
                #         (self.high_progress - self.low_progress) - 1
                # else:
                #     progress = np.clip(progress, -1, 1)
                # self.object_selector.update_weights(obj.nb, progress)


            obj = self.env.objects[self.object_selector.select()]
            transitions, play_reward = self.player.play(obj, self.goal_selector, self.action_selector)
            self.env_steps[obj.nb] += self.ep_env_steps
            if self.her != 0:
                self.memorize_her(transitions)
            else:
                self.memorize(transitions)

            if ep % self.log_freq == 0 and ep > 0:
                # for evaluator in self.evaluators:
                self.evaluator.get_reward()
                self.log()

    def stats(self):
        d = {}
        self.time = 0
        return d


