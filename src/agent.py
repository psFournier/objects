import numpy as np
from prioritizedReplayBuffer import ReplayBuffer
from env_wrappers.registration import make
from docopt import docopt
from dqn import Controller
from approximators import Predictor
from logger import Logger, build_logger
from evaluators import Qval_evaluator, TDerror_evaluator, ApproxError_buffer_evaluator, \
    ApproxError_global_evaluator, ApproxError_objects_evaluator
from objectSelectors import EXP4, RandomObjectSelector
from goalSelectors import Uniform_goal_selector, Buffer_goal_selector
from actionSelectors import Random_action_selector, NN_action_selector
from utils import softmax
from env_wrappers.registration import register

class Agent(object):
    def __init__(self, args, env, wrapper, model, loggers):
        self.env = env
        self.wrapper = wrapper
        self.model = model
        self.buffer = ReplayBuffer(limit=int(5e5), N=self.wrapper.env.nbObjects)
        self.loggers = loggers
        self.env_step = 0
        self.train_step = 0
        self.train_stats = {'rho': 0, 'loss': 0, 'target_mean': 0, 'qval': 0, 'mean_reward': 0, 'tderror': 0}
        self.env_stats = {}

        self._gamma = 0.99
        self._lambda = float(args['--lambda'])
        self.nstep = int(args['--nstep'])
        self.IS = args['--IS']
        self.batch_size = 64
        self.env_steps = 50
        self.train_steps = 100
        self.episodes = 1000
        self.log_freq = 10

    def play(self, object, goal_selector, action_selector):
        self.env.reset()
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

    def bootstrap(self, action_selector):

        bootstrap_goal_selector = Uniform_goal_selector(self.env.nbFeatures)

        for object in range(self.env.nbObjects):

            self.play(object=object,
                      goal_selector=bootstrap_goal_selector,
                      action_selector=action_selector)

    def train(self, object):

        for _ in range(self.train_steps):
            exps = self.buffer.sample(self.batch_size, object)
            stats = self.model.train(exps)
            self.train_step += 1
            for key, val in stats.items():
                self.train_stats[key] += val

    def log(self, stats):

        for logger in self.loggers:
            logger.logkv('env_step', self.env_step)
            logger.logkv('train_step', self.train_step)
            for name, val in self.env_stats.items():
                logger.logkv(name, val / (self.env_steps*self.log_freq))
            for name, val in self.train_stats.items():
                logger.logkv(name, val / (self.train_steps*self.log_freq))
            for name, val in stats.items():
                logger.logkv(name, val)
            logger.dumpkvs()
        for stats in [self.env_stats, self.train_stats]:
            for name, val in stats.items():
                stats[name] = 0

    def learn1(self, object_selector, goal_selector, action_selector):

        self.bootstrap(action_selector)

        for ep in range(self.episodes):

            object_selector.evaluate()

            object = np.random.choice(object_selector.K, p=object_selector.probs)

            self.play(object, goal_selector, action_selector)

            self.train(object)

            object_selector.update(object)

            if ep % self.log_freq == 0:
                self.log(object_selector.stats)

    def learn2(self, object_selector, goal_selector, action_selector):

        for ep in range(self.episodes):

            object = np.random.randint(self.env.nbObjects)

            self.play(object, goal_selector, action_selector)

            if ep % 100 == 0:
                print(self.env_step)

        for ep in range(self.episodes):

            object_selector.evaluate()

            object = np.random.choice(object_selector.K, p=object_selector.probs)

            self.train(object)

            object_selector.update(object)

            if ep % self.log_freq == 0:
                self.log(object_selector.stats)

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


help = """

Usage: 
  agent.py --env=<ENV> [options]

Options:
  --log_dir DIR            Logging directory [default: /home/pierre/PycharmProjects/objects/log/local/]
  --initq VAL              [default: -100]
  --layers VAL             [default: 128,128]
  --her VAL                [default: 0]
  --nstep VAL              [default: 1]
  --alpha VAL              [default: 0]
  --IS VAL                 [default: no]
  --targetClip VAL         [default: 0]
  --lambda VAL             [default: 0]
  --nbObjects VAL          [default: 4]
  --nbFeatures VAL         [default: 3]
  --nbDependences VAL      [default: 2]
  --evaluator VAL          [default: approxglobal]
  --objectselector VAL     [default: rndobject]
  --exp4gamma VAL          [default: 0.2]
  --exp4beta VAL           [default: 3]
  --exp4eta VAL            [default: 0.2]
  --goalselector VAL       [default: unigoal]
  --actionselector VAL     [default: rndaction]
  
"""

if __name__ == '__main__':

    args = docopt(help)
    log_dir = build_logger(args)
    loggerTB = Logger(dir=log_dir,
                      format_strs=['TB'])
    loggerStdoutJSON = Logger(dir=log_dir,
                        format_strs=['json', 'stdout'])

    nbObjects = int(args['--nbObjects'])
    nbFeatures = int(args['--nbFeatures'])
    nbDependences = int(args['--nbDependences'])

    register(
        id='Objects-v0',
        entry_point='environments:Objects',
        kwargs={'nbObjects': nbObjects,
                'nbFeatures': nbFeatures,
                'nbDependences': nbDependences},
        wrapper_entry_point='env_wrappers.objects:Objects'
    )

    env, wrapper = make(args['--env'], args)
    # model = Controller(wrapper, nstep=1, _gamma=0.99, _lambda=0, IS='no')
    model = Predictor(wrapper)
    agent = Agent(args, env, wrapper, model, [loggerTB, loggerStdoutJSON])

    evaluators = {
        'tderror': TDerror_evaluator(agent.buffer, agent.model, agent.wrapper),
        'approxbuffer': ApproxError_buffer_evaluator(agent.buffer, agent.model),
        'approxglobal': ApproxError_global_evaluator(env, agent.model),
        'approxobject': ApproxError_objects_evaluator(env, agent.model)
    }
    evaluator = evaluators[args['--evaluator']]

    object_selectors = {
        'exp4object': EXP4(K=env.nbObjects,
                     evaluator=evaluator,
                     gamma=float(args['--exp4gamma']),
                     beta=float(args['--exp4beta']),
                     eta=float(args['--exp4eta'])),
        'rndobject': RandomObjectSelector(env.nbObjects, evaluator)
    }

    goal_selectors = {
        'buffergoal': Buffer_goal_selector(agent.buffer),
        'unigoal': Uniform_goal_selector(env.nbFeatures)
    }

    action_selectors = {
        'rndaction': Random_action_selector(env.nbActions)
    }

    object_selector = object_selectors[args['--objectselector']]
    goal_selector = goal_selectors[args['--goalselector']]
    action_selector = action_selectors[args['--actionselector']]

    agent.learn2(object_selector=object_selector,
                goal_selector=goal_selector,
                action_selector=action_selector)