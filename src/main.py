import numpy as np
from prioritizedReplayBuffer import ReplayBuffer
from env_wrappers.registration import make
from docopt import docopt
from dqn import Controller
from approximators import Predictor
from logger import Logger, build_logger
from evaluators import Qval_evaluator, TDerror_evaluator, ApproxError_buffer_evaluator, \
    ApproxError_global_evaluator, ApproxError_objects_evaluator, ApproxError_changes_evaluator
# from objectSelectors import EXP4, RandomObjectSelector, EXP4SSP
from goalSelectors import Uniform_goal_selector, Buffer_goal_selector, No_goal_selector,Constant_goal_selector
from actionSelectors import Random_action_selector, State_goal_soft_action_selector, State_action_selector, Epsilon_greedy_action_selector
from utils import softmax
from env_wrappers.registration import register
from agent import Agent
from agentImit import AgentImit
from agentOff import AgentOff
from exp4 import EXP4, Uniform_object_selector, EXP3
from experts import Reached_states_variance_maximizer_expert, Uniform_expert, LP_expert, Object_expert
from evaluators import Reached_states_variance_evaluator, Reached_goals_variance_evaluator, Test_episode_evaluator, Train_episode_evaluator, Test_generalisation_one_evaluator, Test_generalisation_all_evaluator, Test_generalisation
from players import Player

help = """

Usage: 
  main.py --env=<ENV> [options]

Options:
  --log_dir DIR            Logging directory [default: /home/pierre/PycharmProjects/objects/log/local/]
  --initq VAL              [default: -100]
  --layers VAL             [default: 32,32]
  --her VAL                [default: 0]
  --nstep VAL              [default: 1]
  --alpha VAL              [default: 0]
  --IS VAL                 [default: no]
  --targetClip VAL         [default: 0]
  --lambda VAL             [default: 0]
  --nbObjects VAL          [default: 10]
  --evaluator VAL          [default: geneone]
  --objects VAL            [default: uni]
  --expgamma VAL          [default: 0.1]
  --expbeta VAL           [default: 5]
  --expeta VAL            [default: 0.1]
  --goals VAL              [default: uniform]
  --actions VAL            [default: sgsoft]
  --dropout VAL            [default: 1]
  --l2reg VAL              [default: 0]
  --episodes VAL     [default: 5000]
  --episodeSteps VAL    [default: 50]
  --agentEta VAL     [default: 0.1]
  --seed SEED              [default: 1]
  --experts VAL            [default: uni,lp]
  --nbObjectsTrain VAL     [default: 1]
  --globalEval VAL         [default: tderror]
  
"""

if __name__ == '__main__':

    args = docopt(help)
    log_dir = build_logger(args)
    loggerTB = Logger(dir=log_dir,
                      format_strs=['TB'])
    loggerStdoutJSON = Logger(dir=log_dir,
                              format_strs=['json', 'stdout'])

    env, wrapper = make(args['--env'], args)
    env.set_objects(n=int(args['--nbObjects']))

    # model = Predictor(wrapper, layers=np.array([int(l) for l in args['--layers'].split(',')]),
    #                   dropout=float(args['--dropout']), l2reg=float(args['--l2reg']))
    agent = Agent(args, env, wrapper, [loggerTB, loggerStdoutJSON])

    agent.model = Controller(agent,
                       nstep=1,
                       _gamma=0.99,
                       _lambda=0,
                       IS='no',
                       layers=np.array([int(l) for l in args['--layers'].split(',')]),
                       dropout=float(args['--dropout']),
                       l2reg=float(args['--l2reg']))

    # experts_dict = {
    #     'uni': Uniform_expert,
    #     'rsv': Reached_states_variance_maximizer_expert,
    #     'lp': LP_expert
    # }
    # experts = {'uni': Uniform_expert(agent)}
    # for eta in [0.001,0.01,0.1]:
    #     for beta in [0.1, 1, 5]:
    #         for maxlen in [50]:
    #             experts['_'.join(['lp', str(eta), str(beta), str(maxlen)])] = LP_expert(agent, eta, beta, maxlen)
    # experts = {name: experts_dict[name](agent) for name in args['--experts'].split(',')}
    experts = {}
    for object in range(env.nbObjects):
        experts['obj_'+str(object)] = Object_expert(agent, object)

    goal_selectors = {
        'buffer': Buffer_goal_selector,
        'uniform': Uniform_goal_selector,
        'no': No_goal_selector,
        'constant': Constant_goal_selector
    }
    action_selectors = {
        'rnd': Random_action_selector,
        'sgsoft': State_goal_soft_action_selector,
        's': State_action_selector,
        'eg': Epsilon_greedy_action_selector
    }
    evaluators = {
        'geneone': Test_generalisation_one_evaluator,
        'geneall': Test_generalisation_all_evaluator,
        'generalization': Test_generalisation
    }
    object_selectors = {
        'exp4': EXP4,
        'uni': Uniform_object_selector,
        'exp3': EXP3
    }
    global_evaluators = {
        'tderror': TDerror_evaluator
    }
    agent.experts = experts
    agent.object_selector = object_selectors[args['--objects']](agent)
    agent.goal_selector = goal_selectors[args['--goals']](agent)
    agent.action_selector = action_selectors[args['--actions']](agent)
    agent.player = Player(agent)
    agent.evaluator = evaluators[args['--evaluator']](agent)
    agent.global_evaluator = global_evaluators[args['--globalEval']](agent)
    agent.learn()
