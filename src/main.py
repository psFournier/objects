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
from goalSelectors import Uniform_goal_selector, Buffer_goal_selector
from actionSelectors import Random_action_selector, NN_action_selector
from utils import softmax
from env_wrappers.registration import register
from agent import Agent
from exp4 import EXP4
from experts import Reached_states_variance_maximizer_expert, Uniform_expert
from evaluators import State_variance_evaluator

help = """

Usage: 
  main.py --env=<ENV> [options]

Options:
  --log_dir DIR            Logging directory [default: /home/pierre/PycharmProjects/objects/log/local/]
  --initq VAL              [default: -100]
  --layers VAL             [default: 32]
  --her VAL                [default: 0]
  --nstep VAL              [default: 1]
  --alpha VAL              [default: 0]
  --IS VAL                 [default: no]
  --targetClip VAL         [default: 0]
  --lambda VAL             [default: 0]
  --nbObjects VAL          [default: 2]
  --nbFeatures VAL         [default: 3]
  --nbActions VAL          [default: 5]
  --density VAL            [default: 0.5]
  --evaluator VAL          [default: approxglobal]
  --objects VAL     [default: rndobject]
  --exp4gamma VAL          [default: 0.1]
  --exp4beta VAL           [default: 5]
  --exp4eta VAL            [default: 0.1]
  --goals VAL       [default: unigoal]
  --actions VAL     [default: rndaction]
  --dropout VAL            [default: 1]
  --l2reg VAL              [default: 0]
  --train_episodes VAL     [default: 1000]
  --seed SEED              Random seed
  
"""

if __name__ == '__main__':

    args = docopt(help)
    log_dir = build_logger(args)
    loggerTB = Logger(dir=log_dir,
                      format_strs=['TB'])
    loggerStdoutJSON = Logger(dir=log_dir,
                              format_strs=['json', 'stdout'])

    register(
        id='Objects-v0',
        entry_point='environments:Objects',
        kwargs={'seed': int(args['--seed']),
                'nbObjects': int(args['--nbObjects']),
                'nbFeatures': int(args['--nbFeatures']),
                'nbActions': int(args['--nbActions']),
                'density': float(args['--density'])},
        wrapper_entry_point='env_wrappers.objects:Objects'
    )

    env, wrapper = make(args['--env'], args)
    model = Controller(wrapper,
                       nstep=1,
                       _gamma=0.99,
                       _lambda=0,
                       IS='no',
                       layers=np.array([int(l) for l in args['--layers'].split(',')]),
                       dropout=float(args['--dropout']),
                       l2reg=float(args['--l2reg']))
    # model = Predictor(wrapper, layers=np.array([int(l) for l in args['--layers'].split(',')]),
    #                   dropout=float(args['--dropout']), l2reg=float(args['--l2reg']))
    agent = Agent(args, env, wrapper, model, [loggerTB, loggerStdoutJSON])
    # evaluators = {
    #     'tderror': TDerror_evaluator(agent.buffer, agent.model, agent.wrapper),
    #     'approxbuffer': ApproxError_buffer_evaluator(agent.buffer, agent.model),
    #     'approxglobal': ApproxError_global_evaluator(env, agent.model),
    #     'approxobject': ApproxError_objects_evaluator(env, agent.model),
    #     'approxchange': ApproxError_changes_evaluator(env, agent.model)
    # }
    # evaluators = {
    #     'tderror': TDerror_evaluator,
    #     'approxbuffer': ApproxError_buffer_evaluator,
    #     'approxglobal': ApproxError_global_evaluator,
    #     'approxobject': ApproxError_objects_evaluator,
    #     'approxchange': ApproxError_changes_evaluator
    # }
    # evaluator = evaluators[args['--evaluator']](env, agent.model)
    #
    # object_selectors = {
    #     'exp4object': EXP4(K=env.nbObjects,
    #                        evaluator=evaluator,
    #                        gamma=float(args['--exp4gamma']),
    #                        beta=float(args['--exp4beta']),
    #                        eta=float(args['--exp4eta'])),
    #     'rndobject': RandomObjectSelector(env.nbObjects, evaluator, eta=float(args['--exp4eta'])),
    #     'exp4ssp': EXP4SSP(K=env.nbObjects,
    #                        evaluator=evaluator,
    #                        gamma=float(args['--exp4gamma']),
    #                        beta=float(args['--exp4beta']),
    #                        eta=float(args['--exp4eta']))
    # }
    # object_selector = object_selectors[args['--objectselector']]
    experts = [
        Reached_states_variance_maximizer_expert(agent),
        Uniform_expert(agent)
    ]
    object_selector = EXP4(experts=experts,
                           K=env.nbObjects,
                           gamma=float(args['--exp4gamma']),
                           beta=float(args['--exp4beta']))
    goal_selectors = {
        'buffer': Buffer_goal_selector,
        'uniform': Uniform_goal_selector
    }
    action_selectors = {
        'rnd': Random_action_selector,
        'nn': NN_action_selector,
    }
    evaluator = State_variance_evaluator(agent)
    goal_selector = goal_selectors[args['--goals']](agent)
    action_selector = action_selectors[args['--actions']](agent)
    agent.learn3(object_selector=object_selector,
                 goal_selector=goal_selector,
                 action_selector=action_selector,
                 evaluator=evaluator)
