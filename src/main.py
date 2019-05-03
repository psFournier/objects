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
from goalSelectors import Uniform_goal_selector, Buffer_goal_selector, No_goal_selector
from actionSelectors import Random_action_selector, State_goal_action_selector, State_action_selector
from utils import softmax
from env_wrappers.registration import register
from agent import Agent
from exp4 import EXP4, Uniform_object_selector
from experts import Reached_states_variance_maximizer_expert, Uniform_expert
from evaluators import Reached_states_variance_evaluator, Reached_goals_variance_evaluator, Reward_evaluator, Test_episode_evaluator

help = """

Usage: 
  main.py --env=<ENV> [options]

Options:
  --log_dir DIR            Logging directory [default: /home/pierre/PycharmProjects/objects/log/local/]
  --initq VAL              [default: 0]
  --layers VAL             [default: 128,128]
  --her VAL                [default: 0]
  --nstep VAL              [default: 1]
  --alpha VAL              [default: 0]
  --IS VAL                 [default: no]
  --targetClip VAL         [default: 0]
  --lambda VAL             [default: 0]
  --nbObjects VAL          [default: 1]
  --nbFeatures VAL         [default: 3]
  --nbActions VAL          [default: 10]
  --density VAL            [default: 0.1]
  --amplitude VAL            [default: 0.1]
  --evaluator VAL          [default: approxglobal]
  --objects VAL     [default: rndobject]
  --exp4gamma VAL          [default: 0.1]
  --exp4beta VAL           [default: 5]
  --exp4eta VAL            [default: 0.1]
  --goals VAL       [default: unigoal]
  --actions VAL     [default: rndaction]
  --dropout VAL            [default: 1]
  --l2reg VAL              [default: 0]
  --episodes VAL     [default: 200]
  --rndepisodes VAL     [default: 20]
  --seed SEED              Random seed
  --experts VAL            [default: uni]
  
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
        entry_point='environments:ObjectsId',
        kwargs={'seed': int(args['--seed']),
                'nbObjects': int(args['--nbObjects']),
                'nbFeatures': int(args['--nbFeatures']),
                'nbActions': int(args['--nbActions']),
                'density': float(args['--density']),
                'amplitude': float(args['--amplitude'])},
        wrapper_entry_point='env_wrappers.objectsOneGoal:ObjectsOneGoal'
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
    experts_dict = {
        'uni': Uniform_expert,
        'rsv': Reached_states_variance_maximizer_expert
    }
    experts_names = [name for name in args['--experts'].split(',')]
    experts = [experts_dict[name](agent) for name in experts_names]
    goal_selectors = {
        'buffer': Buffer_goal_selector,
        'uniform': Uniform_goal_selector,
        'no': No_goal_selector
    }
    action_selectors = {
        'rnd': Random_action_selector,
        'sg': State_goal_action_selector,
        's': State_action_selector
    }
    evaluators = [Reward_evaluator(agent), Test_episode_evaluator(agent)]
    # agent.object_selector = EXP4(experts=experts,
    #                              K=env.nbObjects,
    #                              gamma=float(args['--exp4gamma']),
    #                              beta=float(args['--exp4beta']))
    agent.object_selector = Uniform_object_selector(K=3)
    agent.goal_selector = goal_selectors[args['--goals']](agent)
    agent.action_selector = action_selectors[args['--actions']](agent)
    agent.learn3(evaluators=evaluators)
