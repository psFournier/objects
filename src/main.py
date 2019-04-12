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
from agent import Agent

help = """

Usage: 
  main.py --env=<ENV> [options]

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