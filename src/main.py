import tensorflow as tf
import numpy as np
from env_wrappers.registration import make
from docopt import docopt
from dqn import Dqn
from agent import Agent
import time
import os
from keras.models import load_model
from logger import Logger, build_logger

help = """

Usage: 
  main.py --env=<ENV> [options]

Options:
  --inv_grad YES_NO        Gradient inversion near action limits [default: 1]
  --max_steps VAL          Maximum total steps [default: 800000]
  --object_steps VAL           Maximum episode steps [default: 200]
  --ep_tasks VAL           Maximum episode tasks [default: 1]
  --log_dir DIR            Logging directory [default: /home/pierre/PycharmProjects/offpolicy/log/local/]
  --eval_freq VAL          Logging frequency [default: 2000]
  --margin VAL             Large margin loss margin [default: 1]
  --gamma VAL              Discount factor [default: 0.99]
  --batchsize VAL          Batch size [default: 64]
  --wimit VAL              Weight for imitaiton loss with imitaiton [default: 1]
  --rnd_demo VAL           Amount of stochasticity in the tutor's actions [default: 0]
  --network VAL            network type [default: 0]
  --filter VAL             network type [default: 0]
  --prop_demo VAL             network type [default: 0.02]
  --freq_demo VAL             network type [default: 100000000]
  --lrimit VAL             network type [default: 0.001]
  --rndv VAL               [default: 0]
  --demo VAL               [default: -1]
  --tutoronly VAL          [default: -1]
  --initq VAL              [default: -100]
  --layers VAL             [default: 128,128]
  --her VAL                [default: 0]
  --nstep VAL              [default: 1]
  --alpha VAL              [default: 0]
  --IS VAL                 [default: no]
  --targetClip VAL         [default: 0]
  --lambda VAL             [default: 0]
  --theta_learn VAL        [default: 10]
  --goal_replay VAL        [default: 0]
  --dueling VAL            [default: 0]
"""


if __name__ == '__main__':

    args = docopt(help)
    log_dir = build_logger(args)
    loggerTB = Logger(dir=log_dir,
                      format_strs=['tensorboard_{}'.format(int(args['--eval_freq'])),
                                   'stdout'])
    loggerJSON = Logger(dir=log_dir,
                        format_strs=['json'])

    env, wrapper = make(args['--env'], args)
    dqn = Dqn(wrapper.state_dim, wrapper.action_dim)
    agent = Agent(args, wrapper, dqn)
    agent.learn()

        #     r, term = agent.step(state, r, term)
        #     if len(agent.buffer) > 10000:
        #         agent.train()
        #     R += r
        #     env_step += 1
        #
        #     if env_step % 2000 == 0:
        #         for obj in range(env_test.N):
        #             w=np.zeros(env_test.N)
        #             w[obj]=1
        #             for val in range(1,env_test.L+1):
        #                 g=np.zeros(env_test.N)
        #                 g[obj] = val/env_test.L
        #                 term_test = False
        #                 step_test = 0
        #                 r_test = 0
        #                 state_test = env_test.reset()
        #                 agent_test.reset(state_test, w, g)
        #                 while step_test < EP_STEPS and not term_test:
        #                     a = agent_test.act()
        #                     state, r, term, _ = env_test.step(a)
        #                     r, term = agent_test.step(state, r, term)
        #                     r_test += r
        #                     step_test += 1
        #                 for logger in [loggerJSON, loggerTB]:
        #                     logger.logkv('testR_{}_{}'.format(obj,g), r_test)
        #
        #         for logger in [loggerJSON, loggerTB]:
        #             logger.logkv('step', env_step)
        #             logger.logkv('episode reward', accumulated_reward/n_episodes)
        #             for metric, val in agent.update_stats.items():
        #                 if agent.update_stats['update'] != 0:
        #                     logger.logkv(metric, val/(agent.update_stats['update']))
        #             logger.dumpkvs()
        #         accumulated_reward = 0
        #         wrapper.stats['changes'] = 0
        #         n_episodes = 0
        #         for metric in agent.update_stats.keys():
        #             agent.update_stats[metric] = 0
        #
        # n_episodes += 1
        # agent.end_episode()
        # accumulated_reward += R




    # model = load_model('../log/local/3_Rooms1-v0/20190212112608_490218/log_steps/model')
    # demo = [int(f) for f in args['--demo'].split(',')]
    # imit_steps = int(float(args['--freq_demo']) * float(args['--prop_demo']))


    # Put demo data in buffer
    # state = env_test.reset()
    # task = 0
    # demo_step = 0
    # demo_ep_step = 0
    # exp = {}
    # traj = []
    # prop = float(args['--rnd_demo'])
    # while demo_step < 500:
    #     done = (env_test.objects[task].s == env_test.objects[task].high)
    #     if done or demo_ep_step >= 200:
    #         state = env_test.reset()
    #         task = np.random.choice([0, 1], p=[prop, 1 - prop])
    #         for i, exp in enumerate(reversed(traj)):
    #             if i == 0:
    #                 exp['next'] = None
    #             else:
    #                 exp['next'] = traj[-i]
    #             agent.buffer.append(exp)
    #         demo_ep_step = 0
    #     else:
    #         exp['s0'] = state.copy()
    #         a, _ = env_test.opt_action(task)
    #         a = np.expand_dims(a, axis=1)
    #         exp['a0'], exp['mu0'], exp['origin'] = a, None, np.expand_dims(1, axis=1)
    #         state = env_test.step(a)[0]
    #         exp['s1'] = state.copy()
    #         traj.append(exp.copy())
    #         demo_ep_step += 1
    #         demo_step += 1
    # env_step = 0
    # train_step = 0
    # state = env.reset()
    # exp = wrapper.reset(state)
    # trajectory = []
    # accumulated_reward = 0
    #
    # while env_step < MAX_STEPS:
    #     a, probs = agent.act(exp, theta=max(0, 100*(env_step - 1e4) / 9e4))
    #     state, r, term, _ = env.step(a)
    #     exp['a0'], exp['mu0'], exp['origin'] = np.expand_dims(a, axis=1), probs[a], np.expand_dims(0, axis=1)
    #     exp['s1'], exp['terminal'], exp['reward'] = state.copy(), r, term
    #     exp['reward'], exp['terminal'] = wrapper.get_r(exp['s1'], exp['g'], exp['w'])
    #     env_step += 1
    #     trajectory.append(exp.copy())
    #     exp['s0'] = np.expand_dims(state, axis=0)
    #     accumulated_reward += exp['reward']
    #
    #     if exp['terminal'] or len(trajectory) >= EP_STEPS:
    #         agent.process_trajectory(trajectory)
    #         trajectory.clear()
    #         state = env.reset()
    #         exp = wrapper.reset(state)
    #
    #     if env_step % TRAIN_FREQ == 0:
    #
    #         for logger in [loggerJSON, loggerTB]:
    #             logger.logkv('step', env_step)
    #             logger.logkv('accumulated reward', accumulated_reward)
    #             logger.dumpkvs()
    #
    #         for i in range(2 ** 10):
    #             _ = agent.train_dqn()
    #             train_step += 1



    # env_step = 1
    # episode_step = 0
    # reward_train = 0
    # reward_test = 0
    # trajectory = []
    # state = env.reset()
    # exp = wrapper.reset(state)
    #
    # while env_step < int(args['--max_steps']):
    #
    #     a, probs = agent.act(exp, theta=max(0, 100*(env_step - 1e4) / 9e4))
    #     exp['a0'], exp['mu0'], exp['origin'] = a, probs[a], np.expand_dims(0, axis=1)
    #     state, r, term, info = env.step(a.squeeze())
    #     exp['s1'] = state.copy()
    #     exp['terminal'], exp['reward'] = wrapper.get_r(exp['s1'], exp['goal'], r, term)
    #     env_step += 1
    #     episode_step += 1
    #
    #     reward_train += exp['reward']
    #
    #     trajectory.append(exp.copy())
    #     exp['s0'] = state
    #
    #     if len(agent.buffer) > 10000:
    #         train_stats = agent.train_dqn()
    #         stats['target_mean'] += train_stats['target_mean']
    #         stats['train_step'] += 1
    #         stats['ro'] += train_stats['ro']
    #
    #     if exp['terminal'] or episode_step >= max_episode_steps:
    #         nb_ep_train += 1
    #         stats['reward_train'] += reward_train
    #         nb_ep += 1
    #         agent.process_trajectory(trajectory)
    #         trajectory.clear()
    #         state = env.reset()
    #         exp = wrapper.reset(state)
    #         episode_step = 0
    #         reward_train = 0
    #         reward_test = 0
    #
    #
    #     if env_step % int(args['--eval_freq'])== 0:
    #         # R = 0
    #         # n=10
    #         # for i in range(n):
    #         #     term_eval, ep_step_eval = 0, 0
    #         #     state_eval = env_test.reset()
    #         #     x = np.random.randint(env_test.nR)
    #         #     y = np.random.randint(env_test.nC)
    #         #     goal_eval = np.array(env_test.rescale([x, y]))
    #         #     while not term_eval and ep_step_eval < max_episode_steps:
    #         #         input = [np.expand_dims(i, axis=0) for i in [state_eval, goal_eval]]
    #         #         qvals = agent.qvals(input)[0].squeeze()
    #         #         action = np.argmax(qvals)
    #         #         a = np.expand_dims(action, axis=1)
    #         #         state_eval = env_test.step(a)[0]
    #         #         term_eval, r_eval = wrapper_test.get_r(state_eval, goal_eval)
    #         #         ep_step_eval += 1
    #         #         R += r_eval
    #
    #         # loggerJSON.logkv('goal_freq', stats['goal_freq'])
    #         for logger in [loggerJSON, loggerTB]:
    #             logger.logkv('step', env_step)
    #             logger.logkv('target_mean', stats['target_mean'] / (stats['train_step'] + 1e-5))
    #             logger.logkv('ro', stats['ro'] / (stats['train_step'] + 1e-5))
    #             logger.logkv('reward_train', stats['reward_train'] / (nb_ep_train + 1e-5))
    #             logger.logkv('reward_test', stats['reward_test'] / (nb_ep_test + 1e-5))
    #             logger.dumpkvs()
    #
    #         stats['target_mean'] = 0
    #         stats['ro'] = 0
    #         stats['train_step'] = 0
    #         stats['reward_train'] = 0
    #         nb_ep_train = 0
    #         stats['reward_test'] = 0
    #         nb_ep_test = 0
    #
    #         t1 = time.time()
    #         print(t1- t0)
    #         t0 = t1
    #         agent.model.save(os.path.join(log_dir, 'model'), overwrite=True)







