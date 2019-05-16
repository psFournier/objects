"""
http://incompleteideas.net/sutton/MountainCar/MountainCar1.cp
permalink: https://perma.cc/6Z2N-PFWC
"""

import math
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np

class MountainCarEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30
    }

    def __init__(self):
        self.min_position = -2
        self.max_position = 2
        self.max_speed = 0.07
        self.goal_position_1 = math.pi / 6
        self.goal_position_2 = -math.pi / 2
        self.nbFeatures = 2
        self.nbActions = 3
        self.nbObjects = 1


        self.low = np.array([self.min_position, -self.max_speed])
        self.high = np.array([self.max_position, self.max_speed])

        self.viewer = None

        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(self.low, self.high)

        self.seed()
        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))

        position, velocity = self.state
        velocity += (action-1)*0.001 + math.cos(3*position)*(-0.0025)
        velocity = np.clip(velocity, -self.max_speed, self.max_speed)
        position += velocity
        position = np.clip(position, self.min_position, self.max_position)
        if (position==self.min_position and velocity<0): velocity = 0

        # done = bool(position >= self.goal_position)
        # reward = -1.0

        self.state = np.array([position, velocity])
        return self.state, -1, False, {}

    def reset(self):
        self.state = np.array([self.np_random.uniform(low=-0.6, high=-0.4), 0])
        return self.state

    def _height(self, xs):
        return np.sin(3 * xs)*.45+.55

    def render(self, mode='human'):
        screen_width = 600
        screen_height = 400

        world_width = self.max_position - self.min_position
        scale = screen_width/world_width
        carwidth=40
        carheight=20


        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)
            xs = np.linspace(self.min_position, self.max_position, 100)
            ys = self._height(xs)
            xys = list(zip((xs-self.min_position)*scale, ys*scale))

            self.track = rendering.make_polyline(xys)
            self.track.set_linewidth(4)
            self.viewer.add_geom(self.track)

            clearance = 10

            l,r,t,b = -carwidth/2, carwidth/2, carheight, 0
            car = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
            car.add_attr(rendering.Transform(translation=(0, clearance)))
            self.cartrans = rendering.Transform()
            car.add_attr(self.cartrans)
            self.viewer.add_geom(car)
            frontwheel = rendering.make_circle(carheight/2.5)
            frontwheel.set_color(.5, .5, .5)
            frontwheel.add_attr(rendering.Transform(translation=(carwidth/4,clearance)))
            frontwheel.add_attr(self.cartrans)
            self.viewer.add_geom(frontwheel)
            backwheel = rendering.make_circle(carheight/2.5)
            backwheel.add_attr(rendering.Transform(translation=(-carwidth/4,clearance)))
            backwheel.add_attr(self.cartrans)
            backwheel.set_color(.5, .5, .5)
            self.viewer.add_geom(backwheel)

            flagx_1 = (self.goal_position_1-self.min_position)*scale
            flagy1_1 = self._height(self.goal_position_1)*scale
            flagy2_1 = flagy1_1 + 50
            flagpole_1 = rendering.Line((flagx_1, flagy1_1), (flagx_1, flagy2_1))
            self.viewer.add_geom(flagpole_1)
            flag_1 = rendering.FilledPolygon([(flagx_1, flagy2_1), (flagx_1, flagy2_1-10), (flagx_1+25, flagy2_1-5)])
            flag_1.set_color(.8,.8,0)
            self.viewer.add_geom(flag_1)

            flagx_2 = (self.goal_position_2 - self.min_position) * scale
            flagy1_2 = self._height(self.goal_position_2) * scale
            flagy2_2 = flagy1_2 + 50
            flagpole_2 = rendering.Line((flagx_2, flagy1_2), (flagx_2, flagy2_2))
            self.viewer.add_geom(flagpole_2)
            flag_2 = rendering.FilledPolygon(
                [(flagx_2, flagy2_2), (flagx_2, flagy2_2 - 10), (flagx_2 + 25, flagy2_2 - 5)])
            flag_2.set_color(.8, .8, 0)
            self.viewer.add_geom(flag_2)


        pos = self.state[0]
        self.cartrans.set_translation((pos-self.min_position)*scale, self._height(pos)*scale)
        self.cartrans.set_rotation(math.cos(3 * pos))

        return self.viewer.render(return_rgb_array = mode=='rgb_array')

    def close(self):
        if self.viewer: self.viewer.close()