import gym
import math
from gym import error, spaces, utils
from gym.utils import seeding

import numpy as np
from os import path

class UnicycleEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30
    }

    def __init__(self):
    	#采样时间
    	self.dt = .05
    	self.steps = 0

    	#设置初始位置
        self.x = 20.
        self.y = 20.
        self.psi = 0.

        #这三个量表示状态
        self.u = 0.
        self.v = 0.
        self.r = 0.

        #这些都是无人艇自身的动力学参数
        self.mu = 2376.5
        self.mv = 3949.5
        self.mr = 3350.
        self.mur = 1799.5
        self.Xuu = -35.4
        self.Xvv = -128.4
        self.Yv = -346.
        self.Yvv = -667.
        self.Nv = -686.
        self.Nvv = 443.
        self.Nr = -1427.

        self.viewer = None

        #观察量包括哪些，范围依次是 待定
        obs_lim = np.array([],dtype=np.float32)

        #动作量包括propulsion和torque,范围是 待定
        act_lim = np.array([propulsion_lim, torque_lim],dtype=np.float32)


        #high = np.array([1., 1., self.max_speed], dtype=np.float32)

        self.action_space = spaces.Box(
            low=-act_lim,
            high=act_lim,
            dtype=np.float32
        )
        
        self.observation_space = spaces.Box(
            low=-obs_lim,
            high=obs_lim,
            dtype=np.float32
        )

        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    #这个函数应该是传入action，然后完成一步的计算。action应该是一个二维的np数组，包含了propulsion和torque。
    #假定action = [propulsion torque]
    def step(self, action):
    	self.steps++
        propulsion = action[0]
        torque = action[1]
        #计算加速度
        udot = (propulsion + self.Xuu * self.u * self.u + self.Xvv * self.v * self.v) / self.mu
        vdot = (- self.mur * self.u * self.r + self.Yv * self.u * self.v + self.Yvv * self.v * abs(self.v)) / self.mv
        rdot = (torque + self.Nv * self.u * self.v + self.Nvv * self.v * abs(self.v) + self.Nr * self.u * self.r) / self.mr
        #计算加速度
        self.u += udot * self.dt
        self.v += vdot * self.dt
        self.r += rdot * self.dt
        #计算最终的位置
        self.psi += self.r * self.dt
        self.x = self.x + self.dt * (self.u * math.cos(self.psi) - self.v * math.sin(self.psi))
        self.y = self.y + self.dt * (self.u * math.sin(self.psi) + self.v * math.cos(self.psi))

        #定义cost function？
        costs = 


        #状态应该包括哪些东西呢？？？
        self.state = np.array([newth, newthdot])
        #返回的是状态，
        return self._get_obs(), costs, self.steps < 50000, {}



    #重置
    def reset(self):
    	#重置无人艇的位置
        self.x = 20.
        self.y = 20.
        self.psi = 0.
        #归零无人艇的速度
        self.u = 0.
        self.v = 0.
        self.r = 0.
        return self._get_obs()


    #返回观测量
    def _get_obs(self):
    	#self.state这个状态应该怎么定义？
    	#在无人艇的问题中应该如何定义这个state。
        theta, thetadot = self.state
        return np.array([np.cos(theta), np.sin(theta), thetadot])


    #完全不会这个
    def render(self, mode='human'):
        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(500, 500)
            self.viewer.set_bounds(-2.2, 2.2, -2.2, 2.2)
            rod = rendering.make_capsule(1, .2)
            rod.set_color(.8, .3, .3)
            self.pole_transform = rendering.Transform()
            rod.add_attr(self.pole_transform)
            self.viewer.add_geom(rod)
            axle = rendering.make_circle(.05)
            axle.set_color(0, 0, 0)
            self.viewer.add_geom(axle)
            fname = path.join(path.dirname(__file__), "assets/clockwise.png")
            self.img = rendering.Image(fname, 1., 1.)
            self.imgtrans = rendering.Transform()
            self.img.add_attr(self.imgtrans)

        self.viewer.add_onetime(self.img)
        self.pole_transform.set_rotation(self.state[0] + np.pi / 2)
        if self.last_u:
            self.imgtrans.scale = (-self.last_u / 2, np.abs(self.last_u) / 2)

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None


def angle_normalize(x):
    return (((x + np.pi) % (2 * np.pi)) - np.pi)