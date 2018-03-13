# 太阳帆运动模型搭建
import numpy as np


class Env:

    def __init__(self):
        self.t = None
        self.state = None
        self.delta_t = 0.01
        self.constant = {'beta': 0.05, 'r0': 1, 'v_r0': 1, 'theta0': 0}
        self.constant['v_theta0'] = 1 / np.sqrt(self.constant['r0'])
        self.reset()
        self.state_dim = len(self.state)
        self.action_dim = 1

        self.abound = np.array([-np.pi, np.pi])

    def render(self):
        pass

    def reset(self):
        self.t = 0
        self.state = np.array(self.constant['r0', 'theta0', 'v_r0', 'v_theta0'])  # [r theta v_r v_theta]
        return self.state.copy()

    def step(self, action):
        # 当前状态的参数值
        _r, _theta, _v_r, _v_theta = self.state
        # 求微分
        r_dot = _v_r
        theta_dot = _v_theta / _r
        v_r_dot = self.constant['beta'] * ((np.cos(action)) ** 3) / (_r ** 2) + (_v_theta ** 2) / _r - 1 / (_r ** 2)
        v_theta_dot = self.constant['beta'] * np.sin(action) * (np.cos(action) ** 2) / (_r ** 2) - _v_r * _v_theta / _r
        # 下一个状态
        self.state += self.delta_t * np.array([r_dot, theta_dot, v_r_dot, v_theta_dot])
        # 判断是否结束

        # 设计reward函数

        return self.state.copy(), reward, done, info
